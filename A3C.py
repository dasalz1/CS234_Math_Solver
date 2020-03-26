import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformer.Models import Transformer, PositionalEncoding
from transformer.bart import BartModel
from transformer.configuration_bart import BartConfig
from transformer.Constants import PAD, EOS
import numpy as np
from torch.distributions import Categorical
from tensorboard_utils import *
from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE


class Policy_Network(nn.Module):
    def __init__(self, num_layers = 2, num_heads = 4, key_dimension = 64, 
                 value_dimension = 64, dropout = 0.1, n_position = 160, 
                 d_char_vec = 512, inner_dimension = 2048, 
                 n_trg_position = MAX_ANSWER_SIZE, n_src_position = MAX_QUESTION_SIZE, padding = 1,
                critic_num_layers=4, critic_kernel_size=4, critic_padding=1, model=None, share_embedding_layers=False, 
                data_parallel=True, use_gpu=True, device='cpu'):
        
        super(Policy_Network, self).__init__()
        
        self.device = device
        bart_config = BartConfig(
            vocab_size=VOCAB_SIZE+1,
            pad_token_id=PAD,
            eos_token_id=EOS,
            d_model=d_char_vec,
            encoder_ffn_dim=inner_dimension,
            encoder_layers=num_layers,
            encoder_attention_heads=num_heads,
            decoder_ffn_dim=inner_dimension,
            decoder_layers=num_layers,
            decoder_attention_heads=num_heads,
            dropout=dropout,
            max_encoder_position_embeddings=n_src_position,
            max_decoder_position_embeddings=n_trg_position
        )

        if data_parallel:
            self.action_transformer = nn.DataParallel(BartModel(bart_config)).to(self.device)#cuda())
        else:
            self.action_transformer = BartModel(bart_config).to(self.device)#cuda()
        # elif use_gpu:
            # self.action_transformer = BartModel(bart_config).cuda()
        # else:
            # self.action_transformer = BartModel(bart_config)

    def loss_op(self, data, op, tb=None, num_iter=None, valid_data=None):
        batch_qs, batch_as = data
        if 'rl' in op:
            policy_losses, value_losses, batch_rewards = self.policy_batch_loss(batch_qs, batch_as, 0.9, device=self.device)

            return policy_losses+value_losses, batch_rewards
        elif 'mle' in op:
            mle_loss, n_correct, n_char = self.mle_batch_loss(batch_qs, batch_as)

            return mle_loss, n_correct/n_char

    def forward(self, src_seq, trg_seq, op='mle'):
        if 'rl' in op:
            action_prob, values = self.action_transformer(input_ids=src_seq, decoder_input_ids=trg_seq, get_value=True)
            action_prob = action_prob[:, -1, :]
            values = values[:, -1, :]
            return action_prob, values
        else:
            action_prob = self.action_transformer(input_ids=src_seq, decoder_input_ids=trg_seq, get_value=False)
            # action_prob = action_prob[:, -1, :]
            return action_prob

    def calc_reward(self, actions_pred, actions, ignore_index=0, sparse_rewards=False):
        # sparse rewards or char rewards
        if sparse_rewards:
            if actions_pred == EOS and actions == EOS:
                return torch.ones_like(actions).cuda().float()
            return torch.zeros_like(actions).cuda().float()
        else:
            # 1 if character is correct
            return (actions_pred==actions).float()

    def get_returns(self, rewards, batch_size, gamma):
        T = rewards.shape[1]
        discounts = torch.tensor(np.logspace(0, T, T, base=gamma, endpoint=False)).view(1, -1).to(self.device)
        all_returns = torch.zeros((batch_size, T)).to(self.device)
    
        for t in range(T):
            temp = (discounts[:, :T-t]*rewards[:, t:]).sum(dim=-1)
            all_returns[:, t] = temp
            (all_returns - all_returns.mean(dim=-1).view(-1, 1)) / (all_returns.std(dim=-1).view(-1, 1) + self.eps)
  
        return all_returns

    def compute_mle_loss(self, pred, target, smoothing, log=False):
        def compute_loss(pred, target, smoothing):
            target = target.contiguous().view(-1)
            if smoothing:
                eps = 0.1
                n_class = pred.size(1)

                one_hot = torch.zeros_like(pred)
                one_hot = one_hot.scatter(1, target.view(-1, 1), 1)
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
                log_prb = F.log_softmax(pred, dim=1)

                non_pad_mask = target.ne(PAD)
                loss = -(one_hot * log_prb).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).sum()  # average later
            else:
                loss = F.cross_entropy(pred, target, ignore_index=PAD, reduction='sum')    
            return loss
    
        loss = compute_loss(pred, target, smoothing)
        pred_max = pred.max(1)[1]
        target = target.contiguous().view(-1)
        non_pad_mask = target.ne(PAD)
        n_correct = pred_max.eq(target)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct

    def mle_batch_loss(self, batch_qs, batch_as):
        trg_as = batch_as[:, 1:]
        # print(batch_qs.shape)
        # print(batch_as.shape)
        pred_logits = self.forward(src_seq=batch_qs, trg_seq=batch_as[:, :-1], op='mle')#(input_ids=batch_qs, decoder_input_ids=batch_as[:, :-1])
        # print(pred_logits.shape)
        pred_logits = pred_logits.reshape(-1, pred_logits.size(-1))
        loss, n_correct = self.compute_mle_loss(pred_logits, trg_as, smoothing=True)

        non_pad_mask = trg_as.ne(PAD)
        n_char = non_pad_mask.sum().item()

        return loss, n_correct, n_char

    def policy_batch_loss(self, batch_qs, batch_as, gamma):
        batch_size, max_len_sequence = batch_qs.shape[0], batch_as.shape[1]
        current_as = batch_as[:, :1]
        complete = torch.ones((batch_size, 1)).to(self.device)
        rewards = torch.zeros((batch_size, 0)).to(self.device)
        values = torch.zeros((batch_size, 0)).to(self.device)
        log_probs = torch.zeros((batch_size, 0)).to(self.device)
        advantages_mask = torch.ones((batch_size, 0)).to(self.device)

        for t in range(1, max_len_sequence):
            advantages_mask = torch.cat((advantages_mask, complete), dim=1)
            action_probs, curr_values = self.forward(src_seq=batch_qs, trg_seq=current_as, op='rl')
            m = Categorical(F.softmax(action_probs, dim=-1))
            actions = m.sample().reshape(-1, 1)

            trg_t = batch_as[:, t].reshape(-1, 1)

            # update decoder output
            current_as = torch.cat((current_as, actions), dim=1)
            curr_log_probs = m.log_prob(actions.contiguous.view(-1)).contiguous().view(-1, 1)
            # calculate reward based on character cross entropy
            curr_rewards = self.calc_reward(actions, trg_t)

            # update terms
            rewards = torch.cat((rewards, curr_rewards), dim=1).to(self.device)
            values = torch.cat((values, curr_values), dim=1).to(self.device)
            log_probs = torch.cat((log_probs, curr_log_probs), dim=1)

            # if the action taken is EOS or if end of sequence trajectory ends
            complete *= (1 - ((actions==EOS) | (trg_t==EOS)).float())
    
        returns = self.get_returns(rewards, batch_size, gamma)
    
        advantages = returns - values
        # advantages = returns
        advantages *= advantages_mask

        policy_losses = (-log_probs * advantages).sum(dim=-1).mean()
        value_losses = F.mse_loss(values, rewards, reduction='mean')
        # batch_rewards = rewards.sum(dim=-1).mean()
        tb_rewards = torch.div(rewards.sum(dim=-1), current_as.ne(PAD).sum(dim=-1)).mean().item()
        # return policy_losses, value_losses, batch_rewards
        return policy_losses, value_losses, tb_rewards