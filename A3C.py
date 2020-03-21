import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformer.Models import Transformer, PositionalEncoding
from transformer.bart import BartModel
from transformer.configuration_bart import BartConfig
from transformer.Constants import PAD, EOS
import numpy as np

from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE


class Policy_Network(nn.Module):
    def __init__(self, num_layers = 2, num_heads = 4, key_dimension = 64, 
                 value_dimension = 64, dropout = 0.1, n_position = 160, 
                 d_char_vec = 512, inner_dimension = 2048, 
                 n_trg_position = MAX_ANSWER_SIZE, n_src_position = MAX_QUESTION_SIZE, padding = 1,
                critic_num_layers=4, critic_kernel_size=4, critic_padding=1, model=None, share_embedding_layers=False, data_parallel=True, use_gpu=True):
        
        super(Policy_Network, self).__init__()
        
        # self.action_transformer = Transformer(n_src_vocab=VOCAB_SIZE + 1, n_trg_vocab=VOCAB_SIZE+1, src_pad_idx=0, trg_pad_idx=0,
        #                        d_char_vec=d_char_vec, d_model=d_char_vec, d_inner=inner_dimension, n_layers=num_layers,
        #                        n_head=num_heads, d_k=key_dimension, d_v=value_dimension, dropout=dropout,
        #                        n_trg_position=n_trg_position, n_src_position=n_src_position,
        #                        trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True) if model == None else model
        
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
            self.action_transformer = nn.DataParallel(BartModel(bart_config).cuda())
        elif use_gpu:
            self.action_transformer = BartModel(bart_config).cuda()
        else:
            self.action_transformer = BartModel(bart_config)

        # if data_parallel:
        #     critic_src_embedding = self.action_transformer.module.shared
        #     critic_trg_embedding = self.action_transformer.module.shared
        #     critic_src_position = self.action_transformer.module.encoder.embed_positions
        #     critic_trg_position = self.action_transformer.module.decoder.embed_positions
        # else:
        #     critic_src_embedding = self.action_transformer.shared
        #     critic_trg_embedding = self.action_transformer.shared
        #     critic_src_position = self.action_transformer.encoder.embed_positions
        #     critic_trg_position = self.action_transformer.decoder.embed_positions

        # if data_parallel:
        #     self.value_head = nn.DataParallel(Critic(conv_layers=critic_num_layers, d_char_vec=d_char_vec, kernel_size=critic_kernel_size,
        #                             n_vocab=VOCAB_SIZE+1, dropout=dropout, padding=critic_padding, 
        #                             src_embedding=critic_src_embedding, 
        #                             trg_embedding=critic_trg_embedding, 
        #                             src_position_enc=critic_src_position, 
        #                             trg_position_enc=critic_trg_position).cuda())
        # else:
        #     self.value_head = Critic(conv_layers=critic_num_layers, d_char_vec=d_char_vec, kernel_size=critic_kernel_size,
        #                             n_vocab=VOCAB_SIZE+1, dropout=dropout, padding=critic_padding, 
        #                             src_embedding=critic_src_embedding, 
        #                             trg_embedding=critic_trg_embedding, 
        #                             src_position_enc=critic_src_position, 
        #                             trg_position_enc=critic_trg_position).cuda()
    def forward(self, src_seq, trg_seq, reg=True, device=None):
        if reg:
            action_prob = self.action_transformer(input_ids=src_seq, decoder_input_ids=trg_seq)
            action_prob = action_prob[:, -1, :]
            # state_values = self.value_head(src_seq, trg_seq)

            # return action_prob, state_values
            return action_prob
        else:
            batch_qs = src_seq; batch_as = trg_seq; gamma=0.9
            batch_size, max_len_sequence = batch_qs.shape[0], batch_as.shape[1]
            current_as = batch_as[:, :1]
            complete = torch.ones((batch_size, 1)).to(device)
            rewards = torch.zeros((batch_size, 0)).to(device)
            # values = torch.zeros((batch_size, 0)).to(self.device)
            log_probs = torch.zeros((batch_size, 0)).to(device)
            advantages_mask = torch.ones((batch_size, 0)).to(device)
            for t in range(1, max_len_sequence):
                advantages_mask = torch.cat((advantages_mask, complete), dim=1)
                # action_probs, curr_values = model(src_seq=batch_qs, trg_seq=current_as)
                action_probs = self.model(src_seq=batch_qs, trg_seq=current_as)
                m = Categorical(F.softmax(action_probs, dim=-1))
                actions = m.sample().contiguous().view(-1, 1)

                trg_t = batch_as[:, t].contiguous().view(-1, 1)

                # update decoder output
                current_as = torch.cat((current_as, actions), dim=1)

                curr_log_probs = -F.cross_entropy(action_probs, trg_t.contiguous().view(-1), ignore_index=0, reduction='none').contiguous().view(-1, 1)

                # calculate reward based on character cross entropy
                curr_rewards = self.calc_reward(actions, trg_t)

                # update terms
                rewards = torch.cat((rewards, curr_rewards), dim=1).to(device)
                log_probs = torch.cat((log_probs, curr_log_probs), dim=1)

                # if the action taken is EOS or if end of sequence trajectory ends
                complete *= (1 - ((actions==EOS) | (trg_t==EOS)).float())
              
            returns = self.get_returns(rewards, batch_size, gamma)

            # advantages = returns - values
            advantages = returns
            advantages *= advantages_mask

            policy_losses = (-log_probs * advantages).sum(dim=-1).mean()
            batch_rewards = rewards.sum(dim=-1).mean()

            return policy_losses, batch_rewards

    def get_returns(self, rewards, batch_size, gamma, device, eps=np.finfo(np.float32).eps.item()):
        T = rewards.shape[1]
        discounts = torch.tensor(np.logspace(0, T, T, base=gamma, endpoint=False)).view(1, -1).to(device)
        all_returns = torch.zeros((batch_size, T)).to(device)
        
        for t in range(T):
            temp = (discounts[:, :T-t]*rewards[:, t:]).sum(dim=-1)
            all_returns[:, t] = temp
            (all_returns - all_returns.mean(dim=-1).view(-1, 1)) / (all_returns.std(dim=-1).view(-1, 1) + eps)
      
        return all_returns

    def calc_reward(self, actions_pred, actions, ignore_index=0, sparse_rewards=False):
        # sparse rewards or char rewards
        if sparse_rewards:
            if actions_pred == EOS and actions == EOS:
                return torch.ones_like(actions).cuda().float()
            return torch.zeros_like(actions).cuda().float()
        else:
            # 1 if character is correct
            return (actions_pred==actions).float()



class Critic(nn.Module):
    
    def __init__(self, conv_layers=4, d_char_vec=512, n_vocab=96, kernel_size=4, dropout=0.1, padding=1, 
                 src_embedding=None, trg_embedding=None, src_position_enc=None, trg_position_enc=None, pad_idx=0):
        
        super(Critic, self).__init__()
        
        self.src_word_emb = src_embedding if src_embedding != None else nn.Embedding(n_vocab, d_char_vec, padding_idx=pad_idx)
        self.trg_word_emb = trg_embedding if trg_embedding != None else nn.Embedding(n_vocab, d_char_vec, padding_idx=pad_idx)
        
        # self.src_position_enc = src_position_enc if src_position_enc != None else PositionalEncoding(d_char_vec, n_position=MAX_QUESTION_SIZE)
        # self.trg_position_enc = trg_position_enc if trg_position_enc != None else PositionalEncoding(d_char_vec, n_position=MAX_ANSWER_SIZE)
        self.dropout = nn.Dropout(p=dropout)
        
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=d_char_vec, out_channels=d_char_vec, kernel_size=kernel_size, padding=padding) for _ in range(conv_layers)])
        self.value_layer = nn.Linear(d_char_vec, 1)
        
    def forward(self, src_seq, trg_seq):
        
        batch_sz, max_src_pos, max_trg_pos = src_seq.shape[0], src_seq.shape[1], trg_seq.shape[1]
        src_seq_pad = torch.cat((src_seq, torch.zeros((batch_sz, MAX_QUESTION_SIZE-max_src_pos), dtype=torch.int64).to(src_seq.device)), dim=1)
        trg_seq_pad = torch.cat((trg_seq, torch.zeros((batch_sz, MAX_ANSWER_SIZE-max_trg_pos), dtype=torch.int64).to(trg_seq.device)), dim=1)
        # src_emb = self.dropout(self.src_position_enc(self.src_word_emb(src_seq_pad)))
        # trg_emb = self.dropout(self.trg_position_enc(self.trg_word_emb(trg_seq_pad)))
        src_emb = self.dropout(self.src_word_emb(src_seq_pad))
        trg_emb = self.dropout(self.trg_word_emb(trg_seq_pad))
        x = torch.cat((src_emb, trg_emb), dim=1)
        conv_out = torch.transpose(x, 1, 2).contiguous()
        for conv in self.conv_layers:
            conv_out = conv(conv_out)
        conv_out = torch.mean(F.relu(conv_out), dim=2)
        output = self.value_layer(conv_out)
        
        return output
