import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE
from dataset import EOS



class Trainer:
    def __init__(self, device='cpu'):
        self.device=device
        self.eps = np.finfo(np.float32).eps.item()
    
    def calc_reward(self, actions_pred, actions, ignore_index=0):
        # 1 if character is correct
        return (actions_pred==actions).float()

    def get_returns(self, rewards, batch_size, gamma):
        T = rewards.shape[1]
        discounts = torch.tensor(np.logspace(0, T, T, base=gamma, endpoint=False)).view(1, -1)
        all_returns = torch.zeros((batch_size, T))
        
        for t in range(T):
            temp = (discounts[:, :T-t]*rewards[:, t:]).sum(dim=-1)
            all_returns[:, t] = temp
            (all_returns - all_returns.mean(dim=-1).view(-1, 1)) / (all_returns.std(dim=-1).view(-1, 1) + self.eps)
    
        return all_returns

    def train_policy_epoch(self, training_data, model, gamma, optimizer):

        model.train()
        ignore_index = model.action_transformer.trg_pad_idx
        
    #     sample batch of questions and answers
        for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):
            batch_qs, batch_as = map(lambda x: x.to(self.device), batch)
            batch_size = batch_qs.shape[0]
            current_as = batch_as[:, :1]
            complete = torch.ones((batch_size, 1))
            rewards = torch.zeros((batch_size, 0))
            values = torch.zeros((batch_size, 0))
            log_probs = torch.zeros((batch_size, 0))
            advantages_mask = torch.ones((batch_size, 0))
            for t in range(1, 5):#MAX_ANSWER_SIZE):
                advantages_mask = torch.cat((advantages_mask, complete), dim=1)
                action_probs, curr_values = model(batch_qs, current_as)
                m = Categorical(F.softmax(action_probs, dim=-1))
                actions = m.sample().contiguous().view(-1, 1)
                
                trg_t = batch_as[:, t].contiguous().view(-1, 1)
                
                # update decoder output
                current_as = torch.cat((current_as, actions), dim=1)
                
                curr_log_probs = -F.cross_entropy(action_probs, trg_t.view(-1), ignore_index=0, reduction='none').contiguous().view(-1, 1)
                
                # calculate reward based on character cross entropy
                curr_rewards = self.calc_reward(actions, trg_t)
                
                # update terms
                rewards = torch.cat((rewards, curr_rewards), dim=1)
                values = torch.cat((values, curr_values), dim=1)
                log_probs = torch.cat((log_probs, curr_log_probs), dim=1)
                
                # if the action taken is EOS or if end of sequence trajectory ends
                complete *= (1 - ((actions==EOS) | (trg_t==EOS)).float())
            

            returns = self.get_returns(rewards, batch_size, gamma)
            
            advantages = returns - values
            advantages *= advantages_mask

            policy_losses = (-log_probs * advantages).sum(dim=-1).mean()

            value_losses = F.mse_loss(values, rewards, reduction='mean')

            optimizer.zero_grad()
            loss = policy_losses + value_losses

            loss.backward()
            optimizer.step()