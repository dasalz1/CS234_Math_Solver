import torch
import torch.distributed as dist
import torch.nn.functional as F
from A3C import Policy_Network
from A3C import Policy_Network
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.multiprocessing import Process, Queue
from multiprocessing import Event
import numpy as np
import pandas as pd
import os
from copy import deepcopy
from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE
from dataset import PAD, EOS
from training import Trainer
from tqdm import tqdm
from transformers import AdamW

PAD_IDX = 0

class Learner(nn.Module):
                        # optim.Adam
  def __init__(self, process_id, gpu='cpu', meta_lr=1e-4, model_params=None, tb=None, checkpoint_path='./checkpoint-mle.pth'):
    super(Learner, self).__init__()
    self.model = Policy_Network(data_parallel=False, use_gpu=False if gpu is 'cpu' else True)
    self.model_pi = Policy_Network(data_parallel=False, use_gpu=False if gpu is 'cpu' else True)

    if checkpoint_path is not '':
      saved_checkpoint = torch.load(checkpoint_path)
      model_dict = saved_checkpoint['model']
      for k, v in list(model_dict.items()):
        kn = k.replace('module.', '')
        model_dict[kn] = v
        del model_dict[k]
    
      # self.model.load_state_dict(model_dict, strict=False)
    
    self.meta_optimizer = optim.SGD(self.model.parameters(), meta_lr)
    self.device='cuda:'+str(process_id) if gpu is not 'cpu' else gpu
    self.model.to(self.device)
    self.model_pi.to(self.device)
    self.model.train()
    self.model_pi.train()
    self.num_iter = 0
    self.eps = np.finfo(np.float32).eps.item()
    self.tb = tb


  def tb_meta_iter(self, batch_rewards, average_value_loss, batch_idx):
    self.tb.add_scalars(
      {
        "batch_average_rewards" : batch_rewards,
        "epoch_value_loss": average_value_loss, 
      },
    group="policy_train",
    sub_group="batch",
    global_step = batch_idx)

  def save_checkpoint(self, model, optimizer, iteration):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),}, "checkpoint-{}.pth".format(iteration))

  def calc_reward(self, actions_pred, actions, ignore_index=0, sparse_rewards=False):
    # sparse rewards or char rewards
    if sparse_rewards:
      if actions_pred == EOS and actions == EOS:
        return torch.ones_like(actions).cuda().float()
      return torch.zeros_like(actions).cuda().float()
    else:
      # 1 if character is correct
      return (actions_pred==actions).float()

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
    pred_logits = self.model(input_ids=batch_qs, decoder_input_ids=batch_as[:, :-1])
    pred_logits = pred_logits.reshape(-1, pred_logits.size(2))
    loss, n_correct = self.compute_mle_loss(pred_logits, trg_as, smoothing=True)
    
    non_pad_mask = trg_as.ne(PAD)
    n_char = non_pad_mask.sum().item()

    return loss, n_correct, n_char
  
  def get_returns(self, rewards, batch_size, gamma):
    T = rewards.shape[1]
    discounts = torch.tensor(np.logspace(0, T, T, base=gamma, endpoint=False)).view(1, -1).to(self.device)
    all_returns = torch.zeros((batch_size, T)).to(self.device)
    
    for t in range(T):
      temp = (discounts[:, :T-t]*rewards[:, t:]).sum(dim=-1)
      all_returns[:, t] = temp
      (all_returns - all_returns.mean(dim=-1).view(-1, 1)) / (all_returns.std(dim=-1).view(-1, 1) + self.eps)
  
    return all_returns

  def policy_batch_loss(self, batch_qs, batch_as, gamma=0.9):
    batch_size, max_len_sequence = batch_qs.shape[0], batch_as.shape[1]
    current_as = batch_as[:, :1]
    complete = torch.ones((batch_size, 1)).to(self.device)
    rewards = torch.zeros((batch_size, 0)).to(self.device)
    values = torch.zeros((batch_size, 0)).to(self.device)
    log_probs = torch.zeros((batch_size, 0)).to(self.device)
    advantages_mask = torch.ones((batch_size, 0)).to(self.device)
    for t in range(1, max_len_sequence):
      advantages_mask = torch.cat((advantages_mask, complete), dim=1)
      # action_probs, curr_values = model(src_seq=batch_qs, trg_seq=current_as)
      action_probs, curr_values = self.model_pi(src_seq=batch_qs, trg_seq=current_as, use_critic=True)
      m = Categorical(F.softmax(action_probs, dim=-1))
      actions = m.sample().contiguous().view(-1, 1)

      trg_t = batch_as[:, t].contiguous().view(-1, 1)

      # update decoder output
      current_as = torch.cat((current_as, actions), dim=1)

      curr_log_probs = m.log_prob(actions.contiguous().view(-1)).contiguous().view(-1, 1)
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

    policy_losses = (-log_probs * advantages).mean(dim=-1).mean()
    value_losses = F.mse_loss(values, rewards, reduction='mean')
    batch_rewards = rewards.sum(dim=-1).mean()
    tb_rewards = torch.div(rewards.sum(dim=-1), current_as.ne(PAD).sum(dim=-1)).mean().item()

    loss = policy_losses + value_losses

    return loss, batch_rewards, tb_rewards

  def forward_temp(self, temp_data):
    dummy_query_x, dummy_query_y = temp_data
    action_probs, values = self.model_pi(src_seq=dummy_query_x, trg_seq=dummy_query_y, use_critic=True)
    m = Categorical(F.softmax(action_probs, dim=-1))
    actions = m.sample().contiguous().view(-1, 1)
    dummy_loss = -m.log_prob(actions.contiguous().view(-1)).contiguous().view(-1, 1).sum() + F.mse_loss(values, torch.zeros_like(values), reduction='mean')
    return dummy_loss

  def forward(self, num_updates, data, use_mle=True, use_rl=False, tb=None, checkpoint_interval=5000, tb_interval=4):

    if self.num_iter != 0 and self.num_iter % checkpoint_interval == 0:
      self.save_checkpoint(self.model_pi, self.optimizer, self.num_iter)

    for copy_param, param in zip(self.model.parameters(), self.model_pi.parameters()):
      param.data.copy_(copy_param.data)

    support_x, support_y, query_x, query_y = map(lambda x: torch.LongTensor(x).to(self.device), data)
    for i in range(num_updates):
      self.meta_optimizer.zero_grad()
      if use_mle:
        loss, n_correct, n_char = self.mle_batch_loss(support_x, support_y)
      elif use_rl:
        loss, _, _ = self.policy_batch_loss(support_x, support_y)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model_pi.parameters(), 1.0)
      self.meta_optimizer.step()

    if use_mle:
      loss, n_correct, n_char = self.mle_batch_loss(support_x, support_y)
      if self.num_iter != 0 and self.num_iter % tb_interval == 0: self.tb_meta_iter(n_correct/n_char, loss, self.num_iter)
    elif use_rl:
      loss, rewards, tb_rewards = self.policy_batch_loss(query_x, query_y)
      if self.num_iter != 0 and self.num_iter % tb_interval == 0: self.tb_meta_iter(tb_rewards, loss, self.num_iter)
    all_grads = autograd.grad(loss, self.model_pi.parameters(), create_graph=True)
    self.num_iter += 1

    return all_grads, (query_x, query_y)

class MetaTrainer:

  def __init__(self, device='cpu', lr=1e-6, meta_lr=1e-4, model_params=None, tb=None, checkpoint_path=None):
    self.meta_learner = Learner(process_id=0, gpu='cpu' if str(device) == 'cpu' else 0, meta_lr=meta_lr, model_params=model_params, tb=tb, checkpoint_path=checkpoint_path)
    self.device=device
    self.optimizer = AdamW(self.meta_learner.model.parameters(), lr)

  def train(self, data_loader, num_updates=5, num_iterations=250000, meta_batch_size=4):
    for num_iter in tqdm(range(int(num_iterations/meta_batch_size)), mininterval=2, leave=False):
      sum_grads = None
      for task in range(meta_batch_size):
        curr_data = data_loader.get_sample()
        curr_grads, temp_data = self.meta_learner(num_updates, curr_data)
        sum_grads = [torch.add(i, j) for i, j in zip(sum_grads, curr_grads)] if sum_grads is not None else curr_grads

      dummy_loss = self.meta_learner.forward_temp(temp_data)
      self._write_grads(sum_grads, dummy_loss)

  def _write_grads(self, all_grads, dummy_loss):
    # reload original model before taking meta-gradients
    self.optimizer.zero_grad()
    hooks = self._hook_grads(all_grads)
    dummy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.meta_learner.model.parameters(), 1.0)
    self.optimizer.step()

    # gpu memory explodes if you dont remove hooks
    for h in hooks:
      h.remove()

  def _hook_grads(self, all_grads):
    hooks = []
    for i, v in enumerate(self.meta_learner.model.parameters()):
      def closure():
        ii = i
        return lambda grad: all_grads[ii]
      hooks.append(v.register_hook(closure()))
    return hooks