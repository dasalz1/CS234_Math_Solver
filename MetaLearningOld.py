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
  def __init__(self, process_id, gpu='cpu', world_size=4, optimizer=AdamW, optimizer_sparse=optim.SparseAdam, optim_params=(1e-7,), model_params=None, tb=None):
    super(Learner, self).__init__()
    print(gpu)
    self.model = Policy_Network(data_parallel=False, use_gpu=False if gpu is 'cpu' else True)
    saved_checkpoint = torch.load("./checkpoint-mle.pth")
    model_dict = saved_checkpoint['model']
    for k, v in list(model_dict.items()):
      kn = k.replace('module.', '')
      model_dict[kn] = v
      del model_dict[k]
    self.model.load_state_dict(model_dict, strict=False)
    if process_id == 0:
      optim_params = (self.model.parameters(),) + optim_params
      self.optimizer = optimizer(*optim_params)
    
    self.meta_optimizer = optim.SGD(self.model.parameters(), 1e-5)
    self.process_id = process_id
    self.device='cuda:'+str(process_id) if gpu is not 'cpu' else gpu
    self.model.to(self.device)
    self.num_iter = 0
    self.world_size = world_size
    self.original_state_dict = {}
    self.eps = np.finfo(np.float32).eps.item()
    self.use_ml = False
    self.use_rl = False
    self.trainer = Trainer(self.use_ml, self.use_rl, self.device)
    self.tb = tb

    # if process == 0:
      # optim_params = optim_params.insert(0, self.model_parameters())
      # self.optimizer = optimizer(*optim_params)

  def save_checkpoint(self, model, optimizer, iteration):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),}, "checkpoint-{}.pth".format(iteration))

  def _hook_grads(self, all_grads):
    hooks = []
    for i, v in enumerate(self.model.parameters()):
      def closure():
        ii = i
        return lambda grad: all_grads[ii]
      hooks.append(v.register_hook(closure()))
    return hooks

  def _write_grads(self, original_state_dict, all_grads, temp_data):
    # reload original model before taking meta-gradients
    self.model.load_state_dict(original_state_dict)
    self.model.to(self.device)
    self.model.train()
    self.optimizer.zero_grad()
    dummy_query_x, dummy_query_y = temp_data
    action_probs, values = self.model(src_seq=dummy_query_x, trg_seq=dummy_query_y, use_critic=True)
    m = Categorical(F.softmax(action_probs, dim=-1))
    actions = m.sample().contiguous().view(-1, 1)
    # dummy_loss = -F.cross_entropy(action_probs, trg_t.contiguous().view(-1), ignore_index=0, reduction='none').sum()
    dummy_loss = -m.log_prob(actions.contiguous().view(-1)).contiguous().view(-1, 1).sum() + F.mse_loss(values, torch.zeros_like(values), reduction='mean')
    hooks = self._hook_grads(all_grads)

    dummy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()

    # gpu memory explodes if you dont remove hooks
    for h in hooks:
      h.remove()

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
      action_probs, curr_values = self.model(src_seq=batch_qs, trg_seq=current_as, use_critic=True)
      m = Categorical(F.softmax(action_probs, dim=-1))
      actions = m.sample().contiguous().view(-1, 1)

      trg_t = batch_as[:, t].contiguous().view(-1, 1)

      # update decoder output
      current_as = torch.cat((current_as, actions), dim=1)

      # curr_log_probs = -F.cross_entropy(action_probs, trg_t.contiguous().view(-1), ignore_index=0, reduction='none').contiguous().view(-1, 1)

      curr_log_probs = -m.log_prob(actions.contiguous().view(-1)).contiguous().view(-1, 1)
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

  def forward_singleton(self, num_updates, data, tb=None, checkpoint_interval=5000, free_interval=25):
    
    original_state_dict = {}

    if self.num_iter != 0 and self.num_iter % checkpoint_interval == 0:
      self.save_checkpoint(self.model, self.optimizer, self.num_iter)

    for k, v in self.model.state_dict().items():
      original_state_dict[k] = v.clone().detach()

    self.model.to(self.device)
    self.model.train()
    if self.device != 'cpu' and self.num_iter != 0 and self.num_iter % free_interval == 0: 
      torch.cuda.empty_cache()

    support_x, support_y, query_x, query_y = map(lambda x: torch.LongTensor(x).to(self.device), data)
    for i in range(num_updates):
      self.meta_optimizer.zero_grad()
      loss, _, _ = self.policy_batch_loss(support_x, support_y)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
      self.meta_optimizer.step()

    loss, rewards, tb_rewards = self.policy_batch_loss(query_x, query_y)
    
    self.trainer.tb_policy_batch(self.tb, tb_rewards, loss, self.num_iter, 0, 1)

    all_grads = autograd.grad(loss, self.model.parameters())
    self._write_grads(original_state_dict, all_grads, (query_x, query_y))
    self.num_iter += 1

  def forward(self, num_updates, data_queue, data_event, process_event, tb=None, log_interval=100, checkpoint_interval=5000):
    data_event.wait()
    try:
      while(True):
        data = data_queue.get()

        if data is None: 
          if self.processes_id == 0: self.save_checkpoint(self.model, self.optimizer, 'final_model')
          break
        dist.barrier(async_op=True)

        if self.process_id == 0:
          original_state_dict = {}
          data_event.clear()

        if self.process_id == 0 and self.num_iter != 0 and self.num_iter % checkpoint_interval == 0:
          self.save_checkpoint(self.model, self.optimizer, self.num_iter)

        # broadcast weights from master process to all others and save them to a detached dictionary for loadinglater
        for k, v in self.model.state_dict().items():
          if self.process_id == 0:
            original_state_dict[k] = v.clone().detach()
          dist.broadcast(v, src=0, async_op=True)

        self.model.to(self.device)
        self.model.train()
        torch.cuda.empty_cache()

        # meta gradients
        support_x, support_y, query_x, query_y = map(lambda x: torch.LongTensor(x).to(self.device), data)
        for i in range(num_updates):
          self.meta_optimizer.zero_grad()
          loss, _, _ = self.policy_batch_loss(support_x, support_y)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
          self.meta_optimizer.step()

        loss, rewards, tb_rewards = self.policy_batch_loss(query_x, query_y)
        
        if self.process_id == 0:
          self.trainer.tb_policy_batch(self.tb, tb_rewards, loss, self.num_iter, 0, 1)

        # loss, pred = self.model(query_x, query_y)
        all_grads = autograd.grad(loss, self.model.parameters())

        for idx in range(len(all_grads)):
          dist.reduce(all_grads[idx].data, 0, op=dist.ReduceOp.SUM, async_op=True)
          all_grads[idx].data = all_grads[idx].data / self.world_size

        if self.process_id == 0:
          self._write_grads(original_state_dict, all_grads, (query_x, query_y))
          # finished batch so can load data again from master
          process_event.set()

        self.num_iter += 1
        data_event.wait()
    except KeyboardInterrupt:
      if self.process_id == 0: self.save_checkpoint(self.model, self.optimizer, 'final_model')


  # def forward(self, x_data, y_data):
    # pass#self.policy_batch_loss(x_data, y_data)


class MetaTrainerSingleton:

  def __init__(self, device='cpu', model_params=None, tb=None):
    self.meta_learner = Learner(process_id=0, gpu='cpu' if str(device) == 'cpu' else 0, world_size=1, model_params=model_params, tb=tb)
    self.device=device

  def train(self, data_loader, num_updates=5, tb=None, num_iterations=250000):
    for num_iter in tqdm(range(num_iterations), mininterval=2, leave=False):
      curr_data = data_loader.get_sample()
      self.meta_learner.forward_singleton(num_updates, curr_data, tb)

class MetaTrainer:

  def __init__(self, world_size, device='cpu', model_params=None, tb=None):
    self.world_size = world_size

    self.meta_learners = [Learner(process_id=process_id, gpu=process_id if device is not 'cpu' else 'cpu', world_size=world_size, model_params=model_params, tb=tb) for process_id in range(world_size)]
    # gpu backend instead of gloo
    self.backend = "nccl"
    
  def init_process(self, process_id, data_queue, data_event, process_event, num_updates, tb=None, address='localhost', port='29500'):
    os.environ['MASTER_ADDR'] = address
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(self.backend, rank=process_id, world_size=self.world_size)
    self.meta_learners[process_id](num_updates, data_queue, data_event, process_event)


  # dataloaders is list of the iterators of the dataloaders for each task
  def train(self, data_loader, num_updates=5, tb=None, num_iterations=250000):
    data_queue = Queue()
    # for notifying when to recieve data
    data_event = Event()
    # for notifying this method when to send new data
    process_event = Event()
    # so doesn't hang on first iteration
    process_event.set()
    
    processes = []
    for process_id in range(self.world_size):
      processes.append(Process(target=self.init_process, 
                        args=(process_id, data_queue, data_event, 
                          process_event, num_updates,
                          tb if process_id == 0 else None)))
      processes[-1].start()

    try:
      for num_iter in tqdm(range(num_iterations), mininterval=2, leave=False):
        process_event.wait()
        process_event.clear()
        for task in range(self.world_size):
          task_data = data_loader.get_sample()
          # task_data = (np.random.randint(0, 20000, (1, 45)), np.random.randint(0, 20000, (1, 5)), np.random.randint(0, 20000, (10, 45)), np.random.randint(0, 20000, (10, 5)))
          # place holder for sampling data from dataset
          data_queue.put((task_data[0], task_data[1], 
                  task_data[2], task_data[3]))
        data_event.set()

      for _ in range(self.world_size):
        data_queue.put(None)
    except KeyboardInterrupt:
      pass

    for p in processes:
      p.join()