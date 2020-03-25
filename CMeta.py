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
  def __init__(self, process_id, gpu='cpu', meta_lr=1e-4, checkpoint_path='./checkpoint-mle.pth'):
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
    
    self.meta_optimizer = optim.SGD(self.model.parameters(), meta_lr)
    self.device='cuda:'+str(process_id) if gpu is not 'cpu' else gpu
    self.model.to(self.device)
    self.model_pi.to(self.device)
    self.model.train()
    self.model_pi.train()
    self.eps = np.finfo(np.float32).eps.item()


  def loss_op(self, data, valid_data, op, num_iter):
    
    for copy_param, param in zip(self.model.parameters(), self.model_pi.parameters()):
      param.data.copy_(copy_param.data)

    for i in range(num_updates):
      self.meta_optimizer.zero_grad()
      loss, acc = self.model_pi.loss_op(data=data, op=op)

      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model_pi.parameters(), 1.0)
      self.meta_optimizer.step()

    loss, acc = self.model_pi.loss_op(data=valid_data, op=op, tb=tb, num_iter=num_iter)

    return loss, acc

  def forward_temp(self, temp_data, op):
    _, _, dummy_query_x, dummy_query_y = temp_data
    dummy_loss, _ = self.model_pi.loss_op(data=(dummy_query_x, dummy_query_y), op=op)#(src_seq=dummy_query_x, trg_seq=dummy_query_y, use_critic=True)
    return dummy_loss

  def write_grads(self, sum_grads, optimizer, dummy_data, op)

      dummy_loss = self.forward_temp(dummy_data, op)
      self._write_grads(sum_grads, dummy_loss, optimizer)

  def _write_grads(self, all_grads, dummy_loss, optimizer):
    # reload original model before taking meta-gradients
    optimizer.zero_grad()
    hooks = self._hook_grads(all_grads)
    dummy_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.optimizer.step()

    # gpu memory explodes if you dont remove hooks
    for h in hooks:
      h.remove()

  def parameters(self):
    return self.model.parameters()

  def _hook_grads(self, all_grads):
    hooks = []
    for i, v in enumerate(self.model.parameters()):
      def closure():
        ii = i
        return lambda grad: all_grads[ii]
      hooks.append(v.register_hook(closure()))
    return hooks