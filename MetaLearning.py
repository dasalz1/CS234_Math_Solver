import torch
import torch.distributed as dist
import torch.nn.functional as F
from A3C import Policy_Network
from A3C import Policy_Network
from torch import nn
from torch import optim
from torch import autograd
from torch.distributions import Categorical
from torch.multiprocessing import Process, Queue
from multiprocessing import Event
import numpy as np
import pandas as pd
import os
from copy import deepcopy
from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE
from dataset import PAD, EOS

PAD_IDX = 0

class Learner(nn.Module):

    def __init__(self, process_id, gpu='cpu', world_size=4, optimizer=optim.Adam, optimizer_sparse=optim.SparseAdam, optim_params=(1e-3, (0.9, 0.995), 1e-8), model_params=None):
        super(Learner, self).__init__()
        print(gpu)
        self.model = Policy_Network(data_parallel=False)
        if process_id == 0:
          optim_params = (self.model.parameters(),) + optim_params
          self.optimizer = optimizer(*optim_params)

        self.meta_optimizer = optim.SGD(self.model.parameters(), 0.03)
        self.process_id = process_id
        self.device='cuda:'+str(process_id) if gpu is not 'cpu' else gpu
        self.model.to(self.device)
        self.num_iter = 0
        self.world_size = world_size
        self.original_state_dict = {}
        self.eps = np.finfo(np.float32).eps.item()
        self.use_ml = False
        self.use_rl = False

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
        self.model.load_state_dict(self.original_state_dict)
        self.model.to(self.device)
        self.model.train()

        self.optimizer.zero_grad()
        dummy_query_x, dummy_query_y = temp_data
        print(" ")
        action_probs = self.model(src_seq=dummy_query_x[0, :], trg_seq=dummy_query_y[0, :1])
        m = Categorical(F.softmax(action_probs, dim=-1))
        actions = m.sample().reshape(-1, 1)
        trg_t = batch_as[0, 1].reshape(-1, 1)
        dummy_loss = -F.cross_entropy(action_probs, trg_t.reshape(-1), ignore_index=0, reduction='none').reshape(-1, 1).sum()
        print(" ")
        hooks = self._hook_grads(all_grads)

        dummy_loss.backward()
        print(" ")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # gpu memory explodes if you dont remove hooks
        for h in hooks:
            h.remove()

        print("finished meta")

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
            action_probs = self.model(src_seq=batch_qs, trg_seq=current_as)
            m = Categorical(F.softmax(action_probs, dim=-1))
            actions = m.sample().reshape(-1, 1)

            trg_t = batch_as[:, t].reshape(-1, 1)

            # update decoder output
            current_as = torch.cat((current_as, actions), dim=1)

            curr_log_probs = -F.cross_entropy(action_probs, trg_t.reshape(-1), ignore_index=0, reduction='none').reshape(-1, 1)

            # calculate reward based on character cross entropy
            curr_rewards = self.calc_reward(actions, trg_t)

            # update terms
            rewards = torch.cat((rewards, curr_rewards), dim=1).to(self.device)
            # values = torch.cat((values, curr_values), dim=1).to(self.device)
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

    def forward(self, num_updates, data_queue, data_event, process_event, tb=None, log_interval=100, checkpoint_interval=10000):
        while(True):
            data_event.wait()
            data = data_queue.get()
            dist.barrier(async_op=True)

            if self.process_id == 0:
                original_state_dict = {}
                data_event.clear()

            if self.process_id == 0 and self.num_iter != 0 and self.num_iter % checkpoint_interval == 0:
                self.save_checkpoint(model, optimizer, self.num_iter)

            # broadcast weights from master process to all others and save them to a detached dictionary for loadinglater
            for k, v in self.model.state_dict().items():
                if self.process_id == 0:
                    self.original_state_dict[k] = v.clone().detach()
            dist.broadcast(v, src=0, async_op=True)

            self.model.to(self.device)
            self.model.train()

        # meta gradients
            support_x, support_y, query_x, query_y = map(lambda x: torch.LongTensor(x).to(self.device), data)
            for i in range(num_updates):
                self.meta_optimizer.zero_grad()
                loss, _ = self.policy_batch_loss(support_x, support_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.meta_optimizer.step()

            loss, rewards = self.policy_batch_loss(query_x, query_y)

            # loss, pred = self.model(query_x, query_y)
            all_grads = autograd.grad(loss, self.model.parameters())

            for idx in range(len(all_grads)):
                dist.reduce(all_grads[idx].data, 0, op=dist.ReduceOp.SUM, async_op=True)
                all_grads[idx] = (all_grads[idx] / self.world_size)

            if self.process_id == 0:
                self.num_iter += 1
                self._write_grads(original_state_dict, temp_grads, (query_x, query_y))
                # finished batch so can load data again from master
                process_event.set()


class MetaTrainer:

    def __init__(self, world_size, device='cpu', model_params=None):
        self.world_size = world_size

        self.meta_learners = [Learner(process_id=process_id, gpu=process_id if device is not 'cpu' else 'cpu', world_size=world_size, model_params=model_params) for process_id in range(world_size)]
        # gpu backend instead of gloo
        self.backend = "gloo"
    
    def init_process(self, process_id, data_queue, data_event, process_event, num_updates, tb, address='localhost', port='29500'):
        os.environ['MASTER_ADDR'] = address
        os.environ['MASTER_PORT'] = port
        dist.init_process_group(self.backend, rank=process_id, world_size=self.world_size)
        self.meta_learners[process_id](num_updates, data_queue, data_event, process_event, tb)


    # dataloaders is list of the iterators of the dataloaders for each task
    def train(self, data_loaders, tb=None, num_updates = 5, num_iters=250000):
        data_queue = Queue()
        # for notifying when to recieve data
        data_event = Event()
        # for notifying this method when to send new data
        process_event = Event()
        # so doesn't hang on first iteration
        process_event.set()
        num_tasks = len(data_loaders)

        processes = []
        for process_id in range(self.world_size):
            processes.append(Process(target=self.init_process, 
                            args=(process_id, data_queue, data_event, 
                            process_event, num_updates, 
                            tb if process_id == 0 else None)))
            processes[-1].start()

        for num_iter in range(num_iters):
            print("num iter:",num_iter)
            process_event.wait()
            process_event.clear()
            tasks = np.random.randint(0, num_tasks, (self.world_size))
            for task in tasks:
                # place holder for sampling data from dataset
                task_data = next(data_loaders[task])
                data_queue.put((task_data[0].numpy()[0], task_data[1].numpy()[0], 
                        task_data[2].numpy()[0], task_data[3].numpy()[0]))
            data_event.set()

        for p in processes:
            p.terminate()
            p.join()