import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformer.Models import Transformer
from torch import nn
from torch import optim
from torch import autograd
from torch.multiprocessing import Process, Queue
from multiprocessing import Event
import numpy as np
import pandas as pd
from utils import save_checkpoint, from_checkpoint_if_exists, tb_mle_meta_batch
import os
from copy import deepcopy

PAD_IDX = 0

class Learner(nn.Module):

	def __init__(self, process_id, gpu='cpu', world_size=4, total_forward=5, optimizer=optim.Adam, optimizer_sparse=optim.SparseAdam, optim_params=(1e-3, (0.9, 0.995), 1e-8), model_params=None):
		super(Learner, self).__init__()

		self.model = Transformer(*model_params)

		if process_id == 0:
			optim_params = (self.model.parameters(),) + optim_params
			self.optimizer = optimizer(*optim_params)
			self.forward_passes = 0

		self.meta_optimizer = optim.SGD(self.model.parameters(), 0.1)
		self.device='cuda:'+str(process_id) if gpu is not 'cpu' else gpu
		self.process_id = process_id
		self.num_iter = 0
		self.world_size = world_size
		self.total_forward = total_forward
		self.original_state_dict = {}


		# if process == 0:
			# optim_params = optim_params.insert(0, self.model_parameters())
			# self.optimizer = optimizer(*optim_params)

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

			  non_pad_mask = target.ne(PAD_IDX)
			  loss = -(one_hot * log_prb).sum(dim=1)
			  loss = loss.masked_select(non_pad_mask).sum()  # average later
			else:
				
			  loss = F.cross_entropy(pred, target, ignore_index=PAD_IDX, reduction='sum')
			return loss
		
		loss = compute_loss(pred, target, smoothing)
		pred_max = pred.max(1)[1]
		target = target.contiguous().view(-1)
		non_pad_mask = target.ne(PAD_IDX)
		n_correct = pred_max.eq(target)
		n_correct = n_correct.masked_select(non_pad_mask).sum().item()
		return loss, n_correct

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
		pred_logits = self.model(dummy_query_x, dummy_query_y[:, :-1])
		pred_logits = pred_logits.contiguous().view(-1, pred_logits.size(2))
		dummy_loss, _ = self.compute_mle_loss(pred_logits, dummy_query_y[:, 1:], smoothing=True)

		# dummy_loss, _ = self.model(temp_data)
		hooks = self._hook_grads(all_grads)

		dummy_loss.backward()

		torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
		
		self.optimizer.step()

		# gpu memory explodes if you dont remove hooks
		for h in hooks:
			h.remove()

		print("finished meta")

	def forward(self, num_updates, data_queue, data_event, process_event, tb=None, log_interval=100, checkpoint_interval=10000):
		while(True):
			data_event.wait()
			data = data_queue.get()
			dist.barrier()
			data_event.clear()

			if self.process_id == 0 and self.num_iter != 0 and self.num_iter % checkpoint_interval == 0:
				save_checkpoint(0, self.model, self.optimizer, suffix=str(self.num_iter))

			# broadcast weights from master process to all others and save them to a detached dictionary for loadinglater
			for k, v in self.model.state_dict().items():
				if self.process_id == 0:
					original_state_dict[k] = v.clone().detach()
				dist.broadcast(v, src=0, async_op=True)

			self.model.to(self.device)
			self.model.train()

			# meta gradients
			support_x, support_y, query_x, query_y = map(lambda x: torch.LongTensor(x).to(self.device), data)
			for i in range(num_updates):
				self.meta_optimizer.zero_grad()
				pred_logits = self.model(support_x, support_y[:, :-1])
				pred_logits = pred_logits.contiguous().view(-1, pred_logits.size(2))
				loss, n_correct = self.compute_mle_loss(pred_logits, support_y[:, 1:], smoothing=True)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				self.meta_optimizer.step()


			pred_logits = self.model(query_x, query_y[:, :-1])
			pred_logits = pred_logits.contiguous().view(-1, pred_logits.size(2))
			loss, n_correct = self.compute_mle_loss(pred_logits, query_y[:, 1:], smoothing=True)

			non_pad_mask = query_y[: 1:].ne(PAD_IDX)
			n_word = non_pad_mask.sum().item()

			acc = torch.FloatTensor([n_correct / n_word])


			# loss, pred = self.model(query_x, query_y)
			all_grads = autograd.grad(loss, self.model.parameters())

			dist.reduce(loss, 0, op=dist.ReduceOp.SUM, async_op=True)
			dist.reduce(acc, 0, op=dist.ReduceOp.SUM)


			for idx in range(len(all_grads)):
				dist.reduce(all_grads[idx].data, 0, op=dist.ReduceOp.SUM, async_op=True)

			if self.process_id == 0 and tb is not None and self.num_iter % log_interval == 0:
				tb_mle_meta_batch(tb, loss.item()/self.world_size, acc/self.world_size, self.num_iter)

			if self.process_id == 0:
				self.num_iter += 1

				if self.forward_passes == 0:
					temp_grads = list(deepcopy(all_grads))
				else:
					for i in range(len(temp_grads)):
						temp_grads[i] += all_grads[i]

				self.num_iter += 1
				self.forward_passes += 1
				if self.forward_passes == self.total_forward:
					self.forward_passes = 0
					self._write_grads(self.original_state_dict, temp_grads, (query_x, query_y))
				else:
					self.model.load_state_dict(self.original_state_dict)

				# finished batch so can load data again from master
				process_event.set()


class MetaTrainer:

	def __init__(self, world_size, device='cpu', model_params=None):
		self.world_size = world_size

		self.meta_learners = [Learner(process_id=process_id, gpu=process_id if device is not 'cpu' else 'cpu', world_size=world_size, model_params=model_params) for process_id in range(world_size)]
		# gpu backend instead of gloo
		self.backend = "gloo"#"nccl"
		
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
			process_event.wait()

			process_event.clear()
			tasks = np.random.randint(0, num_tasks, (self.world_size))
			for task in tasks:
				# place holder for sampling data from dataset
				hey = next(data_loaders[task])
				# print(hey[0].shape)
				print(len(hey))
				print(hey[0].shape)
				print(hey[1].shape)
				print(hey[2].shape)
				print(hey[3].shape)
				data_queue.put((hey[0].numpy()[0], hey[1].numpy()[0], 
								hey[2].numpy()[0], hey[3].numpy()[0]))
				# data_queue.put(hey[0][0].numpy())
				# data_queue2.put(hey[1][0].numpy())
				# data_queue3.put(hey[2][0].numpy())
				# data_queue4.put(hey[3][0].numpy())
			data_event.set()

		new_model = self.meta_learners[0].model.original_state_dict

		for p in processes:
			p.terminate()
			p.join()