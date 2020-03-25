import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_utils import *
from torch import autograd
from torch.distributions.categorical import Categorical
from tqdm import tqdm

class TeacherTrainer:

	def __init__(self, op='mle', device='cpu', teacher_network=True, teacher_model=None, student_model=None, student_optimizer=None, teacher_lr=0.1, validation_samples=1, tb=None):
		self.op = op
		self.device = device


		self.teacher_model = teacher_model
		# Teacher is either parametrized by a neural network or just the raw parameters of the Multinomial distribution
		if teacher_model:
			self.teacher_optimizer = optim.SGD(self.teacher_model.parameters(), teacher_lr)
			self.loss_scale = 1
		else:
			self.teacher_params = torch.Variable(torch.FloatTensor([1/num_categories]*num_categories).contiguous().view(-1), requires_grad=True)
			# scale loss if parametrizing distribution directly
			self.loss_scale = teacher_lr

		self.student_model = student_model
		self.student_optimizer = student_optimizer
		self.tb = tb
		self.validation_samples = validation_samples

	def train_teacher(self, data_loader=None, K=10, task_batch_size=10, num_categories=1, num_iterations=100000):
		category_acc = [0.0]*num_categories
		for num_idx in tqdm(range(num_iterations), mininterval=2, leave=False):

			if self.teacher_model:
				category_probs = self.teacher_model(torch.FloatTensor(category_acc).contiguous().view(1, -1)).contiguous().view(-1)
			else:
				category_probs = self.teacher_params

			# Add random noise to current parameters
			category_probs_mod = category_probs + (0.05 - torch.rand_like(category_probs)*0.1)

			# sample K tasks for this iteration using the normalized modified category probs
			tasks = Categorical(category_probs_mod/category_probs_mods.sum()).sample((K,))

			valid_grads = [0.0]*num_categories
			accs = [[] for _ in range(num_categories)]
			losses = [[] for _ in range(num_categories)]
			sum_grads = None
			task_counts = [0]*num_categories

			for task in tasks:
				data = map(lambda x: torch.LongTensor(x).to(self.device), data_loader[task].get_sample())
				task_counts[task] += 1
				loss, acc = self.student_model.loss_op(data=data, op=self.op)

				# if meta model accumulate all gradients here for final optimization and store validation values here since query is in essence query
				if 'meta' in self.op:
					curr_grads = autograd.grad(loss, self.student_model.parameters(), create_graph=True)
					sum_grads = [torch.add(i, j) for i, j in zip(sum_grads, curr_grads)] if sum_grads is not None else curr_grads
					valid_grads[task] = [torch.add(i, j) for i, j in zip(valid_grads[task], curr_grads)] if valid_grads[task] is not 0.0 else curr_grads
					accs[task].append(acc); losses[task].append(loss)
					iter_loss.append(loss); iter_acc.append(acc)

				# regular optimization step
				else:
					self.student_optimizer.zero_grad()
					loss.backward()
					self.student_optimizer.step()

			if 'meta' in self.op:
				# average gradients and apply meta gradients
				sum_grads = [grad/K for grad in sum_grads]
				self.student_model.write_grads(sum_grads, self.student_optimizer, (data[2], data[3]), op)

			valid_grads, valid_losses, avg_acc, avg_loss = self.create_labels(tasks, valid_grads, category_acc, 
									accs, losses, task_counts, iter_acc, iter_loss)


			self.tb.add_scalars({"iteration_acc": avg_acc, "iteration_loss": avg_loss}, group="train", global_step=num_idx)

			valid_grads = torch.FloatTensor(valid_grads).contiguous().view(-1)
			valid_losses = torch.FloatTensor(valid_losses).contiguous().view(-1)
			# loss scale is in case parameters are direct distribution, in essence the learning rate
			teacher_loss = (torch.dot(category_probs, valid_grads) +  torch.dot(category_probs, valid_losses))*self.loss_scale

			teacher_loss.backward()
			if self.teacher_model:
				self.teacher_optimizer.step()
				self.teacher_optimizer.zero_grad()



	def create_labels(self, tasks, valid_grads, category_acc, accs, losses, task_counts, iter_acc, iter_loss):
		for task in np.unique(tasks):
			if 'meta' in self.op:
				# accumulate validation gradients and metrics from task for loop
				grads = sum([grad/task_counts[task].abs().mean() for grad in valid_grads[task]]).item()
				category_acc[task] = category_acc[task]/2 + np.mean(accs[task])/2
				loss = np.mean(losses[task])
			else:
				# create, process and accumulate validation gradients and metrics for tasks sampled in for loop
				valid_data = map(lambda x: x.to(self.device), data_loader[task].get_valid_sample(self.validation_samples))
				loss, acc = self.student_model.loss_op(valid_data, self.op)
				category_acc[task] = category_acc[task]/2 + acc/2
				grads = autograd.grad(loss, self.student_model.parameters(), create_graph=True)
				grads = sum([(grad/task_counts[task]).abs().mean() for grad in grads]).item()
				loss = loss.item()
				iter_acc.append(acc); iter_loss.append(loss)

			valid_grads[task] = grads
			valid_losses[task] = loss
		
		return valid_grads, valid_losses, np.mean(iter_acc), np.mean(iter_loss)


