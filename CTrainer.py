import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboard_utils import *
from torch import autograd

class TeacherTrainer:

	def __init__(self, op='mle', device='cpu', teacher_model=None, teacher_optimizer=None, student_model=None, student_optimizer=None, tb=None):
		self.op = op
		self.device = device

		self.teacher_model = teacher_model
		self.teacher_optimizer = teacher_optimizer
		self.student_model = student_model
		self.student_optimizer = student_optimizer
		self.tb = tb

	def train_teacher(self, data_loader=None, K=10, task_batch_size=10, num_categories):
		category_acc = [0.0]*num_categories
		" this needs to be done"
		multinomial = Distribution
		for num_idx in range(num_iterations):
			" this needs to be done"
			category_probs = self.teacher_model(torch.FloatTensor(category_acc).contiguous().view(-1, 1))
			" this needs to be done"
			tasks = multinomial.sample(category_probs, K)

			valid_grads = [0.0]*num_categories
			accs = [[] for _ in range(num_categories)]
			sum_grads = None
			task_counts = [0]*num_categories

			for task in tasks:
				data = data_loader[task].get_sample(task_batch_size)
				task_counts[task] += 1
				loss, acc = self.student_model.loss_op(data=data, op=self.op)

				if 'meta' in self.op:
					curr_grads = autograd.grad(loss, self.student_model.parameters(), create_graph=True)
					sum_grads = [torch.add(i, j) for i, j in zip(sum_grads, curr_grads)] if sum_grads is not None else curr_grads
					valid_grads[task] = [torch.add(i, j) for i, j in zip(valid_grads[task], curr_grads)] if valid_grads[task] is not 0.0 else curr_grads
					accs[task].append(acc)
					iter_loss.append(loss); iter_acc.append(acc)
				else:
					self.student_optimizer.zero_grad()
					loss.backward()
					self.student_optimizer.step()

			if 'meta' in self.op:
				sum_grads = [grad/K for grad in sum_grads]
				self.student_model.write_grads(sum_grads, self.student_optimizer, data, op)


			valid_grads, avg_acc, avg_loss = self.create_grad_labels(tasks, valid_grads, category_acc, 
									accs, task_counts, iter_acc, iter_loss)


			self.tb.add_scalars({
				"iteration_acc": avg_acc,
				"iteration_loss": avg_loss
				},
			group="train"
			global_step=num_idx)
							
			valid_grads = torch.FloatTensor(valid_grads).contiguous().view(-1, 1)
			teacher_loss = torch.dot(category_probs.contiguous().view(-1, 1), valid_grads)

			self.teacher_optimizer.zero_grad()
			teacher_loss.backward()
			self.teacher_optimizer.step()

	def create_grad_labels(self, tasks, valid_grads, category_acc, accs, task_counts, iter_acc, iter_loss):
		for task in np.unique(tasks):
			if 'meta' in self.op:
				grads = sum([grad/task_counts[task]).abs().mean() for grad in valid_grads[task]]).item()
				category_acc[task] = category_acc[task]/2 + np.mean(accs[task])/2
			else:
				"this needs to be done"
				valid_data = validation_sampler(task)
				loss, acc = self.student_model.loss_op(valid_data, self.op)
				category_acc[task] = category_acc[task]/2 + acc/2
				grads = autograd.grad(loss, self.student_model.parameters(), create_graph=True)
				grads = sum([(grad/task_counts[task]).abs().mean() for grad in grads]).item()
				iter_acc.append(acc); iter_loss.append(loss.item())

			valid_grads[task] = grads
		
		return valid_grads, np.mean(iter_acc), np.mean(iter_loss)


