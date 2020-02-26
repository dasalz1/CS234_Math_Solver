import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE
#from dataset import PAD, UNK, BOS, EOS
from dataset import PAD



class Trainer:
  def __init__(self, use_mle, use_rl, device='cpu'):
    self.device=device
    self.eps = np.finfo(np.float32).eps.item()
    self.use_mle = use_mle
    self.use_rl = use_rl

  def save_checkpoint(self, epoch, model, optimizer, suffix="default"):
    torch.save({
      'epoch': epoch,
      'model': model,
      'optimizer': optimizer
      }, "checkpoint-{}.pth".format(suffix))

  def from_checkpoint_if_exists(self, model, optimizer):
    epoch = 0
    if os.path.isfile("checkpoint.pth"):
      print("Loading existing checkpoint...")
      checkpoint = torch.load("checkpoint.pth")
      epoch = checkpoint['epoch']
      model.load_state_dict(checkpoint['model'])
      optimizer.load_state_dict(checkpoint['optimizer'])
    return epoch, model, optimizer
  
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

  def tb_mle_batch(self, tb, total_loss, n_char_total, n_char_correct, epoch, batch_idx, data_len):
    tb.add_scalars(
      {
        "loss_per_char" : total_loss / n_char_total,
        "accuracy": n_char_correct / n_char_total,
      },
    group="mle_train",
    sub_group="batch",
    global_step = epoch*data_len+batch_idx)

  def tb_mle_epoch(self, tb, loss_per_char, accuracy, epoch):
    tb.add_scalars(
      {
        "loss_per_char" : loss_per_char,
        "accuracy" : accuracy,
      },
      group="train",
      sub_group="epoch",
      global_step=epoch
    )

  def tb_policy_batch(self, tb, batch_rewards, average_value_loss, epoch, batch_idx, data_len):
    tb.add_scalars(
      {
        "batch_average_rewards" : batch_rewards,
        "epoch_value_loss": average_value_loss, 
      },
    group="mle_train",
    sub_group="batch",
    global_step = epoch*data_len+batch_idx)

  def tb_policy_epoch(self, tb, average_rewards, average_value_loss, epoch):
    tb.add_scalars(
      {
        "epoch_average_reward" : average_rewards,
        "epoch_value_loss": average_value_loss, 
      },
      group="train",
      sub_group="epoch",
      global_step=epoch
    )

  def train(self, training_data, model, optimizer, tb=None, epochs=20, log_interval=100, checkpoint_interval=100):
    
    curr_epoch, model, optimizer = self.from_checkpoint_if_exists(model, optimizer)
    model.train()
    ignore_index = model.module.action_transformer.trg_pad_idx
    eta = 0.95


    for epoch in range(curr_epoch, epochs):
      total_mle_loss = 0.0
      n_char_total = 0.0
      n_char_correct = 0.0
      all_rewards = []
      all_value_losses = []
      for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):
        batch_qs, batch_as = map(lambda x: x.to(self.device), batch)
        optimizer.zero_grad()

        if self.use_rl:
          policy_losses, value_losses, batch_rewards = self.policy_batch_loss(batch_qs, batch_as, model, gamma=0.9)

        mle_loss, n_correct, n_char = self.mle_batch_loss(batch_qs, batch_as, model.module.action_transformer)

        if self.use_mle:
          loss = mle_loss
        elif self.use_rl:
          loss = policy_losses + value_losses
        else:
          loss = (1-eta)*policy_losses + value_losses + eta*mle_loss

        loss.backward()
        optimizer.step()

        n_char_total += n_char
        n_char_correct += n_correct
        total_mle_loss += mle_loss
        if self.use_rl:
          all_rewards.append(batch_rewards)
          all_value_losses.append(value_losses)

        if tb is not None and batch_idx % log_interval == 0:
          self.tb_mle_batch(tb, total_mle_loss, n_char_total, n_char_correct, epoch, batch_idx, len(training_data))
          if not self.use_mle:
            self.tb_policy_batch(tb, batch_rewards, value_losses, epoch, batch_idx, len(training_data))
        
        if batch_idx != 0 and batch_idx % checkpoint_interval == 0:
          self.save_checkpoint(epoch, model, optimizer, suffix=str(batch_idx))
      
      print("average rewards " + str(all_rewards))  
      loss_per_char = total_loss / n_char_total
      accuracy = n_char_correct / n_char_total

      if self.use_rl:
        average_rewards = np.mean(all_rewards)
        average_value_loss = np.mean(all_value_losses)
      
      if tb is not None:
        self.tb_mle_epoch(tb, loss_per_char, accuracy, epoch)
        self.tb_policy_epoch(tb, average_rewards, average_value_loss, epoch)

  def mle_batch_loss(self, batch_qs, batch_as, model):
    trg_as = batch_as[:, 1:]
    pred_logits = model(batch_qs, batch_as[:, :-1])
    pred_logits = pred_logits.view(-1, pred_logits.size(2))
    loss, n_correct = self.compute_mle_loss(pred_logits, trg_as, smoothing=True)
    
    non_pad_mask = trg_as.ne(PAD)
    n_char = non_pad_mask.sum().item()

    return loss, n_correct, n_char

  def policy_batch_loss(self, batch_qs, batch_as, model, gamma):
    batch_size, max_len_sequence = batch_qs.shape[0], batch_as.shape[1]
    current_as = batch_as[:, :1]
    complete = torch.ones((batch_size, 1)).to(self.device)
    rewards = torch.zeros((batch_size, 0)).to(self.device)
    values = torch.zeros((batch_size, 0)).to(self.device)
    log_probs = torch.zeros((batch_size, 0)).to(self.device)
    advantages_mask = torch.ones((batch_size, 0)).to(self.device)
    for t in range(1, max_len_sequence):
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

    batch_rewards = rewards.sum(dim=-1).mean()
    return policy_losses, value_losses, batch_rewards
  
   
  def train_mle_epoch(self, training_data, model, optimizer, epoch, tb=None, log_interval=100):
    model.train()
    total_loss = 0.0
    n_char_total = 0.0
    n_char_correct = 0.0
    for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):
      batch_qs, batch_as = map(lambda x: x.to(self.device), batch)
      trg_as = batch_as[:, 1:]
      optimizer.zero_grad()
      pred_logits = model(batch_qs, batch_as[:, :-1])
      pred_logits = pred_logits.view(-1, pred_logits.size(2))
      loss, n_correct = self.compute_mle_loss(pred_logits, trg_as, smoothing=True)
      loss.backward()
      
      optimizer.step()
      
      total_loss += loss.item()
      
      non_pad_mask = trg_as.ne(PAD)
      n_char = non_pad_mask.sum().item()
      n_char_total += n_char
      n_char_correct += n_correct
      
      if tb is not None and batch_idx % log_interval == 0:
        self.tb_mle_batch(tb, total_loss, n_char_total, n_char_correct, epoch, batch_idx, len(training_data))

    loss_per_char = total_loss / n_char_total
    accuracy = n_char_correct / n_char_total
    
    if tb is not None:
      self.tb_mle_epoch(tb, loss_per_char, accuracy, epoch)         

  def train_policy_epoch(self, training_data, model, gamma, optimizer):

    model.train()
    ignore_index = model.module.action_transformer.trg_pad_idx
    all_rewards = []
    all_value_losses = []
  #     sample batch of questions and answers
    for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):
      batch_qs, batch_as = map(lambda x: x.to(self.device), batch)
      batch_size, max_len_sequence = batch_qs.shape[0], batch_as.shape[1]
      current_as = batch_as[:, :1]
      complete = torch.ones((batch_size, 1))
      rewards = torch.zeros((batch_size, 0))
      values = torch.zeros((batch_size, 0))
      log_probs = torch.zeros((batch_size, 0))
      advantages_mask = torch.ones((batch_size, 0))
      for t in range(1, max_len_sequence):
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

      batch_rewards = rewards.sum(dim=-1).mean()
      all_rewards.append(batch_rewards)
      all_value_losses.append(value_losses)

      if tb is not None and batch_idx % log_interval == 0:
        self.tb_policy_batch(tb, batch_rewards, value_losses, epoch, batch_idx, len(training_data))
    

    average_rewards = np.mean(all_rewards)
    average_value_loss = np.mean(all_value_losses)

    if tb is not None:
      self.tb_policy_epoch(tb, average_rewards, average_value_loss, epoch)




