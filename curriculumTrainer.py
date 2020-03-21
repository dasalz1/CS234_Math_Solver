import numpy as np
import torch
import os
from dataset import EOS, PAD
import torch.nn.functional as F
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from torch.distributions import Categorical
from tqdm import tqdm

class CurriculumTrainer():
    def __init__(self, use_mle, use_rl, device='cpu'):
        self.device = device
        self.eps = np.finfo(np.float32).eps.item()
        self.use_mle = use_mle
        self.use_rl = use_rl

    def save_checkpoint(self, epoch, model, optimizer, scheduler, suffix="default"):
        save_dict = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        if scheduler:
            save_dict['scheduler'] = scheduler.state_dict()
        torch.save(save_dict, f"checkpoint-{suffix}.pth")

    def from_checkpoint_if_exists(self, model, optimizer, scheduler):
        epoch = 0
        model = optimizer = scheduler = None
        if os.path.isfile("checkpoint.pth"):
            print("Loading existing checkpoint...")
            checkpoint = torch.load("checkpoint.pth")
            epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'], strict = False)
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("Error occurred loading optimizer state dict...")
            if scheduler and 'scheduler' in checkpoint and self.use_mle:
                scheduler.load_state_dict(checkpoint['scheduler'])
        return epoch, model, optimizer, scheduler

    def calculate_reward(self, actions_prediction, actions, ignore_index = 0, sparse_rewards = False):
        if sparse_rewards:
            if actions_prediction == EOS and actions == EOS:
                return torch.ones_like(actions).cuda().float()
            return torch.zeros_like(actions).cuda().float()
        else:
            return (actions_prediction == actions).float()

    def get_returns(self, rewards, batch_size, gamma):
        trajectory_length = rewards.shape[1]
        discounts = torch.tensor(np.logspace(0, trajectory_length, trajectory_length, base=gamma, endpoint=False)).view(1, -1).to(self.device)
        all_returns = torch.zeros((batch_size, trajectory_length)).to(self.device)

        for time_step in range(trajectory_length):
            all_returns[:, time_step]= (discounts[:, :trajectory_length - time_step]).sum(dim=-1)
            (all_returns - all_returns.mean(dim=-1).view(-1, 1)) / (all_returns.std(dim=-1).view(-1, 1) + self.eps)
        return all_returns

    def compute_mle_loss(self, prediction, target, smoothing):
        def compute_cross_entropy_loss(prediction, target, smoothing):
            log_probabilities = F.log_softmax(prediction, dim=1)
            target = target.contiguous().view(-1)
            if smoothing:
                eps = 0.1
                num_classes = prediction.size(1)

                one_hot = torch.zeros_like(prediction)
                one_hot = one_hot.scatter(1, target.view(-1, 1), 1)
                one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)

                non_pad_mask = target.ne(PAD)
                loss = -(one_hot * log_probabilities).sum(dim=1)
                loss = loss.masked_select(non_pad_mask).sum()
            else:
                loss = F.cross_entropy(prediction, target, ignore_index=PAD, reduction='sum')
            return loss

        cross_entropy_loss = compute_cross_entropy_loss(prediction, target, smoothing)
        prediction_max = prediction.max(1)[1]
        target = target.contiguous().view(-1)
        non_pad_mask = target.ne(PAD)
        num_correct = prediction_max.eq(target).masked_select(non_pad_mask).sum().item()

        return cross_entropy_loss, num_correct

    def tb_mle_batch(self, tb, total_loss, n_char_total, n_char_correct, epoch, batch_idx, data_len):
        tb.add_scalars(
            {
                "loss_per_char": total_loss / n_char_total,
                "accuracy": n_char_correct / n_char_total,
            },
            group="mle_train",
            sub_group="batch",
            global_step=epoch * data_len + batch_idx)

    def tb_mle_epoch(self, tb, loss_per_char, accuracy, epoch):
        tb.add_scalars(
            {
                "loss_per_char": loss_per_char,
                "accuracy": accuracy,
            },
            group="train",
            sub_group="epoch",
            global_step=epoch
        )

    def tb_policy_batch(self, tb, batch_rewards, average_value_loss, epoch, batch_idx, data_len):
        tb.add_scalars(
            {
                "batch_average_rewards": batch_rewards,
                "epoch_value_loss": average_value_loss,
            },
            group="policy_train",
            sub_group="batch",
            global_step=epoch * data_len + batch_idx)

    def tb_policy_epoch(self, tb, average_rewards, average_value_loss, epoch):
        tb.add_scalars(
            {
                "epoch_average_reward": average_rewards,
                "epoch_value_loss": average_value_loss,
            },
            group="train",
            sub_group="epoch",
            global_step=epoch
        )

    def tb_mle_policy_batch(self, tb, total_loss, n_char_total, n_char_correct, batch_rewards, epoch, batch_idx,
                            data_len):
        tb.add_scalars(
            {
                "loss_per_char": total_loss / n_char_total,
                "accuracy": n_char_correct / n_char_total,
                "batch_average_rewards": batch_rewards,
            },
            group="mle_policy_train",
            sub_group="batch",
            global_step=epoch * data_len + batch_idx)

    def tb_mle_policy_epoch(self, tb, loss_per_char, accuracy, average_rewards, epoch):
        tb.add_scalars(
            {
                "loss_per_char": loss_per_char,
                "accuracy": accuracy,
                "epoch_average_reward": average_rewards,
            },
            group="train",
            sub_group="epoch",
            global_step=epoch
        )

    def train(self, training_data, student_model, student_optimizer, student_scheduler = None, tensorboard = None, num_epochs = 20, log_interval = 1e2, checkpoint_interval = 1e5, iterations = 0):
        # Parameters
        eta = 0.95 #scaling constant from MLE to RL loss
        learning_rate = 6e-4

        # Variables
        total_loss = 0.
        
        current_epoch, model, optimizer, scheduler = self.from_checkpoint_if_exists(student_model, student_optimizer, student_scheduler)
        if model is not None:
            student_model = model
        if optimizer is not None:
            student_optimizer = optimizer
        if scheduler is not None:
            student_scheduler = scheduler
        student_model.train()

        for epoch in range(current_epoch, num_epochs):
            total_mle_loss = 0.0
            num_chars_total = 0.0
            num_chars_correct = 0.0
            all_rewards = []

            optimizer = AdamW(student_model.parameters(), lr = learning_rate)

            if self.use_mle or self.use_rl  :
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=4e4, num_training_steps=len(training_data))

            for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):
                batch_qs, batch_as = map(lambda x: x.to(self.device), batch)
                student_optimizer.zero_grad()

                if not self.use_mle:
                    policy_losses, batch_rewards = self.policy_batch_loss(batch_qs, batch_as, student_model, gamma=0.9)
                if not self.use_rl:
                    mle_loss, num_correct, num_chars = self.mle_batch_loss(batch_qs, batch_as, student_model.action_transformer)

                if self.use_mle:
                    loss = mle_loss
                elif self.use_rl:
                    loss = policy_losses
                else:
                    #TODO: why is there an /2 in the next line?
                    eta_linear_decay = eta - eta * (iterations / (float(len(training_data) * num_epochs) / 2))
                    loss = (1 - eta_linear_decay) * policy_losses + eta_linear_decay * mle_loss
                    iterations += batch_qs.shape[0]
                total_loss += loss
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 0.1)
                student_optimizer.step()
                if scheduler and self.use_mle:
                    scheduler.step()

                if not self.use_rl:
                    num_chars_total += num_chars
                    num_chars_correct += num_correct
                    total_mle_loss += mle_loss
                if not self.use_mle:
                    all_rewards.append(batch_rewards.cpu().numpy())

                if tensorboard is not None and batch_idx % log_interval == 0:
                    if self.use_mle:
                        self.tb_mle_batch(tensorboard, total_mle_loss, num_chars_total, num_chars_correct, epoch, batch_idx,
                                          len(training_data))
                    # TODO: Fix missing value_losses
                    #elif self.use_rl:
                    #    self.tb_policy_batch(tensorboard, batch_rewards, value_losses, epoch, batch_idx, len(training_data))
                    else:
                        self.tb_mle_policy_batch(tensorboard, total_mle_loss, num_chars_total, num_chars_correct, batch_rewards, epoch,
                                                 batch_idx, len(training_data))

    def mle_batch_loss(self, batch_qs, batch_as, model):
        target_answers = batch_as[:, 1:]
        prediction_logits = model(input_ids = batch_qs, decoder_input_ids = batch_as[:, :-1])
        prediction_logits = prediction_logits.reshape(-1, prediction_logits.size(2))
        loss, num_correct = self.compute_mle_loss(prediction_logits, target_answers, smoothing=True)

        non_pad_mask = target_answers.ne(PAD)
        num_char = non_pad_mask.sum().item()

        return loss, num_correct, num_char

    def policy_batch_loss(self, batch_qs, batch_as, model, gamma):
        batch_size, max_len_sequence = batch_qs.shape[0], batch_as.shape[1]
        current_as = batch_as[:, :1]
        complete = torch.ones((batch_size, 1)).to(self.device)
        rewards = torch.zeros((batch_size, 0)).to(self.device)
        log_probs = torch.zeros((batch_size, 0)).to(self.device)
        advantages_mask = torch.ones((batch_size, 0)).to(self.device)
        for timestep in range(1, max_len_sequence):
            advantages_mask = torch.cat((advantages_mask, complete), dim=1)
            # action_probs, curr_values = model(src_seq=batch_qs, trg_seq=current_as)
            action_probs = model(src_seq=batch_qs, trg_seq=current_as)
            m = Categorical(F.softmax(action_probs, dim=-1))
            actions = m.sample().reshape(-1, 1)

            target_timestep = batch_as[:, timestep].reshape(-1, 1)

            # Update decoder output
            current_as = torch.cat((current_as, actions), dim=1)
            current_log_probs = -F.cross_entropy(action_probs, target_timestep.reshape(-1), ignore_index=0,
                                              reduction='none').reshape(-1, 1)

            # Calculate reward based on character-level cross-entropy
            current_rewards = self.calculate_reward(actions, target_timestep)

            # Update terms
            rewards = torch.cat((rewards, current_rewards), dim=1).to(self.device)
            # TODO: Fix values
            log_probs = torch.cat((log_probs, current_log_probs), dim=1)

            # If the action taken is EOS or if end-of-sequence, trajectory ends
            complete *= (1 - ((actions == EOS) | (target_timestep == EOS)).float())

        returns = self.get_returns(rewards, batch_size, gamma)

        advantages = returns
        advantages *= advantages_mask

        policy_losses = (-log_probs * advantages).sum(dim=-1).mean()
        batch_rewards = rewards.sum(dim=-1).mean()

        return policy_losses, batch_rewards

    def train_mle_epoch(self, training_data, model, optimizer, epoch, tb=None, log_interval=100):
        model.train()
        total_loss = 0.0
        num_chars_total = 0.0
        num_chars_correct = 0.0
        for batch_idx, batch in enumerate(tqdm(training_data, mininterval=2, leave=False)):
            batch_qs, batch_as = map(lambda x: x.to(self.device), batch)
            target_answers = batch_as[:, 1:]
            optimizer.zero_grad()
            pred_logits = model(input_ids=batch_qs, decoder_input_ids=batch_as[:, :-1])
            pred_logits = pred_logits.view(-1, pred_logits.size(2))
            loss, n_correct = self.compute_mle_loss(pred_logits, target_answers, smoothing=True)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            non_pad_mask = target_answers.ne(PAD)
            n_char = non_pad_mask.sum().item()
            num_chars_total += n_char
            num_chars_correct += n_correct

            if tb is not None and batch_idx % log_interval == 0:
                self.tb_mle_batch(tb, total_loss, num_chars_total, num_chars_correct, epoch, batch_idx, len(training_data))

        loss_per_char = total_loss / num_chars_total
        accuracy = num_chars_correct / num_chars_total

        if tb is not None:
            self.tb_mle_epoch(tb, loss_per_char, accuracy, epoch)

    def train_policy_epoch(self, training_data, model, gamma, optimizer, tensorboard=None, log_interval=100):
        model.train()
        all_rewards = []
        all_value_losses = []

        # Sample batch of questions and answers
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
                action_probs, curr_values = model(input_ids=batch_qs, decoder_input_ids=current_as)
                m = Categorical(F.softmax(action_probs, dim=-1))
                actions = m.sample().contiguous().view(-1, 1)

                trg_t = batch_as[:, t].contiguous().view(-1, 1)

                # Update decoder output
                current_as = torch.cat((current_as, actions), dim=1)
                curr_log_probs = -F.cross_entropy(action_probs, trg_t.view(-1), ignore_index=0,
                                                  reduction='none').contiguous().view(-1, 1)

                # Calculate reward based on character-level cross-entropy
                current_rewards = self.calculate_reward(actions, trg_t)

                # Update terms
                rewards = torch.cat((rewards, current_rewards), dim=1)
                values = torch.cat((values, curr_values), dim=1)
                log_probs = torch.cat((log_probs, curr_log_probs), dim=1)

                # if the action taken is EOS or if end of sequence trajectory ends
                complete *= (1 - ((actions == EOS) | (trg_t == EOS)).float())

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

            if tensorboard is not None and batch_idx % log_interval == 0:
                # TODO: Fix epoch
                self.tb_policy_batch(tensorboard, batch_rewards, value_losses, epoch, batch_idx, len(training_data))

        average_rewards = np.mean(all_rewards)
        average_value_loss = np.mean(all_value_losses)

        if tensorboard is not None:
            self.tb_policy_epoch(tensorboard, average_rewards, average_value_loss, epoch)






