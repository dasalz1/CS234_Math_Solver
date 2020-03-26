from dataset import GeneratorDataset, batch_collate_fn
from torch.utils import data
from training import Trainer
import torch
import numpy as np
from tensorboard_utils import tb_mle_batch, tb_policy_batch

category_mappings = {
    'training': ['algebra', 'arithmetic', 'probability', 'numbers'],
    'induction': ['polynomials', 'calculus', 'measurement', 'comparison']
}

@staticmethod
def validate(model, batch_idx, mode = 'training', samples_per_category = 1024, use_mle=True, use_rl = False, tensorboard = None):
    torch.seed(1111)
    np.seed(1111)
    if mode not in category_mappings:
        mode = 'training'
        print("Mode not found in category mappings; reverting to default categories.")
    categories = category_mappings[mode]
    num_datapoints = samples_per_category * len(categories)
    validation_datasets = [
        GeneratorDataset(categories = [categories[category_index]],
                         num_iterations=1, batch_size = samples_per_category)
        for category_index in range(len(categories))]
    validation_loaders = [
        data.DataLoader(validation_datasets[category_index], batch_size=samples_per_category,
                        shuffle=True, collate_fn=batch_collate_fn)
        for category_index in range(len(categories))]
    validation_batches = [
        validation_loader[0]
        for validation_loader in validation_loaders]
    batch_qs, batch_as = [
        map(lambda x: x, validation_batch)
        for validation_batch in validation_batches]

    if use_mle and not use_rl:
        loss_by_category = []
        n_char_by_category = []
        n_correct_by_category = []
        for category_index in len(categories):
            loss, n_correct, n_char = Trainer.mle_batch_loss(batch_qs[category_index], batch_as[category_index],
                                                             model.action_transformer)
            loss_by_category.append(loss)
            n_correct_by_category.append(n_correct)
            n_char_by_category.append(n_char)
        average_loss = np.mean(loss_by_category)
        average_n_correct = np.mean(n_correct_by_category)
        average_n_char = np.mean(n_char_by_category)

        if tensorboard is not None:
            tb_mle_batch(tensorboard, average_loss, average_n_char, average_n_correct, batch_idx)

    elif use_rl and not use_mle:
        loss_by_category = []
        reward_by_category = []
        for category_index in len(categories):
            policy_losses, value_losses, batch_rewards = Trainer.policy_batch_loss(batch_qs, batch_as, model, gamma=0.9)
            loss = np.mean(policy_losses + value_losses)
            loss_by_category.append(loss)
            reward_by_category.append(np.mean(batch_rewards))
        average_loss = np.mean(loss_by_category)
        average_reward = np.mean(reward_by_category)
        if tensorboard is not None:
            tb_policy_batch(tensorboard, batch_rewards, value_losses, 0, 0, num_datapoints, batch_idx)

    return

    '''
    
    '''


def meta_validate():
    pass

