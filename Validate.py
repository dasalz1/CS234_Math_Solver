from dataset import GeneratorDataset, batch_collate_fn
from torch.utils import data
from training import Trainer
import torch
import numpy as np
import tensorboard_utils
from A3C import Policy_Network

category_mappings = {
    'training': ['algebra', 'arithmetic', 'probability', 'numbers'],
    'induction': ['polynomials', 'calculus', 'measurement', 'comparison']
}

def validate(model, batch_idx, category_names, mode = 'training', samples_per_category = 16, use_mle=True, use_rl = False, tensorboard = None):
    if mode not in category_mappings:
        mode = 'training'
        print("Mode not found in category mappings; reverting to default categories.")
    categories = category_mappings[mode]
    num_datapoints = samples_per_category * len(categories)
    validation_datasets = [
        GeneratorDataset(categories = [categories[category_index]],
                         num_iterations=1, batch_size = samples_per_category)
        for category_index in range(len(categories))]

    validation_batches = [
        list(map(lambda x: torch.LongTensor(x.values), validation_dataset.get_sample()))
        for validation_dataset in validation_datasets
    ]

    batch_qs = [validation_batch[0] for validation_batch in validation_batches]
    batch_as = [validation_batch[1] for validation_batch in validation_batches]

    if use_mle and not use_rl:
        loss_by_category = []
        n_correct_by_category = []
        n_char_by_category = []
        for category_index in range(len(categories)):
            loss, n_correct, n_char = Trainer.mle_batch_loss(batch_qs[category_index], batch_as[category_index], model.action_transformer)
            loss_by_category.append(loss.item())
            n_correct_by_category.append(n_correct)
            n_char_by_category.append(n_char)

        average_loss = np.mean(loss_by_category)
        average_n_correct = np.mean(n_correct_by_category)
        average_n_char = np.mean(n_char_by_category)

        if tensorboard is not None:
            tensorboard_utils.tb_mle_batch(tensorboard, average_loss, average_n_char, average_n_correct, batch_idx)

        print(f"Validation results for batch {batch_idx}:")
        val_loss_string, val_acc_string = "", ""
        for category_idx, category in enumerate(category_names):
            val_loss_string += f"{category}: {loss_by_category[category_idx]}, "
            val_acc_string += f"{category}: {n_correct_by_category[category_idx] / n_char_by_category[category_idx]}, "
        print(f"Batch-averaged losses:\n{val_loss_string[:-2]}\nBatch-averaged accuracies:\n{val_acc_string[:-2]}\n")

        return {
            'loss_by_category': loss_by_category,
            'n_correct_by_category': n_correct_by_category,
            'n_char_by_category': n_char_by_category
        }

    elif use_rl and not use_mle:
        value_loss_by_category = []
        loss_by_category = []
        batch_reward_by_category = []
        for category_index in range(len(categories)):
            policy_loss, value_loss, batch_reward = Policy_Network.policy_batch_loss(model,
                                                                                   batch_qs[category_index],
                                                                                   batch_as[category_index],
                                                                                   gamma=0.9)
            loss = (policy_loss + value_loss).item()
            value_loss_by_category.append(value_loss.item())
            loss_by_category.append(loss)
            batch_reward_by_category.append(batch_reward)
        average_value_loss = np.mean(value_loss_by_category)
        average_batch_rewards = np.mean(batch_reward_by_category, axis=0)
        if tensorboard is not None:
            tensorboard_utils.tb_policy_batch(tensorboard, average_batch_rewards, average_value_loss, batch_idx)

        print(f"Validation results for batch {batch_idx}:")
        val_loss_string, val_rew_string = "", ""

        for category_idx, category in enumerate(category_names):
            val_loss_string += f"{category}: {loss_by_category[category_idx]}, "
            val_rew_string += f"{category}: {value_loss_by_category[category_idx]}, "
        print(f"Batch-averaged losses:\n{val_loss_string[:-2]}\nBatch-averaged accuracies:\n{val_rew_string[:-2]}\n")

        return {
            'loss_by_category': loss_by_category,
            'value_loss_by_category': value_loss_by_category,
            'batch_reward_by_category': batch_reward_by_category,
        }

    else:
        raise Exception("Validation must be run with either MLE or RL model.")

def meta_validate():


