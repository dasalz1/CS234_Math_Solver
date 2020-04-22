from dataset import GeneratorDataset
from torch import autograd
from training import Trainer
import torch
import numpy as np
import tensorboard_utils
from A3C import Policy_Network
from CMeta import MetaLearner
import torch.optim as optim
import MathConstants
from copy import deepcopy

category_mappings = {
    'training': ['algebra', 'arithmetic', 'probability', 'numbers'],
    'induction': ['polynomials', 'calculus', 'measurement', 'comparison']
}

subcategories_by_category = {
    category_name: sum(MathConstants.subcategories[category_name])
    for category_name in sum(category_mappings.values())
}

def validate(model, batch_idx, category_names, mode = 'training', samples_per_category = 16, use_mle=True, use_rl = False, tensorboard = None):
    if mode not in category_mappings:
        mode = 'training'
        print("Mode not found in category mappings; reverting to default training categories.")
    categories = category_mappings[mode]
    assert ((use_rl and not use_mle) or (use_mle and not use_rl))

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
        exact_by_category = []
        n_correct_by_category = []
        n_char_by_category = []
        for category_index in range(len(categories)):
            loss, n_correct, n_char = Trainer.mle_batch_loss(batch_qs[category_index], batch_as[category_index], model.action_transformer)
            loss_by_category.append(loss.item())
            n_correct_by_category.append(n_correct)
            n_char_by_category.append(n_char)
            ## Calculate exact matching percentage
            num_correct = 0
            pred_logits = model.action_transformer(input_ids=batch_qs[category_index],
                                                   decoder_input_ids=batch_as[category_index])
            pred_chars = torch.max(pred_logits, dim=-1)[1]
            trg_as = batch_as[category_index]
            # TODO (minor): Vectorize this loop?
            for batch_idx in range(samples_per_category):
                if pred_chars[batch_idx].size()[-1] != trg_as.size()[-1]:
                    break
                if torch.all(pred_chars[batch_idx].eq(trg_as[batch_idx])):
                    num_correct += 1
            exact_by_category.append(num_correct / samples_per_category)
            ##

        average_loss = np.mean(loss_by_category)
        average_n_correct = np.mean(n_correct_by_category)
        average_n_char = np.mean(n_char_by_category)



        if tensorboard is not None:
            tensorboard_utils.tb_mle_batch(tensorboard, average_loss, average_n_char, average_n_correct, batch_idx)

        print(f"Validation results for batch {batch_idx}:")
        val_loss_string, val_acc_string, val_exact_string = "", "", ""
        for category_idx, category in enumerate(category_names):
            val_loss_string += f"{category}: {loss_by_category[category_idx]}, "
            val_acc_string += f"{category}: {n_correct_by_category[category_idx] / n_char_by_category[category_idx]}, "
            val_exact_string += f"{category}: {exact_by_category[category_idx]}, "
        print(f"Batch-averaged losses:\n{val_loss_string[:-2]}\nBatch-averaged accuracies:\n{val_acc_string[:-2]}\n\
              Batch-averaged exact matching accuracies:\n{val_exact_string[:-2]}\n")

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

def meta_validate(starting_model, batch_idx, category_names, mode = 'induction', samples_per_category = 16, use_mle=True, use_rl = False, tensorboard = None, K = 5):
    if mode not in category_mappings:
        mode = 'induction'
        print("Mode not found in category mappings; reverting to default induction categories.")
    categories = category_mappings[mode]
    average_support_accs_by_category = [[] for _ in range(len(categories))]
    average_query_accs_by_category = [[] for _ in range(len(categories))]
    average_support_losses_by_category = [[] for _ in range(len(categories))]
    average_query_losses_by_category = [[] for _ in range(len(categories))]
    num_support_sgd_steps = 1
    assert((use_rl and not use_mle) or (use_mle and not use_rl))
    assert(starting_model is MetaLearner)
    op = 'meta-mle' if use_mle else 'meta-rl'
    loss_by_category, acc_by_category, exact_by_category = [], [], []

    # Meta-train separately on each category, treating subcategories as tasks
    for category_index, category in enumerate(categories):
        meta_model = deepcopy(starting_model)
        subcategories = subcategories_by_category[category]
        N = len(subcategories)
        sum_grads, all_data, valid_grads = None, None, [0.0] * len(subcategories)
        support_accs, support_losses = [[] for _ in range(N)], [[] for _ in range(N)]
        query_accs, query_losses = [[] for _ in range(N)], [[] for _ in range(N)]
        optimizer = optim.Adam(meta_model.parameters(), lr=6.e-4, betas=(0.9, 0.995), eps=1e-8)

        # Create 2K datapoints, K for support and K for query
        meta_validation_datasets = [
            GeneratorDataset(categories=subcategory,
                             num_iterations=1, batch_size=2*K)
            for subcategory in subcategories]

        validation_batches = [
            list(map(lambda x: torch.LongTensor(x.values), meta_validation_dataset.get_sample()))
            for meta_validation_dataset in meta_validation_datasets
        ]

        batch_qs_by_category = [validation_batch[0] for validation_batch in validation_batches]
        batch_as_by_category = [validation_batch[1] for validation_batch in validation_batches]

        # Split into support/query
        support_qs = batch_qs_by_category[category_index][:K]
        support_as = batch_as_by_category[category_index][:K]
        query_qs = batch_qs_by_category[category_index][K:]
        query_as = batch_as_by_category[category_index][K:]

        for subcategory_idx in range(N):
            for _ in range(num_support_sgd_steps):
                loss, acc = meta_model.loss_op(data=support_qs, op=op)
                current_step_grads = autograd.grad(loss, meta_model.parameters(), create_graph=True, allow_unused=True)
                sum_grads = [torch.add(i, j) for i, j in zip(sum_grads, current_step_grads) if
                             (j is not None and i is not None)] if sum_grads is not None else current_step_grads
                valid_grads[subcategory_idx] = [torch.add(i, j) for i, j in zip(valid_grads[subcategory_idx], current_step_grads) if
                                     (j is not None and i is not None)] if valid_grads[subcategory_idx] is not 0.0 else current_step_grads
                support_accs[subcategory_idx].append(acc)
                support_losses[subcategory_idx].append(loss.item())

                # average gradients and apply meta-gradients (support)
                for idx in range(len(sum_grads)):
                    sum_grads[idx].data = sum_grads[idx].data / K
                dummy_x, dummy_y = support_qs[0], support_as[0]
                meta_model.write_grads(sum_grads, optimizer, (dummy_x, dummy_y), op)

            # calculate query results
            loss, acc = meta_model.loss_op(data=query_qs, op=op)

            query_accs[subcategory_idx] = np.mean(acc)
            query_losses[subcategory_idx] = np.mean(loss)

        average_support_accs_by_category[category_index] = np.mean(support_accs)
        average_query_accs_by_category[category_index] = np.mean(query_accs)
        average_support_losses_by_category[category_index] = np.mean(support_losses)
        average_query_losses_by_category[category_index] = np.mean(query_losses)

        if tensorboard:
            tensorboard.add_scalars(
                {f"{category} query accuracy": average_support_accs_by_category[category_index]\
               , f"{category} query loss": average_query_losses_by_category[category_index]},
                group="train", sub_group="batch", global_step=batch_idx)

    ######