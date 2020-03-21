from tensorboard_utils import Tensorboard
import argparse, random
import numpy as np
import torch
import torch.optim as optim
from A3C import Policy_Network
from curriculumTrainer import CurriculumTrainer
from datetime import date
from dataset import batch_collate_fn, DeepCurriculumDataset
from old_dataset import MathDatasetManager, question_answer_to_batch_collate_fn
from torch.utils.data import DataLoader
from parameters import CUDA_VISIBLE_DEVICES
from MathConstants import full_categories
from curriculumModel import CurriculumNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='math_ds_dcl', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--use_mle_only", default=False, action='store_true')
parser.add_argument("--use_rl_only", default=False, action='store_true')
args = parser.parse_args()

math_dataset_file_path = "./mathematics_dataset-v1.0"

def init_seed_and_devices():
    seed = 68492
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs...")

    if torch.cuda.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return device

def main(args):
    # User parameters
    data_loader_batch_size = 4
    num_iterations = 1.e5
    categories = full_categories
    mean_accuracy_by_category = torch.zeros(len(categories))
    curriculum_step_length = 16

    # Determine exp_name and unique_id
    exp_name = args.exp_name
    if args.use_mle_only:
        exp_name += "_full_mle"
    elif args.use_rl_only:
        exp_name = "_full_rl"
    else:
        exp_name = "_full_mle_rl"
    unique_id = args.unique_id + "_random_seed_68492"

    # Initialization
    device = init_seed_and_devices()
    args = parser.parse_args()
    tb = Tensorboard(args.exp_name, unique_name=args.unique_id)
    mdsmgr = MathDatasetManager(math_dataset_file_path)

    # Get data categories and types from static dataset
    categories = mdsmgr.get_categories()
    types = mdsmgr.get_types()

    # Algorithm-Specific Parameters
    if args.use_mle_only:
        batch_size = 1024
        checkpoint_interval = 10000
        epochs = 20
        collate_fn = batch_collate_fn#question_answer_to_batch_collate_fn
    else:
        batch_size = 1
        epochs = num_iterations
        num_iterations = 1
        checkpoint_interval = 100
        collate_fn = batch_collate_fn

    # Create teacher and student model and optimizers
    # TODO: nn.DataParallel helpful?
    student_model = Policy_Network().to(device)
    teacher_model = CurriculumNetwork(len(categories))
    student_optimizer = optim.Adam(student_model.parameters(), lr=6.e-4, betas=(0.9, 0.995), eps=1e-8)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=4.e-2, betas=(0.9, 0.995), eps=1e-8)
    student_scheduler = optim.lr_scheduler.MultiStepLR(student_optimizer,
                                               milestones=[round(0.25 * num_iterations), round(0.5 * num_iterations),
                                                           round(0.75 * num_iterations)], gamma=0.1)

    # Periodically create a new DCL dataset but with updated performance info, so teacher can adjust curriculum
    num_curriculum_steps = round(num_iterations / curriculum_step_length)
    for curriculum_step in range(num_curriculum_steps):
        teacher_optimizer.zero_grad()
        # Create dataset and dataloader
        # TODO: num_iterations and batch_size okay here? Were hard-coded to 12 and 4 before, not sure why.
        # TODO: Is there a difference between data_loader_batch_size and batch_size?
        curriculum_dataset = DeepCurriculumDataset(categories, mean_accuracy_by_category, teacher_model, difficulty=0.5,
                                                   num_iterations=num_iterations, batch_size=batch_size)
        #curriculum_dataset = NaïveCurriculumDataset(categories=['algebra', 'arithmetic', 'probability', 'numbers'])
        curriculum_data_loader = DataLoader(curriculum_dataset, batch_size=data_loader_batch_size, shuffle=True,
                                            collate_fn=collate_fn, num_workers=len(CUDA_VISIBLE_DEVICES))
        # Create scheduler, trainer, and optimizer
        # TODO: Split by RL/MLE here? Ex. separate model architectures?
        curriculum_trainer = CurriculumTrainer(args.use_mle_only, args.use_rl_only, device)
        total_loss = curriculum_trainer.train(curriculum_data_loader, student_model, student_optimizer, student_scheduler, tensorboard = tb, num_epochs=epochs, checkpoint_interval=checkpoint_interval, iterations=curriculum_step * curriculum_step_length)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 0.1)
        teacher_optimizer.step()
        # TODO (MAST): backpropagate teacher model based on adapted model's performance on real data, not on generated data

if __name__ == '__main__':
    main(args)
