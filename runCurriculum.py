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
    # Parameters
    data_loader_batch_size = 4
    num_iterations = 1.e5

    # Initialization
    device = init_seed_and_devices()
    args = parser.parse_args()
    tb = Tensorboard(args.exp_name, unique_name=args.unique_id)
    mdsmgr = MathDatasetManager(math_dataset_file_path)

    # Determine exp_name and unique_id
    exp_name = args.exp_name
    if args.use_mle_only:
        exp_name += "_full_mle"
    elif args.use_rl_only:
        exp_name = "_full_rl"
    else:
        exp_name = "_full_mle_rl"
    unique_id = args.unique_id + "_random_seed_68492"

    #Get data categories and types from static dataset
    categories = mdsmgr.get_categories()
    types = mdsmgr.get_types()

    # Algorithm-Specific Parameters
    if args.use_mle_only:
        batch_size = 1024
        checkpoint_interval = 10000
        epochs = 20
        collate_fn = question_answer_to_batch_collate_fn
    else:
        batch_size = 1
        epochs = num_iterations
        num_iterations = 1
        checkpoint_interval = 100
        collate_fn = batch_collate_fn

    # Create dataset and dataloader
    curriculum_dataset = DeepCurriculumDataset('''model performance''')
    curriculum_data_loader = DataLoader(
        curriculum_dataset, batch_size=data_loader_batch_size, shuffle=True,
        collate_fn=batch_collate_fn, num_workers=len(CUDA_VISIBLE_DEVICES))

    # Create model, scheduler, trainer, and optimizer
    # TODO: Split by RL/MLE here? Ex. different models?
    # TODO: nn.DataParallel helpful?
    model = torch.nn.DataParallel(Policy_Network().to(device))#Policy_Network().to(device)
    curriculum_trainer = CurriculumTrainer(args.use_mle_only, args.use_rl_only, device)
    optimizer = optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.995), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[round(0.25 * num_iterations), round(0.5 * num_iterations),
                                                           round(0.75 * num_iterations)], gamma=0.1)

    curriculum_trainer.train(curriculum_data_loader, model, optimizer, scheduler, tb)

if __name__ == '__main__':
    main(args)
