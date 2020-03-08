from tensorboard_utils import Tensorboard
import argparse, random
import numpy as np
import torch
import torch.optim as optim
from A3C import Policy_Network
from training import Trainer
from datetime import date
from dataset import NaïveCurriculumDataset
from dataset import batch_collate_fn

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='EditorPairTrain', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--use_mle_only", default=False, action='store_true')
parser.add_argument("--use_rl_only", default=False, action='store_true')
args = parser.parse_args()

def init_seed_and_devices():
    seed = 12324
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return device

def main(args):
    device = init_seed_and_devices()
    args = parser.parse_args()
    tb = Tensorboard(args.exp_name, unique_name=args.unique_id)

    curriculum_dataset = NaïveCurriculumDataset()
    curriculum_data_loader = torch.utils.data.DataLoader(curriculum_dataset, batch_size=4, collate_fn=batch_collate_fn)
    num_iterations = len(curriculum_data_loader)

    model = torch.nn.DataParallel(Policy_Network().to(device))
    print(f"1: {args.use_mle_only}, 2: {args.use_rl_only}")
    trainer = Trainer(args.use_mle_only, args.use_rl_only, device)
    optimizer = optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.995), eps=1e-8)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[round(0.25 * num_iterations), round(0.5 * num_iterations),
                                                           round(0.75 * num_iterations)], gamma=0.1)

    trainer.train(curriculum_data_loader, model, optimizer, scheduler, tb)

if __name__ == '__main__':
    main(args)
