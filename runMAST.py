from tensorboard_utils import Tensorboard
import argparse, random
import numpy as np
import torch
import torch.optim as optim
from MetaLearning import MetaTrainer
from datetime import date
from dataset import MetaGeneratorDataset
from dataset import batch_collate_fn
from parameters import VOCAB_SIZE, MAX_QUESTION_SIZE
from generator import Generator
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='EditorPairTrain', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--use_mle_only", default=False, action='store_true')
parser.add_argument("--use_rl_only", default=False, action='store_true')
args = parser.parse_args()

mast_batches = 128

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


    for meta_batch_num in range(mast_batches):
        teacher = [iter(DataLoader(MetaGeneratorDataset(categories=["algebra", "probability", "geometry", "calculus"]),
                                   shuffle=True, batch_size=1))]
        # Train student
        student = MetaTrainer(args.meta_batch_size / mast_batches, device=device)
        student.train(teacher, tb, num_updates=args.num_updates)
        # Train teacher
        teacher = [iter(DataLoader(MetaGeneratorDataset(categories=["algebra", "probability", "geometry", "calculus"]),
                                   shuffle=True, batch_size=1))]

if __name__ == '__main__':
    main(args)
