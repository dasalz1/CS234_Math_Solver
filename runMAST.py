from tensorboard_utils import Tensorboard
import argparse, random
import numpy as np
import torch
import torch.optim as optim
from MetaLearning import MetaTrainer
from datetime import date
from dataset import NaïveCurriculumDataset
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
    data_loaders = [iter(
        DataLoader(Generator(args.filepath + '/' + dataset, False, k_shot=args.k_shot), shuffle=True, batch_size=1)) for
                    dataset in repo_files[num_validation_repos:102]]

    model_params = (VOCAB_SIZE, VOCAB_SIZE, 0, 0,
                    args.d_word_vec, args.d_word_vec, args.inner_dimension, args.num_layers,
                    args.num_heads, args.key_dimension, args.value_dimension, args.dropout,
                    MAX_QUESTION_SIZE, MAX_QUESTION_SIZE, True, True)
    print(f"1: {args.use_mle_only}, 2: {args.use_rl_only}")
    trainer = MetaTrainer(args.meta_batch_size, device='cpu', model_params=model_params)
    trainer.train(data_loaders, tb, num_updates=args.num_updates)

if __name__ == '__main__':
    main(args)
