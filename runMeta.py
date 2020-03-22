from tensorboard_utils import Tensorboard
from MetaLearning import MetaTrainer, MetaTrainerSingleton
import argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
# from old_dataset import MathDatasetManager, question_answer_to_batch_collate_fn
from dataset import MetaGeneratorDataset
from datetime import date
from parameters import VOCAB_SIZE, MAX_QUESTION_SIZE

from torch.multiprocessing import set_start_method

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='MathSolver', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--num_layers", default=6, type=int)
parser.add_argument("--num_heads", default=8, type=int)
parser.add_argument("--key_dimension", default=64, type=int)
parser.add_argument("--value_dimension", default=64, type=int)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--d_word_vec", default=512, type=int)
parser.add_argument("--inner_dimension", default=2048, type=int)
parser.add_argument("--world_size", default=4, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--num_updates", default=10, type=int)
parser.add_argument("--k_shot", default=1, type=int)
parser.add_argument("--query_batch_size", default=10, type=int)
parser.add_argument("--num_iterations", default=100000, type=int)
args = parser.parse_args()

# mdsmgr = MathDatasetManager("mathematics_dataset-v1.0")
# ds = MetaRepo('repo_files/beaker_line_pairs.csv', False)
# d = DataLoader(ds, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "redmond")
def main(args):
  random.seed(12324)
  np.random.seed(12324)
  torch.manual_seed(12324)

  num_validation_repos = 100
  tb = Tensorboard(args.exp_name, unique_name=args.unique_id)

  data_loader = MetaGeneratorDataset(categories=['algebra', 'arithmetic', 'numbers', 'comparison'], k_shot=args.k_shot, num_iterations=args.num_iterations, query_batch_size=args.query_batch_size)#, shuffle=True, batch_size=1)#iter(DataLoader(MetaGeneratorDataset(categories=['algebra', 'arithmetic', 'numbers', 'comparison'], k_shot=args.k_shot, num_iterations=args.num_iterations, query_batch_size=args.query_batch_size), shuffle=True, batch_size=1))
  # validation_data_loaders = [iter(DataLoader(MetaGeneratorDataset(categories=['calculus', 'measurement', 'polynomials', 'probability'], k_shot=args.k_shot, num_iterations=args.num_iterations, query_batch_size=args.query_batch_size), shuffle=True, batch_size=1))]

  if torch.cuda.is_available:
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False

  
  if args.world_size == 1:
    trainer = MetaTrainerSingleton(args.world_size, device=device, tb=tb)
  else:
    trainer = MetaTrainer(args.world_size, device=device, tb=tb)
  trainer.train(data_loader, num_updates=args.num_updates, num_iterations=args.num_iterations)

if __name__=='__main__':
  if args.world_size > 1: set_start_method('spawn')
  main(args)
