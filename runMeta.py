from tensorboard_utils import Tensorboard
# from MetaLearning import MetaTrainer, MetaTrainerSingleton
from MetaLearner import MetaTrainer
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
parser.add_argument("--inner_dimension", default=2048, type=int)
parser.add_argument("--meta_batch_size", default=4, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--num_updates", default=10, type=int)
parser.add_argument("--k_shot", default=1, type=int)
parser.add_argument("--query_batch_size", default=10, type=int)
parser.add_argument("--num_iterations", default=100000, type=int)
parser.add_argument("--meta_lr", default=1e-4, type=float)
parser.add_argument("--lr", default=1e-6, type=float)
parser.add_argument("--checkpoint_path", default='./checkpoint-mle.pth', type=str)
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
  tb = Tensorboard(args.exp_name+'_'+str(args.meta_lr)+'_'+str(args.lr), unique_name=args.unique_id)

  data_loader = MetaGeneratorDataset(categories=['algebra', 'arithmetic', 'numbers', 'comparison'], k_shot=args.k_shot, num_iterations=args.num_iterations, query_batch_size=args.query_batch_size)#, shuffle=True, batch_size=1)#iter(DataLoader(MetaGeneratorDataset(categories=['algebra', 'arithmetic', 'numbers', 'comparison'], k_shot=args.k_shot, num_iterations=args.num_iterations, query_batch_size=args.query_batch_size), shuffle=True, batch_size=1))
  # validation_data_loaders = [iter(DataLoader(MetaGeneratorDataset(categories=['calculus', 'measurement', 'polynomials', 'probability'], k_shot=args.k_shot, num_iterations=args.num_iterations, query_batch_size=args.query_batch_size), shuffle=True, batch_size=1))]

  if torch.cuda.is_available:
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
  
  trainer = MetaTrainer(device=device, lr=args.lr, meta_lr=args.meta_lr, tb=tb, checkpoint_path=args.checkpoint_path)
  trainer.train(data_loader, num_updates=args.num_updates, num_iterations=args.num_iterations, meta_batch_size=args.meta_batch_size)

if __name__=='__main__':
  # if args.world_size > 1: set_start_method('spawn')
  main(args)
