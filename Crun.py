from tensorboard_utils import Tensorboard
import argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from MathConstants import *
from dataset import MetaGeneratorDataset, GeneratorDataset
from datetime import date
from CMeta import Learner
from A3C import Policy_Network
from curriculumModel import CurriculumNetwork
from torch.multiprocessing import set_start_method
from CTrainer import TeacherTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='MathSolver', type=str)
parser.add_argument("--unique_id", default=str(date.today()), type=str)
parser.add_argument("--op", default='mle', type=str, help='options are "mle", "rl", "meta-mle", "meta-rl"')
parser.add_argument("--K", default=10, type=int)
parser.add_argument("--task_batch_size", default=10, type=int, help='Used as query batch size or task batch size')
parser.add_argument("--num_updates", default=10, type=int)
parser.add_argument("--k_shot", default=1, type=int)
parser.add_argument("--validation_samples", default=1, type=int)
parser.add_argument("--num_iterations", default=100000, type=int)
parser.add_argument("--student_lr", default=1e-4, type=float)
parser.add_argument("--meta_lr", default=1e-4, type=float)
parser.add_argument("--teacher_lr", default=0.1, type=float)
parser.add_argument("--checkpoint_path", default='', type=str)
parser.add_argument("--teacher_network", default=True, action='store_true')
parser.add_argument("--teacher_hidden_size", default=256, type=int)
args = parser.parse_args()

def main(args):
  random.seed(12324)
  np.random.seed(12324)
  torch.manual_seed(12324)


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  tb = Tensorboard(args.exp_name+'_'+args.op+'_'+str(args.student_lr)+('_'+str(args.meta_lr)) if 'meta' in args.op else '', unique_name=args.unique_id)

  train_categories = ['algebra', 'arithmetic', 'numbers', 'comparison']; train_subcategories = []
  for category in train_categories:
    train_subcategories += [category+'_'+subc for subc in subcategories[category]]
  # train_subcategories = [subc for c in [subcategories[name] for name in train_categories] for subc in c]

  num_categories = len(train_subcategories)

  if 'meta' in args.op:
    data_loader = [MetaGeneratorDataset(categories=[subc], k_shot=args.k_shot, 
                            num_iterations=args.num_iterations, query_batch_size=args.task_batch_size) for subc in train_subcategories]

    student_model = Learner(device=device, meta_lr=args.meta_lr, checkpoint_path=args.checkpoint_path)
  else:
    data_loader = [GeneratorDataset(categories=[subc], num_iterations=args.num_iterations, batch_size=args.task_batch_size) for subc in train_subcategories]

    student_model = Policy_Network(data_parallel=False, device=device).to(device)

  if torch.cuda.is_available:
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False
  
  teacher_model = None
  if args.teacher_network:
    teacher_model = CurriculumNetwork(input_size=num_categories, output_size=num_categories, hidden_layer_size=args.teacher_hidden_size)

  student_optimizer = AdamW(student_model.parameters(), args.student_lr)
  trainer = TeacherTrainer(op=args.op, device=device, teacher_network=args.teacher_network, teacher_model=teacher_model,
            student_model = student_model, student_optimizer=student_optimizer, validation_samples=args.validation_samples, teacher_lr=args.teacher_lr, tb=tb)

  trainer.train_teacher(data_loader, K=args.K, task_batch_size=args.task_batch_size, num_categories=num_categories, num_iterations=args.num_iterations)

if __name__=='__main__':
  # if args.world_size > 1: set_start_method('spawn')
  main(args)
