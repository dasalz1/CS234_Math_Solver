from old_dataset import MathDatasetManager, question_answer_to_batch_collate_fn
from transformer.Models import PositionalEncoding
from torch.utils import data
from A3C import Policy_Network
from training import Trainer
import torch.optim as optim
import torch

from tensorboard_utils import Tensorboard
from tensorboard_utils import tensorboard_event_accumulator

from parameters import CUDA_VISIBLE_DEVICES

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_mle_only", default=False, action='store_true')
parser.add_argument("--use_rl_only", default=False, action='store_true')

if torch.cuda.device_count() > 1:
  print("Using", torch.cuda.device_count(), "GPUs...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

args = parser.parse_args()

mdsmgr = MathDatasetManager("mathematics_dataset-v1.0")


exp_name = "math_ds_full_mle"
unique_id = "2020-25-02"
tb = Tensorboard(exp_name, unique_name=unique_id)

categories = mdsmgr.get_categories()
types = mdsmgr.get_types()

# full dataset
ds = mdsmgr.build_dataset_from_categories_and_types(categories, types)
# ds = mdsmgr.build_dataset_from_module('algebra', 'linear_1d', 'train-easy')

train_loader = data.DataLoader(
    ds, batch_size=1024, shuffle=True,#num_workers=4,
    collate_fn=question_answer_to_batch_collate_fn, num_workers=0)

num_iterations = len(train_loader)

model = torch.nn.DataParallel(Policy_Network().to(device))
trainer = Trainer(args.use_mle_only, args.use_rl_only, device)
optimizer = optim.Adam(model.parameters(), lr=6e-4, betas=(0.9, 0.995), eps=1e-8)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(0.25 * num_iterations), round(0.5 * num_iterations), round(0.75 * num_iterations)], gamma=0.1)

trainer.train(train_loader, model, optimizer, scheduler, tb)
