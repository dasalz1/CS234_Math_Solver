from old_dataset import MathDatasetManager, question_answer_to_batch_collate_fn
from transformer.Models import PositionalEncoding
from torch.utils import data
from A3C import Policy_Network
from training import Trainer
import torch.optim as optim
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_mle", default=False, action='store_true')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = parser.parse_args()

mdsmgr = MathDatasetManager("mathematics_dataset-v1.0")

categories = mdsmgr.get_categories()
types = mdsmgr.get_types()

# full dataset
ds = mdsmgr.build_dataset_from_categories_and_types(categories, types)
# ds = mdsmgr.build_dataset_from_module('algebra', 'linear_1d', 'train-easy')

train_loader = data.DataLoader(
    ds, batch_size=128, shuffle=True,#num_workers=4,
    collate_fn=question_answer_to_batch_collate_fn)

model = torch.nn.DataParallel(Policy_Network().to(device))
trainer = Trainer(args.use_mle, device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

trainer.train(train_loader, model, optimizer)
