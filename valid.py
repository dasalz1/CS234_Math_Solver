from old_dataset import MathDatasetManager, question_answer_to_batch_collate_fn
from dataset import GeneratorDataset, batch_collate_fn
from transformer.Models import PositionalEncoding
from torch.utils import data
from A3C import Policy_Network
from training import Trainer
import torch.optim as optim
import torch
import numpy as np
import random

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from parameters import VOCAB_SIZE, MAX_ANSWER_SIZE, MAX_QUESTION_SIZE, CUDA_VISIBLE_DEVICES
from dataset import PAD, EOS, BOS, UNK
import os
import math

from tensorboard_utils import Tensorboard
from tensorboard_utils import tensorboard_event_accumulator

from tqdm import tqdm

import argparse

torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
  random.seed(68492)
  np.random.seed(68492)
  torch.manual_seed(68492)
  parser = argparse.ArgumentParser()
  parser.add_argument("--use_mle_only", default=False, action='store_true')
  parser.add_argument("--use_rl_only", default=False, action='store_true')
  CUDA_VISIBILE_DEVICES = CUDA_VISIBLE_DEVICES[0:2]
  
  if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs...")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)

  args = parser.parse_args()

  mdsmgr = MathDatasetManager("mathematics_dataset-v1.0")

  if args.use_mle_only:
    exp_name = "math_ds_full_mle"
  elif args.use_rl_only:
    exp_name = "math_ds_full_rl"
  else:
    exp_name = "math_ds_full_mle_rl"
  unique_id = "char_rewards-1-2020-11-03_random_seed_24"
  tb = Tensorboard(exp_name, unique_name=unique_id)

  categories = mdsmgr.get_categories()
  types = mdsmgr.get_types()

  batch_size = 1024
  checkpoint_interval = 10000
  epochs = 20
  if args.use_mle_only:
    collate_fn = question_answer_to_batch_collate_fn
    # full dataset
    ds = mdsmgr.build_dataset_from_categories_and_types(categories, types)
    # toy dataset (smaller version)
    # ds = mdsmgr.build_dataset_from_module('algebra', 'linear_1d', 'train-hard')
  else:
    # generator dataset
    batch_size = 1
    num_iterations = 1
    checkpoint_interval = 1000
    epochs = 5000
    collate_fn = batch_collate_fn
    ds = GeneratorDataset(categories=["algebra", "probability"], num_iterations=num_iterations, batch_size=batch_size)

  valid_loader = data.DataLoader(
      ds, batch_size=batch_size, shuffle=True,#num_workers=4,
      collate_fn=collate_fn, num_workers=len(CUDA_VISIBLE_DEVICES))

  num_iterations = len(valid_loader)

  # relying on epoch not number of iterations
  if not args.use_mle_only:
    num_iterations = num_iterations * epochs
  print("Number of iterations set: {}".format(num_iterations))
  
  model = Policy_Network().to(device)
  checkpoint = torch.load("checkpoint.pth")
  model.load_state_dict(checkpoint['model'])
  model.eval()

  res = 0.0
  for epoch in range(epochs):
    n_char_total = 0.0
    n_char_correct = 0.0
    for batch_idx, batch in enumerate(tqdm(valid_loader, mininterval=2, leave=False)):
      batch_qs, batch_as = map(lambda x: x.to(device), batch)
      trg_as = batch_as[:, 1:]
      pred_logits = model.action_transformer(input_ids=batch_qs, decoder_input_ids=batch_as[:, :-1])
      pred_logits = pred_logits.reshape(-1, pred_logits.size(2))
      non_pad_mask = trg_as.ne(PAD)
      n_char = non_pad_mask.sum().item()
      n_correct = pred_logits.max(1)[1].eq(trg_as)
      print("q:", batch_qs)
      print("a:",  batch_as)
      print("pred:", pred_logits.max(1)[1])
      print(n_correct, n_char)
      n_correct = n_correct.masked_select(non_pad_mask).sum().item()
      n_char_correct += n_correct
      n_char_total += n_char
    accuracy = n_char_correct / float(n_char_total)
    res += accuracy
    print("Accuracy @ epoch {}: {}".format(epoch, accuracy))

  print("Validation accuracy: {}".format(res / epochs))
