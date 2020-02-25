from dataset import MathDatasetManager, question_answer_to_batch_collate_fn
from transformer.Models import PositionalEncoding
from torch.utils import data
from A3C import Policy_Network
from training import Trainer
import torch.optim as optim

mdsmgr = MathDatasetManager("mathematics_dataset-v1.0")
ds = mdsmgr.build_dataset_from_module('algebra', 'linear_1d', 'train-easy')

train_loader = data.DataLoader(
    ds, batch_size=128, shuffle=True,#num_workers=4,
    collate_fn=question_answer_to_batch_collate_fn)

model = Policy_Network()
trainer = Trainer()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

trainer.train(train_loader, model, .99, optimizer)