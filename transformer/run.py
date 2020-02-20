from dataset import MathDatasetManager
from transformer.Models import PositionalEncoding
from A3C import Policy_Network

mdsmgr = MathDatasetManager("mathematics_dataset-v1.0")
ds = mdsmgr.build_dataset_from_module('arithmetic', 'add_or_sub', 'train-easy')


train_loader = data.DataLoader(
    ds, batch_size=128, shuffle=True,#num_workers=4,
    collate_fn=question_answer_to_batch_collate_fn)


model = Policy_Network()
trainer = Trainer()
optimizer = optim.Adam(model.parameters(), lr=1e-2)