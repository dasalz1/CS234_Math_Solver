import pandas as pd
import torch, os, collections, six
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset_utils import sample_from_module, _filter_and_flatten, _make_entropy_fn
from transformer.Constants import PAD
from mathematics_dataset.modules import modules
from utils import np_encode_string


class GeneratorDataset(Dataset):
    
    def __init__(self, categories=["algebra", "probability"], difficulty=0.5, num_iterations=12, batch_size=4):
        problems = collections.defaultdict(lambda: [])
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))
        self.num_iterations = int(num_iterations * batch_size)

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx):

        problem = sample_from_module(self.sampled_modules[np.random.randint(0, len(self.sampled_modules), (1))[0]][1], show_dropped=False)[0]
        ques, anws = np_encode_string(str(problem[0])), np_encode_string(str(problem[1])) 

        return ques, anws
        



def batch_collate_fn(values):
    # q, a = values[0], values[1]
    # print(len(values))
    qs = [values[idx][0] for idx in range(len(values))]
    
    ans = [values[idx][1] for idx in range(len(values))]

    return pd.DataFrame(qs).fillna(PAD).values, pd.DataFrame(ans).fillna(PAD).values



# ds = GeneratorDataset(num_iterations=3, batch_size=4)
# d = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=batch_collate_fn) # can have num workers