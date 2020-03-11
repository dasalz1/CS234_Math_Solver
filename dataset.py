import pandas as pd
import torch, os, collections, six
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset_utils import sample_from_module, _filter_and_flatten, _make_entropy_fn
from transformer.Constants import PAD, EOS, BOS, UNK
from mathematics_dataset.modules import modules
from utils import np_encode_string
from threading import Thread


class GeneratorDataset(Dataset):
    
    def __init__(self, categories=["algebra__linear_1d", "probability"], difficulty=0.5, num_iterations=12, batch_size=4):
        problems = collections.defaultdict(lambda: [])
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))
        self.num_iterations = int(num_iterations * batch_size)

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx):

        problem = sample_from_module(self.sampled_modules[np.random.randint(0, len(self.sampled_modules), (1))[0]][1], show_dropped=False)[0]
        # converts to tokens and adds BOS and EOS tokens
        ques, anws = np_encode_string(str(problem[0])), np_encode_string(str(problem[1])) 

        return ques, anws

class MetaGeneratorDataset(Dataset):
    
    def __init__(self, categories=["algebra__linear_1d", "probability"], difficulty=0.5, num_iterations=12, batch_size=4, k_shot=5):
        problems = collections.defaultdict(lambda: [])
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))
        self.num_iterations = int(num_iterations * batch_size)
        self.k_shot = k_shot

    def __len__(self):
        return self.num_iterations

    def supportProblem(self, sample_module, problem_data):
        support_problem = sample_from_module(sample_module, show_dropped=False)[0]
        support_problem = (np_encode_string(str(support_problem[0])), np_encode_string(str(support_problem[1])))
        problem_data.append(support_problem)

    def __getitem__(self, idx):
        problem_data = []
        problem_threads = []
        sample_module = self.sampled_modules[np.random.randint(0, len(self.sampled_modules))][1]
        for _ in range(self.k_shot):
            problem_threads.append(Thread(target=self.supportProblem, args=(sample_module, problem_data,)))
            problem_threads[-1].start()
        
        problem = sample_from_module(sample_module, show_dropped=False)[0]

        query_ques = torch.LongTensor(pd.DataFrame(np_encode_string(str(problem[0]))).fillna(PAD).values.reshape(1, -1))
        query_ans = torch.LongTensor(pd.DataFrame(np_encode_string(str(problem[1]))).fillna(PAD).values.reshape(1, -1))
        for p_t in problem_threads:
            p_t.join()

        support_ques, support_ans = zip(*problem_data)

        support_ques = torch.LongTensor(pd.DataFrame(support_ques).fillna(PAD).values)
        support_ans = torch.LongTensor(pd.DataFrame(support_ans).fillna(PAD).values)
        
        return support_ques, support_ans, query_ques, query_ans

class NaïveCurriculumDataset(Dataset):

    def __init__(self, categories=["algebra", "probability"], num_iterations=12, batch_size=4):
        self.categories = categories
        self.num_iterations = int(num_iterations * batch_size)
        self.current_iteration = 0

    def __len__(self):
        return self.num_iterations

    def __getitem__(self, idx):
        states = self.categories
        difficulty = self.current_iteration / self.num_iterations
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(self.categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))

        problem = sample_from_module(self.sampled_modules[np.random.randint(0, len(self.sampled_modules), (1))[0]][1], show_dropped=False)[0]
        # converts to tokens and adds BOS and EOS tokens
        ques, anws = np_encode_string(str(problem[0])), np_encode_string(str(problem[1]))

        self.current_iteration += 1
        return ques, anws
        



def batch_collate_fn(values):
    # qs = [values[idx][0] for idx in range(len(values))]
    
    # ans = [values[idx][1] for idx in range(len(values))]
    qs, ans = zip(*values)

    return torch.LongTensor(pd.DataFrame(qs).fillna(PAD).values), torch.LongTensor(pd.DataFrame(ans).fillna(PAD).values)



# ds = GeneratorDataset(num_iterations=3, batch_size=4)
# d = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=batch_collate_fn) # can have num workers