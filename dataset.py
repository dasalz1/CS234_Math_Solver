import pandas as pd
import torch, os, collections, six
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset_utils import sample_from_module, _filter_and_flatten, _make_entropy_fn
from transformer.Constants import PAD, EOS, BOS, UNK
from mathematics_dataset.modules import modules
from utils import np_encode_string
from threading import Thread
from parameters import MAX_QUESTION_SIZE, MAX_ANSWER_SIZE


class GeneratorDataset(Dataset):
    
    def __init__(self, categories=["algebra__linear_1d", "probability"], difficulty=0.5, num_iterations=12, batch_size=4, refresh_rate=10000):
        super(GeneratorDataset, self).__init__()
        self.categories = categories
        self._create_modules()
        self.num_iterations = num_iterations
        self.refresh_rate = refresh_rate
        self.iter = 1

    def __len__(self):
        return self.num_iterations

    def _create_modules(self):
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(self.categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))

    def __getitem__(self, idx):
        if self.iter % self.refresh_rate == 0: self._create_modules()
        try:
            problem = sample_from_module(self.sampled_modules[np.random.randint(0, len(self.sampled_modules), (1))[0]][1], show_dropped=False)[0]
            # converts to tokens and adds BOS and EOS tokens
            ques, anws = np_encode_string(str(problem[0])), np_encode_string(str(problem[1])) 
        except:
            return self.__getitem__(idx)

        self.iter += 1
        return ques, anws

class MetaGeneratorDataset(Dataset):
    
    def __init__(self, categories=["algebra__linear_1d", "probability"], difficulty=0.5, num_iterations=32, query_batch_size=4, k_shot=5, refresh_rate=100):
        super(MetaGeneratorDataset, self).__init__()
        self._create_modules()
        self.num_iterations = num_iterations
        self.k_shot = k_shot
        self.categories = categories
        self.query_batch_size = query_batch_size
        self.iter = 1

    def __len__(self):
        return self.num_iterations

    def _create_modules(self):
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(self.categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))

    def supportProblem(self, sample_module, problem_data):
        support_problem = sample_from_module(sample_module, show_dropped=False)[0]
        support_problem = (np_encode_string(str(support_problem[0])), np_encode_string(str(support_problem[1])))
        problem_data.append(support_problem)

    def __getitem__(self, idx):
        if self.iter % self.refresh_rate == 0: self._create_modules()
        try:
            query_data = []
            supp_data = []
            problem_threads = []
            sample_module = self.sampled_modules[np.random.randint(0, len(self.sampled_modules))][1]

            for _ in range(self.query_batch_size):
                problem_threads.append(Thread(target=self.supportProblem, args=(sample_module, query_data,)))
                problem_threads[-1].start()

            for _ in range(self.k_shot):
                problem_threads.append(Thread(target=self.supportProblem, args=(sample_module, supp_data,)))
                problem_threads[-1].start()
            
            # problem = sample_from_module(sample_module, show_dropped=False)[0]

            # support_ques = torch.LongTensor(pd.DataFrame(np_encode_string(str(problem[0]))).fillna(PAD).values.reshape(1, -1))
            # support_ans = torch.LongTensor(pd.DataFrame(np_encode_string(str(problem[1]))).fillna(PAD).values.reshape(1, -1))
            
            for p_t in problem_threads:
                p_t.join()

            if len(query_data) < self.query_batch_size or len(supp_data) < self.k_shot:
                return self.__getitem__(0)

            query_ques, query_ans = zip(*query_data)

            query_ques = pd.DataFrame(query_ques).fillna(PAD).values.reshape(self.query_batch_size, -1)
            query_ans = pd.DataFrame(query_ans).fillna(PAD).values.reshape(self.query_batch_size, -1)

            support_ques, support_ans = zip(*supp_data)

            support_ques = pd.DataFrame(support_ques).fillna(PAD).values.reshape(self.k_shot, -1)
            support_ans = pd.DataFrame(support_ans).fillna(PAD).values.reshape(self.k_shot, -1)
        except:
            return self.__getitem__(0)
        
        self.iter += 1
        
        return support_ques, support_ans, query_ques, query_ans

    def get_sample(self):
        return self.__getitem__(0)

class NaïveCurriculumDataset(Dataset):

    def __init__(self, categories=["algebra", "probability"], num_iterations=12, batch_size=4):
        super(NaïveCurriculumDataset, self).__init__()
        self.categories = categories
        self.total_iterations = int(num_iterations * batch_size)
        self.current_iteration = 0

    def __len__(self):
        return self.total_iterations

    def __getitem__(self, idx):
        difficulty = self.current_iteration / self.total_iterations
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(self.categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))

        problem = sample_from_module(self.sampled_modules[np.random.randint(0, len(self.sampled_modules), (1))[0]][1], show_dropped=False)[0]
        # converts to tokens and adds BOS and EOS tokens
        ques, anws = np_encode_string(str(problem[0])), np_encode_string(str(problem[1]))

        self.current_iteration += 1
        return ques, anws
        
class DeepCurriculumDataset(Dataset):
    def __init__(self, categories, mean_accuracy_by_category, difficulty=0.5, num_iterations = 12, batch_size = 4, model = None):
        assert(len(self.categories) == len(mean_accuracy_by_category))
        self.categories = categories
        self.total_iterations = int(num_iterations * batch_size)
        self.current_iteration = 0

        assert (np.sum(self.category_probabilities) == 1)
        self.category_probabilities = self.model.forward(mean_accuracy_by_category)
        initial_modules = modules.train(_make_entropy_fn(difficulty, 1))
        filtered_modules = _filter_and_flatten(categories, initial_modules)
        self.sampled_modules = list(six.iteritems(filtered_modules))

    def __len__(self):
        return self.total_iterations

    def __getitem__(self, idx):
        # TODO: Following line could have list shaping/access issues? Should review torch.multinomial and sample_from_module functions definitions
        # Also, can Torch backprop through multinomial stochasticity in the first place?
        problem = sample_from_module(self.sampled_modules[torch.multinomial(self.category_probabilities, 1)[0]], show_dropped=False)[0]
        # converts to tokens and adds BOS and EOS tokens
        ques, anws = np_encode_string(str(problem[0])), np_encode_string(str(problem[1]))

        return ques, anws


def batch_collate_fn(values):
    # qs = [values[idx][0] for idx in range(len(values))]
    
    # ans = [values[idx][1] for idx in range(len(values))]
    qs, ans = zip(*values)

    return torch.LongTensor(pd.DataFrame(qs).fillna(PAD).values), torch.LongTensor(pd.DataFrame(ans).fillna(PAD).values)



# ds = GeneratorDataset(num_iterations=3, batch_size=4)
# d = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=batch_collate_fn) # can have num workers
