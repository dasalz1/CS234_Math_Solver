import numpy as np
import glob
import pandas as pd
import torch
from torch.utils import data
from transformer.Constants import PAD, UNK, BOS, EOS
from pathlib import Path
from utils import np_encode_string, np_decode_string, question_to_batch_collate_fn, question_answer_to_batch_collate_fn

class LazyFileMathDataset(data.Dataset):
    """Stream loads math dataset file in a lazy way (optional)
    pandas is used for naive streaming as Python doesn't provide any better tool for that critical feature"""
    def __init__(self, file, lazy_load=False, max_elements=None, log=False):
        self.file = Path(file)
        self.lazy_load = lazy_load
        self.max_elements = max_elements

        fn = self.file.name.replace(".txt", "")
        self.category, self.module = fn.split("__")

        if not self.lazy_load:
            self.build_dataset()
            if log:
                print(f"Initialized MathDataset with file {self.file} (category:{self.category}, module:{self.module}) containing {self.qas.shape[0]} pairs of questions/answers")
        else:
            self.qas = None
            if log:
                print(f"Initialized MathDataset with file {self.file} (category:{self.category}, module:{self.module}) in lazy mode")

      
    def _read_build_dataset(self):
        self.df = pd.read_csv(self.file, header=None, sep='\n', names=['qa'], engine='c')
        self._build_dataset()
    
    def _build_dataset(self):
        if self.max_elements is not None:
            self.df_max = self.df.iloc[0:self.max_elements*2]
        else:
            self.df_max = self.df
        self.questions = self.df_max[0::2]
        self.questions.reset_index(inplace=True, drop=True)
        self.questions.rename(columns={ "qa" : "questions" }, inplace=True)
        self.answers = self.df_max[1::2]
        self.answers.reset_index(inplace=True, drop=True)
        self.answers.rename(columns={ "qa" : "answers" }, inplace=True)
        self.qas = pd.concat([self.questions, self.answers], axis=1)
        
    def set_max_elements(self, max_elements):
        self.max_elements = max_elements
        if self.qas is None:
            self._read_build_dataset()
        else:
            self._build_dataset()
        
    def __getitem__(self, idx):
        if self.qas is None:
            self._read_build_dataset()            
        question, answer = self.qas.iloc[idx]
        return {
            "q": question, "q_enc": np_encode_string(question),
            "a": answer, "a_enc": np_encode_string(answer),
        }

    def __len__(self):
        if self.qas is None:
           self._read_build_dataset() 
        return self.qas.shape[0]
    

class MathDatasetManager(data.Dataset):
    """A Math Dataset manager starting at root directory (like v1.0) to extract files and build torch datasets
    in a lazy loading and streamed way based on specific types/categories/modules presented in paper.
    
    It indexes difficulty/use-case types:
        - train-easy
        - train-medium
        - train-hard
        - interpolate
        - extrapolate
    
    and all categories:
        - algebra
        - numbers
        - polynomials
        - arithmetic
        - measurement
        - comparison
        - probability
        - calculus
        
    and all modules in those categories:
        - mul
        - add_or_sub_in_base
        - simplify_surd
        - mul_div_multiple
        - mixed
        - nearest_integer_root
        - div
        - add_or_sub
        - add_sub_multiple
        - add_sub_multiple_longer
        - mul_div_multiple_longer
        - div_big
        - mul_big
        - mixed_longer
        - add_or_sub_big
        - etc...
    """
    def __init__(self, root_dir, log=False):
        self.root_dir = Path(root_dir)

        self.dirs = {
            "train-easy" : self.root_dir / "train-easy",
            "train-medium" : self.root_dir / "train-medium",
            "train-hard" : self.root_dir / "train-hard",
            "interpolate" : self.root_dir / "interpolate",
            "extrapolate" : self.root_dir / "extrapolate",
        }
        
        self.dfs = {}
                
        for k, dir in self.dirs.items():
            files = [ff for ff in glob.glob(str(dir) + "/**/*.txt", recursive=True)]
            for f in files:
                ds = LazyFileMathDataset(f, lazy_load = True, log=log)
                if ds.category not in self.dfs:
                    self.dfs[ds.category] = {}
                if ds.module not in self.dfs[ds.category]:
                    self.dfs[ds.category][ds.module] = {
                        "easy" : {}, "medium" : {}, "hard" : {},
                        "interpolate": {}, "extrapolate": {}
                    }

                self.dfs[ds.category][ds.module][k] = ds                    

        print(f"initialized MultiFilesMathDataset with categories {list(self.dfs.keys())} and types {list(self.dirs.keys())}")

    def get_types(self):
        """retrieves all math typesfor this multi-file dataset"""
        return self.dirs.keys()            
        
    def get_categories(self):
        """retrieves all math problem categories in this multi-file dataset"""
        return self.dfs.keys()
    
    def get_modules_for_category(self, c):
        """retrieves all mathematical modules in a math problem category"""
        return self.dfs[c].keys()
    
    def _build_datasets_from_category(self, category, typ, max_elements=None):
        ds = []
        for k, m in self.dfs[category].items():
            if typ in m:
                m[typ].set_max_elements(max_elements)
                ds.append(m[typ])
                print(f"added module {category}/{k}/{typ}")
        return ds
        
    def build_dataset_from_category(self, category, typ, max_elements=None):
        """Build a dataset for all modules in a category"""
        print(f"adding category {category}/../{typ}")
        ds = self._build_datasets_from_category(category, typ, max_elements=max_elements)
        return data.ConcatDataset(ds)

    def build_dataset_from_category_all_types(self, category, types, max_elements=None):
        """Build a dataset for all modules in a category with all types"""
        ds = []
        for typ in types:
            try:
                print(f"adding category {category}/../{typ}")
                dss = self._build_datasets_from_category(category, typ, max_elements=max_elements)
                ds.extend(dss)
            except:
                continue
        return data.ConcatDataset(ds)
    
    def build_dataset_from_categories(self, categories, typ, max_elements=None):
        """Build a dataset for all modules in several categories"""
        ds = []
        for c in categories:
            print(f"adding category {c}/../{typ}")
            dss = self._build_datasets_from_category(c, typ, max_elements=max_elements)
            ds.extend(dss)
        return data.ConcatDataset(ds)

    def build_dataset_from_module(self, category, module, typ, max_elements=None):
        """Build a dataset from a single module in a category"""
        self.dfs[category][module][typ].set_max_elements(max_elements)
        return self.dfs[category][module][typ]

    def build_dataset_from_modules(self, category, modules, typ, max_elements=None):
        """Build a dataset from several modules in a category"""
        ds = []
        for module in modules:
            self.dfs[category][module][typ].set_max_elements(max_elements)
            ds.append(self.dfs[category][module][typ])
        return data.ConcatDataset(ds)
    
    def build_dataset_from_categories_and_types(self, categories, types, max_elements=None):
        """Build a dataset for all modules in several categories"""
        ds = []
        for c in categories:
            for typ in types:
                try:
                    print(f"adding category {c}/../{typ}")
                    dss = self._build_datasets_from_category(c, typ, max_elements=max_elements)
                    ds.extend(dss)
                except:
                    continue
        return data.ConcatDataset(ds)