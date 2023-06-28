# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 15:06:13 2023

@author: cormac
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
set_seed(3407)

class TextDataset(Dataset):
    """ 
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, data, block_size): 
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.s2i = { ch:i for i,ch in enumerate(chars) }
        self.i2s = { i:ch for i,ch in enumerate(chars) }
        self.data = data
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data_size = data_size
    
    def __len__(self):
        return self.data_size # ...
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.block_size

    def __getitem__(self, idx): 
        
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.s2i[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
       
        return x, y
    


# with open('c:/Data/Text/Books/shakespeare.txt.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
# tds = textDS.TextDataset(text[0:10000])
    
# from mingpt.model import GPT
# model_config = GPT.get_default_config()
# model_config.model_type = 'gpt-mini'
# model_config.vocab_size = 50257 # openai's model vocabulary
# model_config.block_size = 1024  # openai's model block_size (i.e. input context length)
# model = GPT(model_config)
    
# from mingpt.trainer import Trainer
# train_config = Trainer.get_default_config()
# train_config.learning_rate = 5e-4 # many possible options, see the file
# train_config.max_iters = 100
# train_config.batch_size = 32
# trainer = Trainer(train_config, model, tds)
# trainer.run()

# context = text[200:220]
# x = torch.tensor([tds.s2i[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
# model.generate(idx = x , max_new_tokens=10)

# [tds.i2s[i] for i in [ 1, 41, 44, 45, 53,  1, 12, 31, 39, 51, 49,  1, 20, 31, 48, 33, 39, 51,
#          49,  1, 50, 38, 38, 38, 38, 38, 45, 51, 38, 45]]