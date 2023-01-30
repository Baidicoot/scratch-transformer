# TODO:
# - use oliver's tensor cores
# - allow model saving/loading
# - BPE or equiv.

from model2 import GPT
from trainer import Trainer

import pickle

from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

from utils import Cfg as Cfg

import torch
from random import randint
import numpy as np

class FileCrawler:
    def __init__(self, f, seq_len):
        self.f = f
        self.seq_len = seq_len
    
    def __next__(self):
        chs = self.f.read(self.seq_len+1)
        if len(chs) < self.seq_len:
            raise StopIteration
        a = torch.tensor(list(map(ord, chs))).long()
        return (a[:-1], a[1:])

class TextDataset(IterableDataset):
    def __init__(self, file, seq_len):
        self.filepath = file
        self.seq_len = seq_len
        
        self.vocab_size = 256

    def __iter__(self):
        f = open(self.filepath)
        f.read(randint(0,self.seq_len))
        return FileCrawler(open(self.filepath), self.seq_len)


train_dataset = TextDataset("datasets/shakespeare.txt", 256)

model_config = Cfg(embed_dim=128, n_layers=6, num_heads=4, seq_len=256, p_drop=0.1)
model_config.vocab_size = train_dataset.vocab_size

model = GPT(model_config)

train_config = Trainer.get_default_config()
train_config.max_iters = 10000
trainer = Trainer(train_config, model, train_dataset)

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

model.eval()

x = model.generate(torch.tensor([list(map(ord, "hark, "))]).to(trainer.device), 200)

print("".join(map(chr,x.view(x.size(1)))))