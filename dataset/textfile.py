from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader

from random import randint
import torch

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