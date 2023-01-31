import argparse
import os

from model.gpt import GPT

from utils import Cfg

import torch

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path")

args = parser.parse_args()

model_config = Cfg(
    embed_dim=256,
    n_layers=8,
    num_heads=8,
    seq_len=512,
    p_drop=0.1,
    vocab_size = 256
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GPT(model_config)
model.load_state_dict(torch.load(args.path))
model.to(device)
model.eval()

while True:
    prompt = input("you>")
    prompt = torch.tensor([list(map(ord, prompt))])

    x = model.generate(prompt.to(device), 1000)
    x = "".join(map(chr,x))

    print("shakespeare>", x)