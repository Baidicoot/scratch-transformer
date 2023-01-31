# TODO:
# - use oliver's tensor cores
# - allow model saving/loading
# - BPE or equiv.

import torch
import torch.optim as optim

from utils import Cfg
from trainer import Trainer, optimizer

from dataset.textfile import TextDataset

from model.gpt import GPT

torch.cuda.empty_cache()

train_dataset = TextDataset("datas/shakespeare.txt", 512)

model_config = Cfg(
    embed_dim=256,
    n_layers=8,
    num_heads=8,
    seq_len=512,
    p_drop=0.1,
    vocab_size = train_dataset.vocab_size
)

model = GPT(model_config)

train_config = Cfg(
    device = 'auto',
    max_iters = 20000,
    batch_size = 64,
    num_workers = 1
)

optimizer_config = Cfg(
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay = 0.1
)

optimizer = optimizer(optim.AdamW, model, optimizer_config)

trainer = Trainer(train_config, model, train_dataset, optimizer)

def batch_end_callback(trainer):
    if trainer.iter_num % 50 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
    if trainer.iter_num % 2000 == 0:
        torch.save(model.state_dict(), f"trained/gpt-shakespeare/iter_{trainer.iter_num}.pt")

trainer.set_callback('on_batch_end', batch_end_callback)

trainer.run()

torch.save(model.state_dict(), f"trained/gpt-shakespeare/complete.pt")

model.eval()

x = model.generate(torch.tensor([list(map(ord, "hark, "))]).to(trainer.device), 500)

print("".join(map(chr,x.view(x.size(1)))))