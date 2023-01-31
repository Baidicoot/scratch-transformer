import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler, autocast

from utils import Cfg

def optimizer(optim, model, cfg):
    groups = model.parameters()
    if hasattr(model, "optim_groups"):
        groups = model.optim_groups(cfg)
    return optim(groups, **cfg.dict)

class Trainer:
    @staticmethod
    def get_default_config():
        return Cfg(
            # device to train on
            device = 'auto',
            # dataloder parameters
            num_workers = 4,
            # optimizer parameters
            max_iters = None
        )

    def __init__(self, config, model, train_dataset, optimizer, lossfn = None, sampler = None):
        self.config = config
        self.model = model

        self.optimizer = optimizer

        self.lossfn = lossfn
        if lossfn is None:
            self.lossfn = lambda outputs, targets : F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1)
        
        self.sampler = sampler
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print("using %.2fM parameter model" % (n_params/1e6,))
        print("running on", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler = self.sampler,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)

        scaler = GradScaler()

        while True:
            self.optimizer.zero_grad()

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            if self.device == "cuda":
                with autocast():
                    logits = model(x)
                    self.loss = self.lossfn(logits, y)
            else:
                logits = model(x)
                self.loss = self.lossfn(logits, y)

            # backprop and update the parameters
            scaler.scale(self.loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break