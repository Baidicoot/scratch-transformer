import torch
import torch.nn as nn
from torch.nn import functional as F

class Cfg:
    def __init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

class FeedForwardNetwork(nn.Module):
    def __init__(self, cfg):
        super(FeedForwardNetwork, self).__init__()
        self.l1 = nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(4 * cfg.embed_dim, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.p_drop)
    
    def forward(self, x):
        return self.drop(self.l2(self.act(self.l1(x))))

# transformer block
class Block(nn.Module):
    def __init__(self, cfg):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, cfg.p_drop, batch_first=True)
        self.ln_2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = FeedForwardNetwork(cfg)
        self.cfg = cfg
    
    def forward(self, x, attn_mask):
        self_attn, _ = self.attn(x, x, x, attn_mask)
        x = x + self.ln_1(self_attn)
        x = x + self.ln_2(self.ffn(x))
        return x

# decoder-only transformer
class GPT(nn.Module):
    def __init__(self, cfg):
        super(GPT, self).__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.seq_len+1, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.p_drop)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.embed_dim)
        self.unembed = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        self.cfg = cfg
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # inputs expected as matrix (batch_size, seq_len)
    def forward(self, inputs, targets=None):

        positions = torch.arange(inputs.size(1)).repeat(inputs.size(0), 1) + 1
        pad_mask = inputs.eq(self.cfg.pad_id)
        positions.masked_fill_(pad_mask, 0)

        outputs = self.drop(self.embed(inputs) + self.pos_embed(positions))

        for layer in self.layers:
            outputs = layer(outputs, pad_mask)

        outputs = self.unembed(self.ln(outputs))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1)
        
        return outputs, loss

    def configure_optimizers(self, optim_cfg):
        # want to NOT decay biases, or anything from LayerNorm or Embedding
        decay = set()
        no_decay = set()

        blacklist_decay_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"): # or isinstance(m, blacklist_decay_modules):
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optim_cfg.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=optim_cfg.lr, betas=optim_cfg.betas)
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0):
        for _ in range(max_new_tokens):
            idx_cropped = idx if idx.size(1) <= self.cfg.seq_len else idx[:, -self.cfg.seq_len:]

            logits, _ = self(idx_cropped)
            logits = logits[:, -1, :] / temp

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx