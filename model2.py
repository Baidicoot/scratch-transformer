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

class MultiheadSelfAttention(nn.Module):
    def __init__(self, cfg):
        super(MultiheadSelfAttention, self).__init__()
        # q,k,v for every head, concatenated
        self.queries = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.keys = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.values = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.attn_drop = nn.Dropout(cfg.p_drop)
        self.out_drop = nn.Dropout(cfg.p_drop)

        self.cfg = cfg
        self.head_dim = cfg.embed_dim // cfg.num_heads

        self.register_buffer("mask",
            torch.tril(torch.ones(cfg.seq_len, cfg.seq_len))
                .view(1, 1, cfg.seq_len, cfg.seq_len))

    # expects inputs of dim (batch_size, seq_len, embed_dim)
    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()

        k = self.keys(inputs).view(batch_size, seq_len, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        q = self.queries(inputs).view(batch_size, seq_len, self.cfg.num_heads, self.head_dim).transpose(1, 2)
        v = self.values(inputs).view(batch_size, seq_len, self.cfg.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.head_dim**-0.5
        attn = attn.masked_fill(torch.eq(self.mask[:, :, :seq_len, :seq_len], 0), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.cfg.embed_dim)
        out = self.out_drop(self.out_proj(out))

        return out

# transformer block
class Block(nn.Module):
    def __init__(self, cfg):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = MultiheadSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = FeedForwardNetwork(cfg)
        self.cfg = cfg
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x

# decoder-only transformer
class GPT(nn.Module):
    def __init__(self, cfg):
        super(GPT, self).__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.p_drop)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.embed_dim)
        self.unembed = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        self.cfg = cfg
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print("using %.2fM parameter model" % (n_params/1e6,))
    
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
        
        # gpt-2 inspired madness
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02*(2 * self.cfg.n_layers)**-0.5)

    def forward(self, inputs, targets=None):
        positions = pos = torch.arange(0, inputs.size(1)).unsqueeze(0)

        outputs = self.drop(self.embed(inputs)) + self.pos_embed(positions)

        for layer in self.layers:
            outputs = layer(outputs)

        outputs = self.unembed(self.ln(outputs))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1), ignore_index=-1)
        
        return outputs, loss

    def configure_optimizers(self, optim_cfg):
        # want to NOT decay biases, or anything from LayerNorm or Embedding
        # for some reason
        decay = set()
        no_decay = set()

        whitelist_decay_modules = (nn.Linear, )
        blacklist_decay_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_decay_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_decay_modules):
                    no_decay.add(fpn)
        
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