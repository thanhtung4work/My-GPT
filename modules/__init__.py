import torch
from torch import nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, n_embd, headsize, blocksize, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, headsize, bias=False)
        self.query = nn.Linear(n_embd, headsize, bias=False)
        self.value = nn.Linear(n_embd, headsize, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(blocksize, blocksize)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        # (B, T, hs) * (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        # (B, T, hs)
        out = wei @ v
        # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, headsize, blocksize, dropout, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, headsize, blocksize, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(headsize * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        # (B, T, headsize * n_heads)
        out = self.dropout(self.proj(out))
        # (B, T, n_embd)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout()
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads, blocksize, dropout):
        super().__init__()
        headsize = n_embd // n_heads
        self.self_attention = MultiHeadAttention(n_embd, headsize, blocksize, dropout, n_heads)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.self_attention(self.ln1(x))
        # (B, T, n_embd)
        x = x + self.feed_forward(self.ln2(x))
        # (B, T, n_embd)
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, blocksize, n_heads, dropout, n_layers):
        self.token_embd = nn.Embedding(vocab_size, n_embd)
        self.position_embd = nn.Embedding(blocksize, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads, blocksize, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        self.blocksize = blocksize
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        B, T = idx.shape

        tok_embd = self.token_embd(idx)
        pos_embd = self.position_embd(torch.argsort(T))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_condf = idx[:, -self.blocksize:]
            logits, loss = self(idx_condf)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim = -1) # (B, T + 1)
        return idx