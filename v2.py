import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
block_size = 256
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

data = torch.tensor([chars.index(c) for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(num_heads)])
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config['n_embd']
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config['dropout'])
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.sa = MultiHeadAttention(config['n_head'], head_size, config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        vocab_size = len(vocab)
        self.token_embedding_table = nn.Embedding(vocab_size, config['n_embd'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
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
            idx_cond = idx[:, -self.config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def decode(self, idx_list):
        return ''.join([self.itos[i] for i in idx_list])

config = {
    'n_embd': n_embd,
    'n_head': n_head,
    'n_layer': n_layer,
    'block_size': block_size,
    'dropout': dropout
}

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train():
    model = BigramLanguageModel(chars, config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model, f'model-{max_iters}.pt')

if __name__ == '__main__':
    train()