import mlflow
import torch

from modules import GPTModel
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("Transformer training!")


# Hyperparams
batchsize = 32
blocksize = 64
n_embd = 64
n_heads = 2
n_layers = 2
dropout = 0.2
learning_rate=1e-4
max_iters = 1000
eval_interval = 500
eval_iters = 200

# Preprocessing text here
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Uniqe chars in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping from char to int
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - blocksize, (batchsize,))
    x = torch.stack([data[i:i+blocksize] for i in ix])
    y = torch.stack([data[i+1:i+blocksize+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

with mlflow.start_run() as run:
    mlflow.log_params({
        "batchsize": batchsize,
        "blocksize": blocksize,
        "n_embd": n_embd,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters
    })
    
    # Create GPT model
    model = GPTModel(vocab_size, n_embd, blocksize, n_heads, dropout, n_layers)
    print(sum(p.numel() for p in model.parameters()), ' parameters')


    # Create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            mlflow.log_metrics({"train_loss":  losses['train'], "val_los": losses['val']})

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    mlflow.log_metrics({"train_loss":  losses['train'], "val_los": losses['val']})
    mlflow.pytorch.log_model(
        model, "transformer_model"
    )

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))