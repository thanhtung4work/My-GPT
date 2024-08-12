import mlflow
import mlflow.pytorch
import torch
from modules import GPTModel

# Set up MLflow tracking
mlflow.set_experiment("Transformer Training")

# Hyperparameters
batch_size = 64
block_size = 128
n_embd = 16
n_heads = 2
n_layers = 2
dropout = 0.2
learning_rate = 1e-4
max_iters = 8000
eval_interval = 500
eval_iters = 200

# Preprocess text
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings from character to integer and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# Data loading function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

# Estimate loss function
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
        out[split] = losses.mean().item()
    model.train()
    return out

# Training loop with MLflow logging
with mlflow.start_run():
    mlflow.log_params({
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "eval_iters": eval_iters
    })
    
    # Initialize the model
    model = GPTModel(vocab_size, n_embd, block_size, n_heads, dropout, n_layers)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Create an optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # Evaluate loss on training and validation sets at intervals
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            mlflow.log_metrics({"train_loss": losses['train'], "val_loss": losses['val']}, step = iter)

        # Get a batch of data and compute the loss
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final logging and model saving
    mlflow.pytorch.log_model(model, "transformer_model")

# Generate text from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))