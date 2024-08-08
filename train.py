from modules import GPTModel

# Hyperparams
batchsize = 32
blocksize = 16
n_embd = 32
n_heads = 4
n_layers = 4
dropout = 0.2

# Preprocessing text here


model = GPTModel(100, 32, 32, 4, 0.2, 3)