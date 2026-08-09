# Hyperparameters

vocab_size = 8000  # Number of tokens in vocabulary post-BPE
seq_len = 128  # Number of tokens in a single training sequence (can be increased to 256)
batch_size = 32  # Number of training examples per batch (can possibly be increased to 64)
d_model = 128  # Vector dimension for token embeddings (can be increased to 256)
feedforward_dim = 512  # Matrix dimension for feedforward layers (can be increased to 1024)
n_layers = 2  # Number of transformer blocks (should be within range 2-8)
n_heads = 4  # Number of attention heads (can be increased to 8)
# d_k = d_model // n_heads (?)

# Attention mechanism: Masked causal multi-head self-attention
# Loss function: Cross-entropy
# Positional encodings: Sinusoidal

# TODO: Experiment with 3 or 4 layers (n_layers)
# TODO: Consider n_heads=8 for richer attention patterns (no overfitting risk, just more fine-grained)
# TODO: How many epochs?
# TODO: Learn about optimizer and learning rates - Adam optimizer, learning rate (1e-4, 2e-4), warmup, cosine annealing schedule, linear warmup, cosine decay ?
# TODO: Add hyperparameter customizability. Implement as model attribute ?
# TODO: Learn CUDA and implement GPU acceleration ?