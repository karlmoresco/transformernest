import numpy as np
from transformernest.config import Config


class Trainer:
    def __init__(
            self,
            model,  # Model to train
            dataset,  # Dataset to train on
            config: Config,  # configuration
            batch_size: int = 32,  # Number of training examples per batch
            lr: float = 2e-4,  # Learning rate for weight updates
            epochs: int = 8  # Number of passes over the dataset
    ):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.seq_len = self.config.seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

    def batches(self, tokenized_corpus):
        n_sequences = len(tokenized_corpus) - self.seq_len

        # Make training examples with stride = 1
        x_sequences = np.array([tokenized_corpus[i:i+self.seq_len] for i in range(n_sequences)])
        y_sequences = np.array([tokenized_corpus[i+1:i+self.seq_len+1] for i in range(n_sequences)])

        n_batches = x_sequences.shape[0] // self.batch_size

        x_batches = np.array_split(x_sequences[:n_batches*self.batch_size], n_batches)
        y_batches = np.array_split(y_sequences[:n_batches*self.batch_size], n_batches)

        return x_batches, y_batches
