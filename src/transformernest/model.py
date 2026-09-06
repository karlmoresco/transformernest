import numpy as np
from transformernest.utils import he_init, relu, normal_init
from transformernest.config import Config


class Attention:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.Wq = normal_init()



class MLP:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.d_model = self.config.d_model
        self.d_ff = self.config.d_ff

        self.W1 = he_init(self.d_model, self.d_ff)
        self.b1 = np.zeros(self.d_ff)
        self.W2 = he_init(self.d_ff, self.d_model)
        self.b2 = np.zeros(self.d_model)
    
    def forward(self, X):
        self.X = X
        self.Z1 = X @ self.W1 + self.b1
        self.H1 = relu(self.Z1)
        self.Z2 = self.H1 @ self.W2 + self.b2
        return self.Z2


class Embedding:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        # Initialize embeddings: each row corresponds to a vector embedding
        # (length d_model) of a token in the vocabulary.
        self.E = normal_init(self.config.vocab_size, self.config.d_model, self.config.embedding_std)
        self.token_ids = None

    def forward(self, token_ids):
        # Each value in token_ids selects a row of E:
        self.token_ids = token_ids  # token_ids: shape (batch, seq_len)
        return self.E[token_ids]


class TransformErnest():
    def __init__(
            self,
            config: Config,
    ):
        self.config = config
        self.positional_encoding = self._get_positional_encoding()

    def _get_positional_encoding(self):
        """Sinusoidal positional encoder"""
        seq_len = self.config.seq_len
        d_model = self.config.d_model

        pe = np.zeros((seq_len, d_model))
        positions = np.arange(seq_len).reshape(-1, 1)  # shape: (seq_len, 1)

        N = 10_000

        factors = np.exp(-np.log(N) * np.arange(0, d_model, 2) / d_model).reshape(1, -1)  # shape: (1, d_model / 2)
        pe[:, 0::2] = np.sin(positions * factors)
        pe[:, 1::2] = np.cos(positions * factors)

        return pe

    def encode_position(self, sequence):
        return sequence + self.positional_encoding


