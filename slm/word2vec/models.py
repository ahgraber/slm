import logging

import torch
import torch.nn as nn

from slm.word2vec import vocab

# %%
logger = logging.getLogger(__name__)

# %%
# For all the following models, the training complexity is proportional to
#    `O = E × T × Q`
# where E is number of the training epochs,
# T is the number of the words in the training set and
# Q is defined further for each model architecture.
# Common choice is E = 3 − 50 and T up to one billion.

EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1
NEG_SAMPLE_POWER = 0.75
NEG_SAMPLE_COUNT = 5


class CBOW_Model(nn.Module):  # NOQA: N801
    """Model to learn Word2Vec embedding space."""

    def __init__(self, vocab: vocab.Vocab):
        super().__init__()
        self.vocab = vocab
        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab.size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=self.vocab.size,
        )

    def forward(self, inputs_):
        """Forward pass."""
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x
