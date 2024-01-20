# %%
import logging
import os
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import numpy.typing as npt

import torch

import matplotlib.pyplot as plt

from slm.word2vec import vocab

# %%
logger = logging.getLogger(__name__)

# %%
N_CONTEXT = 4  # 4 is original w2v for CBOW; 10 for Skipgram
MAX_SEQUENCE_LENGTH = 256  # 0 => ignore
SUBSAMPLE = True
SUBSAMPLE_THRESHOLD = 10e-5


def sliding_window_contexts(seq: list[int], n: int):
    """Create contexts from sliding window."""
    # get all start idx that have sufficient tokens for full context window
    width = n * 2
    if width >= len(seq):
        logger.debug("Warning: context width exceeds bounds of available sequence length.  Returning empty list.")
    valid_starts = len(seq) - width  # start indices allowing sufficient tokens for full context window
    contexts = [seq[start : (start + width + 1)] for start in range(valid_starts)]
    return contexts


def calc_prob(counts: Union[int, npt.NDArray], total_words: Union[int, npt.NDArray], t: float):
    """Calculate likelihood to retain word.

    Each word w_i in the training set is discarded with probability
    `P_discard(w_i) = 1 - sqrt(t / f(w_i))`,
    where `f(w_i)` is the frequency of word `w_i` and
    `t` is a chosen threshold, typically around 10^-5.

    NOTE: the equation used in the c implementation is inverted to be a probability-of-retention:
    REF: https://github.com/tmikolov/word2vec/blob/20c129af10659f7c50e86e3be406df663beff438/word2vec.c#L407
    ```C
    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
    ```

    `t` modifies the threshold where the likelihood exceeds 1, making it useful to
    parameterize how frequent a word must be to be downsampled.
    """
    freqs = counts / total_words
    return (np.sqrt(freqs / t) + 1) * t / freqs


def plot_subsample_weights(vocab: vocab.Vocab):
    """Plot subsample weights across various `t` params."""
    x = np.linspace(
        start=1,
        stop=vocab.wordfreq(vocab[len(vocab.specials)]),  # count of most-frequent word
        num=10_000,
    )
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(x, calc_prob(x, total_words=vocab.counter.total(), t=10e-3), label="t=10e-3")
    ax.plot(x, calc_prob(x, total_words=vocab.counter.total(), t=10e-4), label="t=10e-4")
    ax.plot(x, calc_prob(x, total_words=vocab.counter.total(), t=10e-5), label="t=10e-5")
    ax.axhline(y=1, color="black", linewidth=0.25)
    ax.annotate(
        "Words above y=1 _will_ be subsampled;\nwords below _may_ be subsampled.",
        xy=(0.6, 0.085),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        # family="sans-serif",
        fontsize=12,
    )
    # ax.set_xlim([0, 10e7])
    ax.set_ylim([0, 10])
    plt.legend(loc="upper right")
    plt.show()


def calc_subsample_weights(vocab: vocab.Vocab, t: float) -> dict[str, float]:
    """Calculate subsampling weight for all words in vocab."""
    words, counts = zip(*vocab.counter.most_common(vocab.size))
    word2sswt = dict(
        zip(
            words,
            calc_prob(
                np.array(counts),
                total_words=sum(counts),
                t=t,
            ),
        )
    )

    return word2sswt


def subsample_freq_words(record: dict[str, Any], word2sswt: dict[str, float]) -> list:
    """Probabilistically remove frequent words from sequence _prior to_ generating contexts.

    Parameters
    ----------
    record : dict[str, Any]
        _description_
    word2sswt : dict[str, float]
        Dictionary providing lookup for word: subsample proportion/weight

    Returns
    -------
    list
        Filtered of token ids
    """
    # In large corpora, the most frequent words can easily occur hundreds of millions of times.
    # Such words usually provide less information value than  the rare words.

    subsample_mask = [word2sswt[tok] > random.random() if tok in word2sswt else True for tok in record["tokens"]]

    # we filter based on words (tokens) but return the sequence of token ids
    ids_seq = list(np.array(record["ids"])[subsample_mask])
    return ids_seq


# %%
class CBOW:
    """Batch / collate logic for CBOW word2vec model to be used with Pytorch Dataloader."""

    def __init__(
        self,
        vocab: vocab.Vocab,
        context_len: int = N_CONTEXT,
        max_sequence_len: int = MAX_SEQUENCE_LENGTH,
        subsample: bool = SUBSAMPLE,
        t: float = SUBSAMPLE_THRESHOLD,
    ):
        self.vocab = vocab
        self.context_len = context_len
        self.max_sequence_len = max_sequence_len
        self.subsample = subsample
        self.t = t

        if self.subsample:
            self.word2sswt = calc_subsample_weights(self.vocab, t=self.t)

    def __call__(self, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Make CBOW object callable."""
        return self.collate(batch)

    def collate(self, batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Collate function for CBOW word2vec model to be used with Pytorch Dataloader.

        Parameters
        ----------
        batch : list[dict[str, Any]]
            _description_
        """
        batch_input, batch_output = [], []
        for record in batch:
            ids_seq = record["ids"]

            if len(ids_seq) < self.context_len * 2 + 1:
                continue
            if self.max_sequence_len > 0:
                ids_seq = ids_seq[: self.max_sequence_len]

            # Subsampling Frequent Words
            # Probabilistically remove frequent words from sequence _prior to_ generating contexts (effectively increasing window size)
            if self.subsample:
                ids_seq = subsample_freq_words(record, word2sswt=self.word2sswt)

            # sliding window over list
            for context in sliding_window_contexts(ids_seq, n=self.context_len):
                center = context.pop(self.context_len)
                batch_input.append(context)
                batch_output.append(center)

        batch_input = torch.tensor(batch_input, dtype=torch.long)
        batch_output = torch.tensor(batch_output, dtype=torch.long)
        return batch_input, batch_output


# %%
# def collate_skipgram(
#     batch: list[dict[str, Any]],
#     context_len: int = N_CONTEXT,
#     subsample: bool = True,
#     neg_sample: bool = True,
# ):
#     """Collate function for skipgram word2vec model to be used with Pytorch Dataloader."""

#     if subsample:
#         word2subsample_wt = calc_subsample_weights(
#             vocab, t=10e-5
#         )  # TODO: pass as arg? don't calculate on each call to collate

#     sample_table = calc_sample_table(vocab, p=0.75)  # TODO: pass as arg? don't calculate on each call to collate

#     batch_input, batch_output = [], []
#     for record in batch:
#         ids_seq = record["ids"]

#         if len(ids_seq) < context_len * 2 + 1:
#             continue
#         if MAX_SEQUENCE_LENGTH:
#             ids_seq = ids_seq[:MAX_SEQUENCE_LENGTH]

#         # Subsampling Frequent Words
#         # Probabilistically remove frequent words from sequence _prior to_ generating contexts (effectively increasing window size)
#         if subsample:
#             ids_seq = dl.subsample_freq_words(record, word2sswt=word2subsample_wt)

#         # Dynamic window size:
#         # Since the more distant words are usually less related to the current word than those close to it,
#         # give less weight to the distant words by sampling less from those words in our training examples.
#         # For each training word, randomly select a number R in range <1; C>,
#         # and use R words from history and R words from the future of the current word as correct labels".
#         n = random.randint(1, context_len)
#         for idx in range(len(ids_seq) - n * 2):
#             context = ids_seq[idx : (idx + context_len * 2 + 1)]
#             center = context.pop(context_len)  # pop central token out of context window

#             positive_samples = [[tok, 1] for tok in context]

#             # Negative Sampling
#             negative_samples = [[vocab[ns], 0] for ns in random.sample(sample_table, k=5)]  # TODO:k as arg
#             # TODO: ensure that negative_samples is always len k, does not contain duplicates, and does not contain words from positive_samples

#             batch_input.append(center)
#             batch_output.append(positive_samples + negative_samples)
#             assert set([len(x) for x in batch_output]) == 1

#     batch_input = torch.tensor(batch_input, dtype=torch.long)
#     batch_output = torch.tensor(batch_output, dtype=torch.long)
#     return batch_input, batch_output
