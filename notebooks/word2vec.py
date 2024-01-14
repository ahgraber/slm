# %%
import collections
import functools
import itertools
import json
import logging
import math
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any, Callable, Iterable, Literal, Optional, Union

import tqdm

import numpy as np
import pandas as pd

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

# may have to include `.env` file at project root containing `PYTHONPATH="./../src"`
sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.data import constants, preprocess  # NOQA: E402
from slm.utils import get_project_root, torch_device  # NOQA: E402
from slm.word2vec import loaders, models, trainer, vocab  # NOQA: E402

# %%
logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"  # noqa: N806
logging.basicConfig(format=LOG_FMT)
logging.captureWarnings(True)
logger.setLevel(logging.INFO)

slm_logger = logging.getLogger("slm")
slm_logger.setLevel(logging.INFO)

# %%
ROOT_DIR = get_project_root()
ARTIFACT_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

# %%
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
rnd = np.random.default_rng(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)


# %% [markdown]
# # Word2vec
#
# ## Preprocessing
#
# Run `w2v_clean_data` and `w2v_build_vocab` scripts first!
# These clean the data, and build a vocabulary we can load below.
#
# ```sh
# # preprocess - normalize and sentence-split
# python scripts/w2v_clean_data.py -d bookcorpus
# python scripts/w2v_clean_data.py -d commoncrawl
# python scripts/w2v_clean_data.py -d wikipedia
#
# # build vocab files
# python scripts/w2v_build_vocab.py -d bookcorpus
# python scripts/w2v_build_vocab.py -d commoncrawl
# python scripts/w2v_build_vocab.py -d wikipedia
# ```

# %%
# load and combine vocabs
vocabulary = vocab.Vocab(
    size=60_000, min_freq=5, unk_token=vocab.UNK_TOKEN, special_tokens=[vocab.START_TOKEN, vocab.END_TOKEN]
)
for pkl in (ARTIFACT_DIR / "vocab").glob("*_vocab.pkl"):
    logger.debug(f"Loading {pkl}")
    with pkl.open("rb") as f:
        _v = pickle.load(f)
    vocabulary.update(_v.counter)

logger.info("Combined vocab ")


# %%
# load preprocessed datasets
split = "train"
key = "text"
map_kwargs = constants.MAP_KWARGS
map_kwargs["input_columns"] = key
map_kwargs["batch_size"] = 128

trn_pct, val_ptc, tst_pct = 0.75, 0.15, 0.10
trn_dsets, val_dsets, tst_dsets = [], [], []
trn_sample, val_sample, tst_sample = [], [], []

for dataset in ["bookcorpus"]:  # , "commoncrawl"]:
    ds = preprocess.load_data(dataset)

    nrows = ds.num_rows
    logger.info(f"{dataset} has {nrows} records")

    split = ds.train_test_split(train_size=int(nrows * trn_pct), seed=SEED)
    trn_ds = split["train"]

    split = split["test"].train_test_split(test_size=int(nrows * tst_pct), seed=SEED)
    val_ds = split["train"]
    tst_ds = split["test"]

    trn_sample.append(trn_ds.num_rows)
    val_sample.append(val_ds.num_rows)
    tst_sample.append(tst_ds.num_rows)

    trn_dsets.append(trn_ds)
    val_dsets.append(val_ds)
    tst_dsets.append(tst_ds)


# %%
# proportionally interleave datasets
# NOTE: probably requires iterable_datasets
trn_ds = datasets.interleave_datasets(
    [ds.to_iterable_dataset() for ds in trn_dsets],
    probabilities=[i / sum(trn_sample) for i in trn_sample],  # sample according to relative sizes
    seed=SEED,
    stopping_strategy="all_exhausted",
)
val_ds = datasets.interleave_datasets(
    [ds.to_iterable_dataset() for ds in val_dsets],
    probabilities=[i / sum(trn_sample) for i in trn_sample],  # sample according to relative sizes
    seed=SEED,
    stopping_strategy="all_exhausted",
)
tst_ds = datasets.interleave_datasets(
    [ds.to_iterable_dataset() for ds in tst_dsets],
    probabilities=[i / sum(trn_sample) for i in trn_sample],  # sample according to relative sizes
    seed=SEED,
    stopping_strategy="all_exhausted",
)
trn_ds = trn_ds.select_columns(key)
val_ds = val_ds.select_columns(key)
tst_ds = tst_ds.select_columns(key)

trn_sample = sum(trn_sample)
val_sample = sum(val_sample)
tst_sample = sum(tst_sample)

del trn_dsets, val_dsets, tst_dsets

# %%
tokenizer = preprocess.w2v_tokenizer(vocab=vocabulary)
trn_ds = trn_ds.map(lambda batch: preprocess.tokenize_batch(batch, tokenizer=tokenizer), **map_kwargs)
val_ds = val_ds.map(lambda batch: preprocess.tokenize_batch(batch, tokenizer=tokenizer), **map_kwargs)
tst_ds = tst_ds.map(lambda batch: preprocess.tokenize_batch(batch, tokenizer=tokenizer), **map_kwargs)
# NOTE: from w2v stats investigation, it takes 8-10 hrs to traverse iterabledataset

# NOTE: if map dataset, add num_proc>1 for multiprocessing
# ds.set_transform(lambda batch: tokenize_batch(batch[key]), columns=[key])
# # ds[0]

# each record now has attributes: text, ids, tokens, attention_mask

# %%
# investigate subsampling of frequent words
# loaders.plot_subsample_weights(vocabulary)


# %%
# NOTE:
# For the experiments reported ..., we used
# three training epochs with stochastic gradient descent and backpropagation.
# We chose starting learning rate 0.025 and decreased it linearly.

# %%
epochs = 50
model = models.CBOW_Model(vocab=vocabulary)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.025)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda epoch: (epochs - epoch) / epochs,
)
device = torch_device()

cbow_trainer = trainer.Trainer(
    model=model,
    batch_size=constants.BATCH_SIZE,
    epochs=50,
    collate_fn=loaders.CBOW(vocabulary),
    trn_dataset=trn_ds,
    trn_sample=trn_sample,
    val_dataset=val_ds,
    val_sample=val_sample,
    criterion=nn.CrossEntropyLoss(),
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    device=torch_device(),
    checkpoint_frequency=5,
    log_dir=ROOT_DIR / "runs",
    model_dir=ARTIFACT_DIR,
    model_name="CBOW",
)

# %%
cbow_trainer.train()
print("Training finished.")

cbow_trainer.save_model()
cbow_trainer.save_loss()

# %%

# %%


# %%
# def calc_sample_table(vocab: vocab.Vocab, p: float = 0.75) -> list[str]:
#     """Calculate negative sampling probability for all words in vocab and populate table with proportional membership."""

#     words, counts = zip(*vocab.counter.most_common(vocab.size))
#     denominator = sum(counts) ** p

#     table_size = vocab.size * 100  # approximate table length
#     table = [
#         [word] * int(prob * table_size)
#         for word, prob in zip(
#             words,
#             (np.array(counts) ** p) / denominator,
#         )
#     ]

#     return list(itertools.chain.from_iterable(table))


# def collate_skipgram(
#     vocab: vocab.Vocab,
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
#             ids_seq = loaders.subsample_freq_words(record, word2sswt=word2sswt)

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


# class SkipGram_Model(nn.Module):
#     """
#     Implementation of Skip-Gram model described in paper:
#     https://arxiv.org/abs/1301.3781
#     """

#     def __init__(self, vocab_size: int):
#         super(SkipGram_Model, self).__init__()
#         self.embeddings = nn.Embedding(
#             num_embeddings=vocab_size,
#             embedding_dim=EMBED_DIMENSION,
#             max_norm=EMBED_MAX_NORM,
#         )
#         self.linear = nn.Linear(
#             in_features=EMBED_DIMENSION,
#             out_features=vocab_size,
#         )

#     def forward(self, inputs_):
#         x = self.embeddings(inputs_)
#         x = self.linear(x)
#         return x


# %%
# see https://github.com/ddehueck/skip-gram-negative-sampling for example with tensorboard
# see https://github.com/lukysummer/SkipGram_with_NegativeSampling_Pytorch/blob/master/SkipGram_NegativeSampling.ipynb for example of using pytorch for negative sampling


# %%
# use PCA or Multiple Linear Dscriminant Analysis to perfectly align "gender" and "royalty" axes for king - man + woman = queen?
