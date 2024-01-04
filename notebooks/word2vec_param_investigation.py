# %%
from collections import Counter
import logging
import os
from pathlib import Path
import pickle
import random
import sys

from tqdm import tqdm

import numpy as np
import pandas as pd

import datasets

import matplotlib.pyplot as plt

# from torchtext.vocab import Vocab, build_vocab_from_iterator
# may have to include `.env` file at project root containing `PYTHONPATH="./../src"`
sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.data import constants  # NOQA: E402
from slm.data.preprocess import (  # NOQA: E402
    load_data,
    tokenize,
    tokenize_batch,
    w2v_tokenizer,
)
from slm.utils import get_project_root  # NOQA: E402
from slm.word2vec.vocab import END_TOKEN, START_TOKEN, UNK_TOKEN, Vocab  # NOQA: E402

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

# %% [markdown]
# # Word2vec Parameters
#
# How do we know what the optimal parameters are for Word2Vec?
#
# Specifically, what thresholds should we use to configure Vocabulary limits (vocab size, minimum word frequency)?
#
# What do we know about our sentence lengths -
# what is a good compromise between including as many samples as possible (including short sentences),
# and including a wide context for CBOW or Skipgram windows?


# %%
# load and combine vocabs
vocab = Vocab(size=40_000, min_freq=5, unk_token=UNK_TOKEN, special_tokens=[START_TOKEN, END_TOKEN])
for pkl in (ARTIFACT_DIR / "vocab").glob("*_vocab.pkl"):
    logger.debug(f"Loading {pkl}")
    with pkl.open("rb") as f:
        _v = pickle.load(f)
    vocab.update(_v.counter)

logger.info("Combined vocab ")


# %% [markdown]
# ## Vocabulary Stats

# %%
# vocab frequencies
words, freqs = zip(*vocab.counter.most_common())

# sample
for i in [40, 60, 80, 100]:
    i = i * 1000
    print(f"{words[i]}: {freqs[i]}")

# kph: 7349
# schick: 3463
# porthole: 2032
# wolski: 1344

# describe
pd.options.display.float_format = "{:_.2f}".format
print("Word Frequency stats:")
print(pd.Series(freqs).describe())
# NOTE - words appearing only once account for >50% of total vocab
pd.reset_option("^display")

# Word Frequency stats:
# count    13_500_402.00
# mean            878.38
# std         325_098.68
# min               1.00
# 25%               1.00
# 50%               1.00
# 75%               4.00
# max     587_685_970.00


# %%
# plot vocab frequencies
fig, ax = plt.subplots()
pd.Series(freqs).plot(
    figsize=(12, 12),
    ax=ax,
    logy=True,
    title="Word Frequencies",
    xlabel="Word Rank (lower is more frequent)",
    ylabel="(log) Word Frequency (<word> counted Y times)",
)
for i in [20, 40, 60, 80, 100, 200, 400, 800]:
    i = i * 1000

    ax.annotate(
        f"rank {i//1000:_}k: {words[i]}",
        xy=(i, freqs[i]),
        xycoords="data",
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops=dict(  # NOQA: C408
            arrowstyle="-",
            connectionstyle="arc3,rad=0",
            relpos=(0, 0.5),  # start on rhs of text
        ),
        # family="sans-serif",
        fontsize=12,
    )

plt.savefig(ROOT_DIR / "docs" / "img" / "Vocab - Word Frequencies", bbox_inches="tight")
plt.show()  # must follow savefig

# %%
# plot 2nd order frequencies
word_frequency_counts = Counter(freqs)
word_freqs, freq_counts = zip(*word_frequency_counts.most_common())

# describe
pd.options.display.float_format = "{:_.2f}".format
print("Word Count and Count Frequency stats:")
print(
    pd.DataFrame(
        word_frequency_counts.most_common(),
        columns=["word_freqs", "freq_counts"],
    ).describe()
)
# NOTE - words appearing only once account for >50% of total vocab
pd.reset_option("^display")

# Word Count and Count Frequency stats:
#           word_freqs  freq_counts
# count      36_790.00    36_790.00
# mean      308_158.75       366.96
# std     6_220_107.29    38_742.84
# min             1.00         1.00
# 25%         9_438.25         1.00
# 50%        24_987.50         1.00
# 75%        81_850.75         2.00
# max   587_685_970.00 7_100_663.00

# %%
# plot
fig, ax = plt.subplots()
(
    pd.DataFrame(word_frequency_counts.most_common(), columns=["word_freqs", "freq_counts"])
    .sort_values("word_freqs")
    .plot(
        x="freq_counts",
        y="word_freqs",
        kind="scatter",
        alpha=0.3,
        s=10,
        figsize=(12, 12),
        ax=ax,
        # logx=True,
        logy=True,
        title="Word Frequency Counts",
        xlabel="Number Words @ Frequency (Y words with frequency X)",
        ylabel="(log) Word Frequency (Word counted X times)",
    )
)
ax.annotate(
    "Most words occur only once",
    xy=(1, 10),
    xycoords="data",
    xytext=(0.5 * 10e5, 1),
    textcoords="data",
    arrowprops=dict(  # NOQA: C408
        arrowstyle="-[",
        connectionstyle="arc3,rad=-.35",
        relpos=(0.0, 0.5),  # start on lhs of text
    ),
    # family="sans-serif",
    fontsize=12,
)
ax.annotate(
    "The most frequent words are extreme outliers",
    # xy=(0.06, 0.5),
    # textcoords='axes fraction',
    xy=(7 * 10e5, 1),
    xycoords="data",
    xytext=(3 * 10e5, 1),
    textcoords="data",
    arrowprops=dict(  # NOQA: C408
        arrowstyle="-[",
        connectionstyle="arc3,rad=0",
        relpos=(1.0, 0.5),  # start on rhs of text
    ),
    # family="sans-serif",
    fontsize=12,
)

plt.savefig(ROOT_DIR / "docs" / "img" / "Vocab - Word Frequency Counts", bbox_inches="tight")
plt.show()  # must follow savefig

# %%
pd.options.display.float_format = "{:_.2f}".format
print("% Occurrence of most frequent word-frequencies")
print(
    pd.DataFrame(word_frequency_counts.most_common(10), columns=["word_freqs", "freq_counts"])
    .sort_values("word_freqs")
    .set_index("word_freqs")
    / word_frequency_counts.total()
)
# NOTE - words appearing only once account for >50% of total vocab
pd.reset_option("^display")

# % Occurrence of most frequent word-frequencies
#             freq_counts
# word_freqs
# 1                  0.53
# 2                  0.14
# 3                  0.06
# 4                  0.04
# 5                  0.03
# 6                  0.02
# 7                  0.02
# 8                  0.01
# 9                  0.01
# 10                 0.01

# %% [markdown]
# ### Analysis
#
# The vast majority of words appear only once!
# If we require at least 4 occurrences, we are left with only the most-frequently occurring 25% of words seen.  This is still 3.375M words!
#
# If we restrict to 40k most-common, the least-frequent word ('kph') has 7349 occurrences.
# If we restrict to 60k most-common, the least-frequent word ('schick') has 3463 occurrences.
# If we restrict to 80k most-common, the least-frequent word ('porthole') has 2032 occurrences.
# If we restrict to 100k most-common, the least-frequent word ('wolski') has 1344 occurrences.
#
# > _Side note:_
# > Given XKCD's success explaining things with the 1_000 most common words ([thing-explainer](https://xkcd.com/thing-explainer), [Up Goer Five](https://xkcd.com/1133/)),
# > we probably don't need to expand the vocabulary as much as we might expect.

# %%
# load preprocessed datasets
split = "train"
key = "text"
map_kwargs = constants.MAP_KWARGS
map_kwargs["input_columns"] = key
# map_kwargs["batch_size"] = 64

dsets = []
dsamples = []
for dataset in ["bookcorpus", "commoncrawl"]:
    ds = load_data(dataset)

    nrows = ds.num_rows
    logger.info(f"{dataset} has {nrows} records")
    dsamples.append(nrows)

    dsets.append(ds.to_iterable_dataset())


# %%
# proportionally interleave datasets
ds = datasets.interleave_datasets(
    dsets,
    probabilities=[i / sum(dsamples) for i in dsamples],  # sample according to relative sizes
    seed=SEED,
    stopping_strategy="all_exhausted",
)
ds = ds.select_columns(key)

# %%
tokenizer = w2v_tokenizer(vocab=vocab)
ds = ds.map(lambda batch: tokenize_batch(batch, tokenizer=tokenizer), **map_kwargs)
# NOTE: approx 9.5 hrs to traverse iterabledataset

# NOTE: if map dataset, add num_proc>1 for multiprocessing
# ds.set_transform(lambda batch: tokenize_batch(batch[key]), columns=[key])
# # ds[0]

# %%
# approx 9.5 hrs
n_tokens = [len(record["ids"]) for record in tqdm(iter(ds), total=sum(dsamples))]

# %% [markdown]
# ## Sequence Stats

# %%
pd.options.display.float_format = "{:_.2f}".format
print(pd.Series(n_tokens).describe())
pd.reset_option("^display")

# count   374_610_744
# mean             24.23
# std              14.60
# min               2
# 25%              14
# 50%              22
# 75%              32

# %%
sequence_freqs = pd.Series(Counter(n_tokens)).sort_index()

pd.options.display.float_format = "{:_.2f}".format
print(sequence_freqs.describe())
pd.reset_option("^display")

# count        1_349
# mean       277_695.14
# std      1_548_104.81
# min              1
# 25%              1
# 50%              5
# 75%             58
# max     12_162_832


# %%
fig, ax = plt.subplots(figsize=(12, 12))
sequence_freqs.sort_index().plot(
    ax=ax,
    logx=True,
    title="Sequence Length Frequencies",
    xlabel="(log) Sequence Length Rank (lower is more frequent)",
    ylabel="Sequence Length Frequency (<word> counted Y times)",
)
for i in [5, 10, 15, 20, 40, 80, 100]:
    ax.annotate(
        f"{i} tokens: {sequence_freqs[i] / sum(sequence_freqs):.2f}%",
        xy=(i, sequence_freqs[i]),
        xycoords="data",
        xytext=(10, 10),
        textcoords="offset points",
        arrowprops=dict(  # NOQA: C408
            arrowstyle="-",
            connectionstyle="arc3,rad=0",
            relpos=(0, 0.5),  # start on rhs of text
        ),
        # family="sans-serif",
        fontsize=12,
    )

plt.savefig(ROOT_DIR / "docs" / "img" / "Sentence Lengths (n tokens).png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Analysis
#
# We think about word2vec context as length on each side of the central word.
# A word2vec context length of 3 means _,_,_, keyword, _,_,_ --> minimum length of 7 (2n + 1).
#
# If we set a minimum context of 5, we require a minimum sequence length of 11 tokens.  11 tokens is ~15th percentile, so we retain ~85% of our samples.

# %%
