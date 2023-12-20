# %%
from functools import partial
import logging
from pathlib import Path
import pickle
import random
import re
import string
import sys
from typing import Any, Callable, Iterable, Optional, Union
import unicodedata

from tqdm import tqdm

import numpy as np

from nltk.tokenize import TreebankWordDetokenizer
import spacy

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer

# from torchtext.vocab import Vocab, build_vocab_from_iterator
# may have to include `.env` file at project root containing `PYTHONPATH="./../src"`
# sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.preprocess import (  # NOQA: E402
    SentencePreTokenizer,
    batch_map,
    normalizer,
    parse_sentences,
    parse_words,
    prenormalizer,
    sentence_splitter,
    word_splitter,
)
from slm.utils import flatten, get_project_root  # NOQA: E402
from slm.word2vec.vocab import SEP_TOKEN, UNK_TOKEN, Vocab  # NOQA: E402

# %%
logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"  # noqa: N806
logging.basicConfig(format=LOG_FMT)
logging.captureWarnings(True)

logger.setLevel(logging.INFO)

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
# ## Preprocess
#
# For word2vec, preprocessing (moving from raw text to tensors ready for model input involves:
#
# - normalization
# - [optional] adding separator tokens at sentence boundaries
# - pre-token splitting
# - vocabulary creation
# - word-to-token conversion
#
# Ideally, we will create a repeatable preprocessing pipeline that uses generators to avoid
# loading or expanding the entire dataset in-memory
#
# Ref: https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt#building-a-tokenizer-block-by-block

# %%
DATASETS = [
    "bookcorpus",
    "commoncrawl",  # "realnewslike"
    "wikipedia",  # "20231101.en"
    # ("wikimedia/wikisource", # "20231201.en"
]

LOADER_KWARGS = {
    "download_mode": "reuse_cache_if_exists",  # ["force_redownload", "reuse_cache_if_exists", "reuse_dataset_if_exists"]
    "verification_mode": "basic_checks",  # ["all_checks", "basic_checks"]
    "num_proc": 8,  # load with num_proc, then convert to iterable_dataset
}
SPLIT = "train"
KEY = "text"
BATCH_SIZE = 64
MAP_KWARGS = {
    "input_columns": KEY,
    "batched": True,
    "batch_size": BATCH_SIZE,
}
BUFFER_SIZE = 1_000


# %%
def bookcorpus(
    name: str = None,
    split: str = "train",
    key: str = "text",
    loader_kwargs: Optional[dict] = LOADER_KWARGS,
    map_kwargs: Optional[dict] = MAP_KWARGS,
    iterable: bool = True,
):
    """Prepare bookcorpus dataset with predefined preprocessing steps.

    NOTE: Does not include final normalization or tokenization.
    """
    path = "bookcorpus"
    ds = datasets.load_dataset(path, split=split, **loader_kwargs)
    ds = ds.select_columns(key)
    n_samples = datasets.load_dataset_builder(path, name).info.splits[split].num_examples

    if iterable:
        ds = ds.to_iterable_dataset()

    # bookcorpus appears to have been tokenized by NLTK's TreebankWordTokenizer
    # https://github.com/huggingface/datasets/issues/486
    twd = TreebankWordDetokenizer()

    def treebank_detokenize(blob: str, twd: TreebankWordDetokenizer = twd) -> str:
        return twd.detokenize(blob.split())

    ds = ds.map(partial(batch_map, key=key, fn=treebank_detokenize), **map_kwargs)

    # standard pre-normalizaition
    ds = ds.map(partial(batch_map, key=key, fn=prenormalizer.normalize_str), **map_kwargs)
    # standard sentence splitter
    ds = ds.map(partial(batch_map, key=key, fn=parse_sentences), **map_kwargs)

    return ds, n_samples


def commoncrawl(
    name: str = "realnewslike",
    split: str = "train",
    key: str = "text",
    loader_kwargs: Optional[dict] = LOADER_KWARGS,
    map_kwargs: Optional[dict] = MAP_KWARGS,
    iterable: bool = True,
):
    """Prepare C4 dataset with predefined preprocessing steps.

    NOTE: Does not include final normalization or tokenization.
    """
    path = "c4"
    ds = datasets.load_dataset(path, name, split=split, **loader_kwargs)
    ds = ds.select_columns(key)
    n_samples = datasets.load_dataset_builder(path, name).info.splits[split].num_examples

    if iterable:
        ds = ds.to_iterable_dataset()

    # standard pre-normalizaition
    ds = ds.map(partial(batch_map, key=key, fn=prenormalizer.normalize_str), **map_kwargs)
    # standard sentence splitter
    ds = ds.map(partial(batch_map, key=key, fn=parse_sentences), **map_kwargs)

    return ds, n_samples


def wikipedia(
    name: str = "20231101.en",
    split: str = "train",
    key: str = "text",
    loader_kwargs: Optional[dict] = LOADER_KWARGS,
    map_kwargs: Optional[dict] = MAP_KWARGS,
    iterable: bool = True,
):
    """Prepare wikimedia/wikipedia dataset with predefined preprocessing steps.

    NOTE: Does not include final normalization or tokenization.
    """
    path = "wikimedia/wikipedia"
    ds = datasets.load_dataset(path, name, split=split, **loader_kwargs)
    ds = ds.select_columns(key)
    n_samples = datasets.load_dataset_builder(path, name).info.splits[split].num_examples

    if iterable:
        ds = ds.to_iterable_dataset()
    # standard pre-normalizaition
    ds = ds.map(partial(batch_map, key=key, fn=prenormalizer.normalize_str), **map_kwargs)

    # wikipedia specific cleaning
    def clean_wiki_articles(article: str) -> str:
        """Remove standard wikipedia appendices from article blob, and remove headings."""
        # we know wikipedia layout schema; we can remove any 'see also' links and references
        # https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Layout#Standard_appendices_and_footers
        for heading in [
            "Gallery",
            "See [Aa]lso",
            "Notes and [Rr]eferences",
            "Notes",
            "Endnotes",
            "Foot[Nn]otes",
            "Works [Cc]ited",
            "References",
            "Sources",
            "Citations",
            "Bibliography",
            "Further [Rr]eading",
            "External [Ll]inks",
        ]:
            match = re.search(heading + r"\s\n", article)
            if match:
                article = article[: match.start()]

        # Non-appendix headings defined as spans of 1-5 words between newlines without punctuation.
        heading = re.compile(
            r"""[\n\r]{1}            # match single newline/carriage return
                \ *(\w+\ ?){1,5}\ *  # match 1-5 words, with optional preceeding/succeeding space
                [\n\r]{1}            # match single newline/carriage return
            """,
            re.X,
        )
        return heading.sub("\n\n", article)

    ds = ds.map(partial(batch_map, key=key, fn=clean_wiki_articles), **map_kwargs)

    # standard sentence splitter
    ds = ds.map(partial(batch_map, key=key, fn=parse_sentences), **map_kwargs)

    return ds, n_samples
