# %%
from collections import Counter, OrderedDict
from functools import partial
import itertools
import logging
import math
import os
from pathlib import Path
import random
import re
import string
import sys
from typing import Any, Callable, Iterable, Optional, Union
import unicodedata

import nltk
import spacy
from tqdm import tqdm

import numpy as np

import datasets
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer

# from torchtext.vocab import Vocab, build_vocab_from_iterator
# may have to include `.env` file at project root containing `PYTHONPATH="./../src"`
sys.path.insert(0, str(Path(__file__ + "/../../src").resolve()))
from slm.preprocess import SentencePreTokenizer, pipeline  # NOQA: E402
from slm.utils import flatten, get_project_root, init_nltk, init_spacy  # NOQA: E402
from slm.word2vec.config import W2VConfig  # NOQA: E402
from slm.word2vec.vocab import Vocab  # NOQA: E402

# %%
logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"  # noqa: N806
logging.basicConfig(format=LOG_FMT)
logging.captureWarnings(True)

logger.setLevel(logging.INFO)

# %%
ROOT_DIR = get_project_root()
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

# %%
init_nltk(model="punkt", savedir=DATA_DIR / "nltk")
init_spacy(model="en_core_web_sm")

# %% [markdown]
# ## Source data
#
# Huggingface's `datasets` package provides easy access to predefined data.
#
# These are type `datasets.arrow_dataset.Dataset` (https://huggingface.co/docs/datasets/main/en/about_arrow)
# Loading these creates a memory map but does not _actually_ load files into memory
# We can also leverage dataset functionality like map() and filter() to apply batched transformations
# or even use an IterableDataset to lazily apply transformations as needed

# %%
# wiki = datasets.load_dataset(
#     "wikimedia/wikipedia",
#     name="20231101.en",
#     split="train[:256]",
#     download_mode="reuse_cache_if_exists",  # rebuild dataset obj every time
#     verification_mode="basic_checks",
#     cache_dir=DATA_DIR / "wikipedia",
#     num_proc=4,
# )

# %% [markdown]
# ### Sample (wikipedia)
#
# ```py
# {'id': '1',
#  'url': 'https://simple.wikipedia.org/wiki/April',
#  'title': 'April',
#  'text': 'April is the fourth month...'
# }
# ```
#
# ### Schema
#
# ```txt
# id (str): ID of the article.
# url (str): URL of the article.
# title (str): Title of the article.
# text (str): Text content of the article.
# ```

# %%
ds_kwargs = {
    "split": "train",
    # "download_mode": "reuse_cache_if_exists", # for testing
    "download_mode": "reuse_dataset_if_exists",  # for prod
    "verification_mode": "basic_checks",
    "streaming": True,
    # "num_proc":8,  # only if streaming == False
}
key = "text"

# %%
wiki = datasets.load_dataset(
    "wikimedia/wikipedia",
    name="20231101.en",
    cache_dir=DATA_DIR / "wikipedia",
    **ds_kwargs,
)
wiki_len = (
    datasets.load_dataset_builder("wikimedia/wikipedia", "20231101.en").info.splits[ds_kwargs["split"]].num_examples
)
# %%
# wiki = wiki.to_iterable_dataset()
wiki = wiki.select_columns(key)
# wiki = wiki.shuffle(SEED)
# if isinstance(wiki, datasets.Dataset):
#     wiki = wiki.flatten_indices()


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
# normalize control chars
prenormalizer = normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), "\n")
# use nltk punkt sentence splitter
sentence_splitter = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer())

# define tokenizer here for use both in building vocab and tokenizeing
tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), ""),  # normalize control chars
        normalizers.Replace(Regex(r"[\s]"), " "),  # normalize whitespace
    ]
)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # split on whitespace or non-word non-space character
tokenizer.processor = processors.TemplateProcessing(  # add separators
    single="<SEP> $0 <SEP>",
    # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("<SEP>", 1)],  # ensure that tokenizer.vocab()["<SEP>"] == 1
)


# %%
# Map preprocessing functions to dataset
# https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map
# https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.IterableDataset.map
# https://huggingface.co/docs/datasets/main/en/about_map_batch#map

# wrap preprocessing fn in partial so its only input is a list of text as a positional arg
preprocessor = partial(
    pipeline,
    key=key,
    b2s_kwargs={"is_wiki": True, "normalizer": prenormalizer, "splitter": sentence_splitter},
    s2w_kwargs={"normalizer": tokenizer.normalizer, "splitter": tokenizer.pre_tokenizer},
)

# if wiki is iterabledataset, preprocessor executes only as batch is called
batch_size = 32
wiki = wiki.map(preprocessor, input_columns=key, batched=True, batch_size=batch_size)

# %%
ds = iter(wiki)
counter = Counter()
for blob in tqdm(ds, total=math.ceil(wiki_len / batch_size)):
    counter.update(flatten(blob[key]))
else:
    print("Done")

# %%
counter


# %%
vocab = Vocab()
vocab.update(counter)


# %%
import pickle

with (DATA_DIR / "vocab").open("wb") as f:
    pickle.dump(f, vocab)

# %%


# %%


# %%

# %%
books = datasets.load_dataset(
    "bookcorpus",
    cache_dir=DATA_DIR / "bookcorpus",
    **ds_kwargs,
)
books_len = datasets.load_dataset_builder("bookcorpus").info.splits[ds_kwargs["split"]].num_examples

# %%
c4 = datasets.load_dataset(
    "c4",
    name="realnewslike",  # "en"
    cache_dir=DATA_DIR / "c4",
    **ds_kwargs,
)
c4_len = datasets.load_dataset_builder("c4", "realnewslike").info.splits[ds_kwargs["split"]].num_examples


# %%

# %%

# %% [markdown]
# ## Tokenize
#
# Ref: https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt#building-a-tokenizer-block-by-block

# %%
tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>"))
# use normalizer, pre_tokenizer defined above during Vocab pipeline
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer  # split on whitespace or non-word non-space character
tokenizer.processor = processors.TemplateProcessing(  # add separators
    single="<SEP> $0 <SEP>",
    # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("<SEP>", 1)],  # ensure that tokenizer.vocab()["<SEP>"] == 1
)

# %%
# trainer can build vocab
trainer = trainers.WordLevelTrainer(
    vocab_size=50_000,
    # min_frequency=3,
    show_progress=True,
    special_tokens=["<UNK>", "<SEP>"],
)


def batch_corpus_generator(dataset: str, key: str = "text", batchsize: int = 64):
    """Generate batches from corpus dataset."""
    for i in range(0, len(dataset), batchsize):
        yield dataset[i : i + batchsize][key]


tokenizer.train_from_iterator(
    batch_corpus_generator(dataset=wiki, key="text", batchsize=10), trainer=trainer, length=len(wiki)
)

# %%
print(tokenizer.get_vocab_size())
tokenizer.get_vocab()


# %%
# tokenize
output = tokenizer.encode(ARISTOTLE)

# %%
print(output.tokens)

# %%
print(output.ids)
# %%
print(tokenizer.decode(output.ids))


# %%
# TODO: interleave all datasets
from datasets import interleave_datasets

wiki = load_dataset(..., streaming=True)
books = load_dataset(..., streaming=True)
c4 = load_dataset(..., streaming=True)
ds = interleave_datasets([wiki, books, c4])
