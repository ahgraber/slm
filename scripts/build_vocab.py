# %%
import argparse
from collections import Counter, OrderedDict
from functools import partial
import itertools
import logging
import math
import os
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

import nltk
import spacy

import datasets
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer

from slm.preprocess import (  # NOQA: E402
    SentencePreTokenizer,
    blob_to_sentences,
    pipeline,
    sentences_to_words,
    wiki_remove_headings,
    wiki_truncate_appedices,
)
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

os.environ["HF_DATASETS_OFFLINE"] = 1  # use cached only

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, type=str)
    parser.add_argument("-n", "--name", required=False, type=str, default=None)
    parser.add_argument("-s", "--split", required=False, type=str, default="train")
    parser.add_argument("-k", "--key", required=False, type=str, default="text")

    args = parser.parse_args()

    if args.dataset.__contains__("wiki"):
        logging.debug("Detected wiki dataset")

    ds_kwargs = {
        # "download_mode": "force_redownload",  # if corrupted
        "download_mode": "reuse_cache_if_exists",  # for testing
        # "download_mode": "reuse_dataset_if_exists",  # for prod
        "verification_mode": "basic_checks",
        # "verification_mode": "all_checks",
    }

    # TODO:
    # - use datasets.load_dataset_builder(dataset, name) to load data file
    # //
    # try to laod from local first
    # otherwise download

    ds = datasets.load_dataset(
        name=args.name,
        split=args.split,
        streaming=True,
        **ds_kwargs,
    )
    # except datasets.data_files.EmptyDatasetError:
    #     ds = datasets.load_dataset(
    #         args.dataset,
    #         name=args.name,
    #         split=args.split,
    #         cache_dir=args.cache_dir,
    #         num_proc=8,
    #         **ds_kwargs,
    #     )
    #     ds = ds.to_iterable_dataset()

    ds_len = datasets.load_dataset_builder(args.dataset, args.name).info.splits[args.split].num_examples
    logger.info(f"{args.dataset}'s '{args.split}' split has {ds_len} examples")

    ds = ds.select_columns(args.key)

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

    # tokenizer.processor = processors.TemplateProcessing(  # add separators
    #     single="<SEP> $0 <SEP>",
    #     # pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #     special_tokens=[("<SEP>", 1)],  # ensure that tokenizer.vocab()["<SEP>"] == 1
    # )

    preprocess = partial(
        pipeline,
        key=args.key,
        fn=blob_to_sentences,
        kwargs={
            "is_wiki": args.dataset.__contains__("wiki"),
            "normalizer": prenormalizer,
            "splitter": sentence_splitter,
        },
    )
    tokenize = partial(
        pipeline,
        key=args.key,
        fn=sentences_to_words,
        kwargs={
            "normalizer": tokenizer.normalizer,
            "splitter": tokenizer.pre_tokenizer,
        },
    )

    # if wiki is iterabledataset, preprocessor executes only as batch is called
    batch_size = 64
    ds = ds.map(preprocess, input_columns=args.key, batched=True, batch_size=batch_size)
    ds = ds.map(tokenize, input_columns=args.key, batched=True, batch_size=batch_size)

    logging.info("Begin processing dataset & counting...")
    iterds = iter(ds)
    counter = Counter()
    for blob in tqdm(iterds, total=ds_len):
        counter.update(flatten(blob[args.key]))
    else:
        logging.info("Counting complete")

    logging.info("Creating vocab...")
    vocab = Vocab(counter)
    logging.info("Saving vocab...")
    with (DATA_DIR / f"{args.dataset}_vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)

    logging.info("Process complete.")
