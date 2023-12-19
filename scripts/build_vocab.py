# %%
import argparse
from collections import Counter
from functools import partial
import logging
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any, Callable, Iterable, Optional, Union

from tqdm import tqdm

import numpy as np

import datasets
from tokenizers import Regex, Tokenizer, models, normalizers, pre_tokenizers
import torch

sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.preprocess import (  # NOQA: E402
    SentencePreTokenizer,
    blob_to_sentences,
    pipeline,
    sentences_to_words,
    wiki_remove_headings,
    wiki_truncate_appedices,
)
from slm.utils import flatten, get_project_root, init_nltk, init_spacy  # NOQA: E402
from slm.word2vec.vocab import Vocab  # NOQA: E402

# %%
logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"  # noqa: N806
logging.basicConfig(format=LOG_FMT)
logging.captureWarnings(True)

logger.setLevel(logging.INFO)

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

# %%
init_nltk(model="punkt", savedir=DATA_DIR / "nltk")
init_spacy(model="en_core_web_sm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, type=str)
    parser.add_argument("-n", "--name", required=False, type=str, default=None)
    parser.add_argument("-s", "--split", required=False, type=str, default="train")
    parser.add_argument("-k", "--key", required=False, type=str, default="text")
    parser.add_argument("-p", "--save_path", required=False, type=Path, default=None)

    args = parser.parse_args()
    dataset = args.dataset
    name = args.name
    split = args.split
    key = args.key

    save_prefix = f"{dataset}_{name}" if name else f"{dataset}"
    save_path = args.save_path if args.save_path is not None else Path(ARTIFACT_DIR / f"{save_prefix}_vocab.pkl")

    ds_kwargs = {
        # "download_mode": "force_redownload",  # if corrupted
        "download_mode": "reuse_cache_if_exists",
        # "download_mode": "reuse_dataset_if_exists",
        "verification_mode": "basic_checks",
        # "verification_mode": "all_checks",
    }

    ds_len = datasets.load_dataset_builder(dataset, name).info.splits[split].num_examples
    logger.info(f"{dataset}'s '{split}' split has {ds_len} examples")

    ds = datasets.load_dataset(
        dataset,
        name=name,
        split=split,
        num_proc=8,
        **ds_kwargs,
    )
    ds = ds.to_iterable_dataset()
    ds = ds.select_columns(key)

    # normalize control chars
    prenormalizer = normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), "\n")
    # use nltk punkt sentence splitter
    sentence_splitter = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer())

    # NOTE: normalizer and pretokenizer must be same as those in Tokenizer used for modeling
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), ""),  # normalize control chars
            normalizers.Replace(Regex(r"[\s]"), " "),  # normalize whitespace
        ]
    )
    word_splitter = pre_tokenizers.Whitespace()

    preprocess = partial(
        pipeline,
        key=key,
        fn=blob_to_sentences,
        kwargs={
            "is_wiki": dataset.__contains__("wikipedia"),
            "normalizer": prenormalizer,
            "splitter": sentence_splitter,
        },
    )
    tokenize = partial(
        pipeline,
        key=key,
        fn=sentences_to_words,
        kwargs={
            "normalizer": normalizer,
            "splitter": word_splitter,
        },
    )

    # if wiki is iterabledataset, preprocessor executes only as batch is called
    batch_size = 64
    ds = ds.map(preprocess, input_columns=key, batched=True, batch_size=batch_size)
    ds = ds.map(tokenize, input_columns=key, batched=True, batch_size=batch_size)

    logger.info("Begin processing dataset & counting...")
    iterds = iter(ds)
    counter = Counter()
    for blob in tqdm(iterds, total=ds_len):
        counter.update(flatten(blob[key]))
    else:
        logger.info("Counting complete")

    logger.info("Creating vocab...")
    vocab = Vocab(counter)
    logger.info("Saving vocab...")
    with save_path.open("wb") as f:
        pickle.dump(vocab, f)

    logger.info("Process complete.")
