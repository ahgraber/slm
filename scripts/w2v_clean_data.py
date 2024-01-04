"""Predefined preprocessing for a selection of huggingface datasets."""

# %%
import argparse
from collections import Counter
from functools import partial
import logging
from pathlib import Path
import pickle
import re
import sys
from typing import Optional, Union

from tqdm import tqdm

from nltk.tokenize import TreebankWordDetokenizer

import datasets

sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.data import constants  # NOQA: E402
from slm.data.preprocess import (  # NOQA: E402
    clean_wiki_articles,
    normalizer,
    parse_sentences,
    parse_words,
    prenormalizer,
    prep_data,
    treebank_detokenize,
)
from slm.utils import flatten, get_project_root  # NOQA: E402

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
DATA_DIR = ROOT_DIR / "data"

# %%
MANAGED_DATASETS = ["bookcorpus", "commoncrawl", "wikipedia"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, choices=constants.MANAGED_DATASETS.keys())
    parser.add_argument("-n", "--name", required=False, type=str, default=None)
    parser.add_argument("-s", "--split", required=False, type=str, default="train")
    parser.add_argument("-k", "--key", required=False, type=str, default="text")
    # parser.add_argument("-o", "--save_dir", required=False, type=Path, default=None)

    args = parser.parse_args()
    dataset = args.dataset
    path = constants.MANAGED_DATASETS[dataset]["path"]
    name = args.name if args.name else constants.MANAGED_DATASETS[dataset]["name"]
    split = args.split
    key = args.key

    match dataset:
        case "bookcorpus":
            map_fns = [
                treebank_detokenize,  # undo existing pretokenization
                normalizer.normalize_str,  # standard text normalization
            ]

        case "commoncrawl":
            map_fns = [
                parse_sentences,  # standard text normalization & split into sentences
            ]

        case "wikipedia":
            map_fns = [
                prenormalizer.normalize_str,  # pre-cealn
                clean_wiki_articles,  # clean wikipedia articles
                parse_sentences,  # standard text normalization & split into sentences
            ]

    # replicate default save logic to check if files exist
    save_prefix = f"{path}/{name}" if name else path
    save_dir = DATA_DIR / save_prefix
    save_dir.mkdir(parents=True, exist_ok=True)

    prep_data(
        path,
        name=name,
        map_fns=map_fns,
        save_dir=save_dir,
    )

    logger.info("Process complete.")
