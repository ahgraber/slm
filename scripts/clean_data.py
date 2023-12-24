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

slm_logger = logging.getLogger("slm")

logger.setLevel(logging.INFO)
slm_logger.setLevel(logging.INFO)

# %%
ROOT_DIR = get_project_root()
DATA_DIR = ROOT_DIR / "data"

# %%
managed_datasets = ["bookcorpus", "commoncrawl", "wikipedia"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, choices=managed_datasets)
    parser.add_argument("-n", "--name", required=False, type=str, default=None)
    parser.add_argument("-s", "--split", required=False, type=str, default="train")
    parser.add_argument("-k", "--key", required=False, type=str, default="text")
    # parser.add_argument("-p", "--save_dir", required=False, type=Path, default=None)

    args = parser.parse_args()
    dataset = args.dataset
    name = args.name
    split = args.split
    key = args.key

    # replicate default save logic to check if files exist
    save_prefix = f"{dataset}_{name}" if name else f"{dataset}"
    save_dir = Path(DATA_DIR / f"{save_prefix}")

    match dataset:
        case "bookcorpus":
            map_fns = [
                treebank_detokenize,  # undo existing pretokenization
                normalizer.normalize_str,  # standard text normalization
            ]
            prep_data(
                "bookcorpus",
                map_fns=map_fns,
                save_dir=save_dir,
            )

        case "commoncrawl":
            map_fns = [
                parse_sentences,  # standard text normalization & split into sentences
            ]
            prep_data(
                "c4",
                name=name if name else "realnewslike",
                map_fns=map_fns,
                save_dir=save_dir,
            )

        case "wikipedia":
            map_fns = [
                prenormalizer.normalize_str,  # pre-cealn
                clean_wiki_articles,  # clean wikipedia articles
                parse_sentences,  # standard text normalization & split into sentences
            ]
            prep_data(
                "wikimedia/wikipedia",
                name=name if name else "20231101.en",
                map_fns=map_fns,
                save_dir=save_dir,
            )

    logger.info("Process complete.")
