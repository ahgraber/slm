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
from slm.data import (  # NOQA: E402
    DATASETS,
    LOADER_KWARGS,
    MAP_KWARGS,
    bookcorpus,
    commoncrawl,
    wikipedia,
)
from slm.utils import flatten, get_project_root  # NOQA: E402
from slm.word2vec.vocab import Vocab  # NOQA: E402

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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
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

    shards = Path(save_dir).glob("shard_*")
    if any(shards):
        print("File(s) exist in save dir.  Continuing will overwrite existing file(s).")
        response = str(input("Continue? [y/N]  "))
        if response.lower() not in ["y", "yes"]:
            print("Exiting at user request.")
            sys.exit(0)

    loader_kwargs = LOADER_KWARGS
    loader_kwargs["download_mode"] = "reuse_cache_if_exists"  # use existing raw download, but not any prior work

    map_kwargs = MAP_KWARGS
    # map_kwargs["input_columns"] = key

    match dataset:
        case "bookcorpus":
            dset = bookcorpus(
                split=split,
                key=key,
                loader_kwargs=loader_kwargs,
                map_kwargs=map_kwargs,
            )

        case "commoncrawl":
            dset = commoncrawl(
                name=name if name else "realnewslike",
                split=split,
                key=key,
                loader_kwargs=loader_kwargs,
                map_kwargs=map_kwargs,
            )

        case "wikipedia":
            dset = wikipedia(
                name=name if name else "20231101.en",
                split=split,
                key=key,
                loader_kwargs=loader_kwargs,
                map_kwargs=map_kwargs,
            )

    logger.info("Process complete.")
