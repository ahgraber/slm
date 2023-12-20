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

from tqdm import tqdm

import numpy as np

sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.datasets import DATASETS, LOADER_KWARGS, MAP_KWARGS, bookcorpus, commoncrawl, wikipedia  # NOQA: E402
from slm.preprocess import batch_map, parse_words  # NOQA: E402
from slm.utils import flatten, get_project_root  # NOQA: E402
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, choices=DATASETS)
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

    if save_path.exists():
        print("File exists at save path.  Continuing will overwrite existing file.")
        response = str(input("Continue? [y/N]  "))
        if response.lower() not in ["y", "yes"]:
            print("Exiting at user request.")
            SystemExit()

    loader_kwargs = LOADER_KWARGS
    loader_kwargs["download_mode"] = "reuse_cache_if_exists"

    map_kwargs = MAP_KWARGS
    map_kwargs["input_columns"] = key

    match dataset:
        case "bookcorpus":
            dset, dsamples = bookcorpus(
                split=split,
                key=key,
                loader_kwargs=loader_kwargs,
                map_kwargs=map_kwargs,
                iterable=True,
            )
        case "commoncrawl":
            dset, dsamples = commoncrawl(
                name=name if name else "realnewslike",
                split=split,
                key=key,
                loader_kwargs=loader_kwargs,
                map_kwargs=map_kwargs,
                iterable=True,
            )

        case "wikipedia":
            dset, dsamples = wikipedia(
                name=name if name else "20231101.en",
                split=split,
                key=key,
                loader_kwargs=loader_kwargs,
                map_kwargs=map_kwargs,
                iterable=True,
            )

    logger.info(f"{dataset}'s '{split}' split has {dsamples} records")

    dset = dset.map(partial(batch_map, key=key, fn=parse_words), **map_kwargs)

    logger.info("Begin processing dataset & counting...")
    iterds = iter(dset)
    counter = Counter()
    for blob in tqdm(iterds, total=dsamples):
        counter.update(flatten(blob[key]))
    else:
        logger.info("Counting complete")

    logger.info("Creating vocab...")
    vocab = Vocab(counter)
    logger.info("Saving vocab...")
    with save_path.open("wb") as f:
        pickle.dump(vocab, f)

    logger.info("Process complete.")
