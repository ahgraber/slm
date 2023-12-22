# %%
from functools import partial
import logging
from pathlib import Path
import re
from typing import Optional, Union

from nltk.tokenize import TreebankWordDetokenizer

import datasets

from slm.preprocess import (  # NOQA: E402
    batch_map,
    nltk_sentence_splitter,
    normalizer,
    parse_sentences,
    parse_words,
    prenormalizer,
    word_splitter,
)
from slm.utils import flatten, get_project_root  # NOQA: E402

# %%
logger = logging.getLogger(__name__)

# Ref: https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt#building-a-tokenizer-block-by-block

# %%
ROOT_DIR = get_project_root()
ARTIFACT_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

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
N_SHARDS = 32
BATCH_SIZE = 64
MAP_KWARGS = {
    "batched": True,
    "batch_size": BATCH_SIZE,
}
BUFFER_SIZE = 1_000


# %%
def gen_from_iterable_dataset(ds: datasets.IterableDataset):
    """Convert IterableDataset to generator.

    NOTE: Helper fn. for converting IterableDataset to Dataset.
    """
    yield from ds


# %%
def bookcorpus(
    name: str = None,
    split: str = "train",
    key: str = "text",
    loader_kwargs: Optional[dict] = LOADER_KWARGS,
    map_kwargs: Optional[dict] = MAP_KWARGS,
    # save_dir: Optional[Union[str, Path]] = None,
):
    """Prepare bookcorpus dataset with predefined preprocessing steps.

    NOTE: Does not include final normalization or tokenization.
    """
    path = "bookcorpus"

    save_prefix = f"{path}_{name}" if name else f"{path}"
    # save_dir = save_dir if save_dir is not None else Path(DATA_DIR / f"{save_prefix}")
    save_dir = Path(DATA_DIR / f"{save_prefix}")

    shards = Path(save_dir).glob("*.arrow")
    if any(shards):
        logger.info("Loading from saved work...")
        # ds = datasets.concatenate_datasets([datasets.load_from_disk(shard) for shard in shards])
        ds = datasets.load_from_disk(save_dir)
    else:
        # if no files found
        ds = datasets.load_dataset(path, name, split=split, **loader_kwargs)
        logger.info(f"Before processing, {ds}'s '{split}' split has {ds.num_rows} records")

        ds = ds.select_columns(key)
        ds = ds.to_iterable_dataset()

        # bookcorpus appears to have been tokenized by NLTK's TreebankWordTokenizer
        # https://github.com/huggingface/datasets/issues/486
        twd = TreebankWordDetokenizer()

        def treebank_detokenize(record: str, twd: TreebankWordDetokenizer = twd) -> str:
            return twd.detokenize(record.split())

        map_kwargs["input_columns"] = key
        # undo TreebankWordTokenizer
        ds = ds.map(partial(batch_map, key=key, fn=treebank_detokenize), **map_kwargs)

        # apply standard text normalization
        ds = ds.map(partial(batch_map, key=key, fn=normalizer.normalize_str), **map_kwargs)

        # convert iterable to standard dataset for saving
        logger.info("Processing...")
        ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
        logger.info(f"After processing, {ds}'s '{split}' split has {ds.num_rows} records")

        # save to reload later
        logger.info("Saving...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(Path(save_dir), num_shards=N_SHARDS, num_proc=8)
        # for shard_idx in range(N_SHARDS):
        #     shard = ds.shard(num_shards=N_SHARDS, index=shard_idx, contiguous=True)
        #     shard.save_to_disk(Path(save_dir) / f"shard_{N_SHARDS}")

    return ds


def commoncrawl(
    name: str = "realnewslike",
    split: str = "train",
    key: str = "text",
    loader_kwargs: Optional[dict] = LOADER_KWARGS,
    map_kwargs: Optional[dict] = MAP_KWARGS,
    # save_dir: Optional[Union[str, Path]] = None,
):
    """Prepare C4 dataset with predefined preprocessing steps.

    NOTE: Does not include final normalization or tokenization.
    """
    path = "c4"

    save_prefix = f"{path}_{name}" if name else f"{path}"
    save_dir = Path(DATA_DIR / f"{save_prefix}")

    shards = Path(save_dir).glob("shard_*")
    if any(shards):
        logger.info("Loading from saved work...")
        # ds = datasets.concatenate_datasets([datasets.load_from_disk(shard) for shard in shards])
        ds = datasets.load_from_disk(save_dir)

    else:
        # if no files found
        ds = datasets.load_dataset(path, name, split=split, **loader_kwargs)
        logger.info(f"Before processing, {ds}'s '{split}' split has {ds.num_rows} records")

        ds = ds.select_columns(key)
        ds = ds.to_iterable_dataset()

        map_kwargs["input_columns"] = key
        # apply standard text normalization and split into sentences
        ds = ds.map(partial(batch_map, key=key, fn=parse_sentences), **map_kwargs)

        # convert iterable to standard dataset for saving
        logger.info("Processing...")
        ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
        logger.info(f"After processing, {ds}'s '{split}' split has {ds.num_rows} records")

        # save to reload later
        logger.info("Saving...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(Path(save_dir), num_shards=N_SHARDS, num_proc=8)
        # for shard_idx in range(N_SHARDS):
        #     shard = ds.shard(num_shards=N_SHARDS, index=shard_idx, contiguous=True)
        #     shard.save_to_disk(Path(save_dir) / f"shard_{N_SHARDS}")

    return ds


def wikipedia(
    name: str = "20231101.en",
    split: str = "train",
    key: str = "text",
    loader_kwargs: Optional[dict] = LOADER_KWARGS,
    map_kwargs: Optional[dict] = MAP_KWARGS,
    # save_dir: Optional[Union[str, Path]] = None,
):
    """Prepare wikimedia/wikipedia dataset with predefined preprocessing steps.

    NOTE: Does not include final normalization or tokenization.
    """
    path = "wikimedia/wikipedia"

    save_prefix = f"{path}_{name}" if name else f"{path}"
    save_dir = Path(DATA_DIR / f"{save_prefix}")

    shards = Path(save_dir).glob("shard_*")
    if any(shards):
        logger.info("Loading from saved work...")
        # ds = datasets.concatenate_datasets([datasets.load_from_disk(shard) for shard in shards])
        ds = datasets.load_from_disk(save_dir)
    else:
        # if no files found
        ds = datasets.load_dataset(path, name, split=split, **loader_kwargs)
        logger.info(f"Before processing, {ds}'s '{split}' split has {ds.num_rows} records")

        ds = ds.select_columns(key)
        ds = ds.to_iterable_dataset()

        # wikipedia specific cleaning
        def clean_wiki_articles(article: str) -> str:
            """Remove standard wikipedia appendices from article, and remove headings."""
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

            # TODO: this causes some articles to hang due to recursive regex
            # # Non-appendix headings defined as spans of 1-5 words between newlines without punctuation.
            # heading = re.compile(
            #     r"""[\n\r]{1}            # match single newline/carriage return
            #         \ *(\w+\ ?){1,5}\ *  # match 1-5 words, with optional preceeding/succeeding space
            #         [\n\r]{1}            # match single newline/carriage return
            #     """,
            #     re.X,
            # )
            # article = heading.sub("\n\n", article)

            return article

        map_kwargs["input_columns"] = key
        # clean tags/linebreaks
        ds = ds.map(partial(batch_map, key=key, fn=prenormalizer.normalize_str), **map_kwargs)
        # clean wikipedia articles
        ds = ds.map(partial(batch_map, key=key, fn=clean_wiki_articles), **map_kwargs)
        # apply standard text normalization and split into sentences
        ds = ds.map(partial(batch_map, key=key, fn=parse_sentences), **map_kwargs)

        # convert iterable to standard dataset for saving
        logger.info("Processing...")
        ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
        logger.info(f"After processing, {ds}'s '{split}' split has {ds.num_rows} records")

        # save to reload later
        logger.info("Saving...")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(Path(save_dir), num_shards=N_SHARDS, num_proc=8)
        # for shard_idx in range(N_SHARDS):
        #     shard = ds.shard(num_shards=N_SHARDS, index=shard_idx, contiguous=True)
        #     shard.save_to_disk(Path(save_dir) / f"shard_{N_SHARDS}")

    return ds
