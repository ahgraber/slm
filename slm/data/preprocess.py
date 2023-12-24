# %%
from functools import partial
import logging
from pathlib import Path
import re
from typing import Callable, Optional, Union

from nltk.tokenize import TreebankWordDetokenizer

import datasets
from tokenizers import (
    Regex,
    normalizers,
    pre_tokenizers,
)

from slm.data import constants
from slm.data.sentence_pretokenizer import SentencePreTokenizer
from slm.utils import flatten, get_project_root

# %%
logger = logging.getLogger(__name__)

ROOT_DIR = get_project_root()
DATA_DIR = ROOT_DIR / "data"

# %%
prenormalizer = normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), "\n")  # normalize control chars
normalizer = normalizers.Sequence(
    [
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), ""),  # normalize control chars
        normalizers.Replace(Regex(r"[\s]"), " "),  # normalize whitespace
    ]
)
split_sentences_nltk = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer(kind="nltk"))
split_sentences_spacy = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer(kind="spacy"))

word_splitter = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ]
)


def parse_sentences(
    record: str,
    normalizer: normalizers.Normalizer = normalizer,
    splitter: pre_tokenizers.PreTokenizer = split_sentences_spacy,
) -> list[str]:
    """Apply standard text normalization and split blob into sentences."""
    record = normalizer.normalize_str(record)
    return [sentence for sentence, _span in splitter.pre_tokenize_str(record)]


def parse_words(
    record: str,
    normalizer: normalizers.Normalizer = normalizer,
    splitter: pre_tokenizers.PreTokenizer = word_splitter,
) -> list[list[str]]:
    """Apply standard text normalization and split blob into words.

    Useful for extracting similar WordLevel tokens if Tokenizer cannot be used due to lack of Vocabulary.
    """
    record = normalizer.normalize_str(record)
    return [word for word, _span in splitter.pre_tokenize_str(record)]


# %%
twd = TreebankWordDetokenizer()


def treebank_detokenize(record: str, twd: TreebankWordDetokenizer = twd) -> str:
    """Revert NLTK's TreebankWordTokenizer.

    Specifically, bookcorpus appears to have been pretokenized.
    ref: https://github.com/huggingface/datasets/issues/486
    """
    return twd.detokenize(record.split())


# wikipedia specific cleaning
def clean_wiki_articles(article: str, headings=constants.WIKI_HEADINGS) -> str:
    """Remove standard wikipedia appendices from article."""
    # we know wikipedia layout schema; we can remove any 'see also' links and references
    # https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Layout#Standard_appendices_and_footers
    for heading in headings:
        # headings tend to be followed by at least 2 whitespace chars, including a newlinee
        match = re.search(heading + r"\s\n", article)
        if match:
            article = article[: match.start()]

    return article


# %%
def batch_map(
    batch: list[str],
    key: str,
    fn: Callable,
    **kwargs,
) -> dict[str, list[list[str]]]:
    """Allow function that handles single examples to operate over batch."""
    return {key: list(flatten([fn(record, **kwargs) for record in batch]))}


def gen_from_iterable_dataset(ds: datasets.IterableDataset):
    """Convert IterableDataset to generator.

    NOTE: Helper fn. for converting IterableDataset to Dataset.
    """
    yield from ds


def prep_data(
    dataset: str,
    name: str = None,
    key: str = constants.KEY,
    map_fns: list[Callable] = None,
    save_dir: Union[Path, str] = None,
    loader_kwargs: Optional[dict] = constants.LOADER_KWARGS,
    map_kwargs: Optional[dict] = constants.MAP_KWARGS,  # include map_fns
):
    """Prepare dataset given preprocessing functions.

    NOTE: Does not include final normalization or tokenization.
    """
    if save_dir is None:
        save_prefix = f"{dataset}_{name}" if name else f"{dataset}"
        save_dir = DATA_DIR / f"{save_prefix}"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = Path(save_dir).glob("*.arrow")
    if any(shards):
        raise FileExistsError("File(s) exist in save dir.  Load with `load_data` or delete.")

    # if no files found
    ds = datasets.load_dataset(dataset, name, **loader_kwargs)
    logger.info(f"Before processing, {dataset}'s '{loader_kwargs['split']}' split has {ds.num_rows} records")

    ds = ds.select_columns(key)
    ds = ds.to_iterable_dataset()

    if map_fns is None:
        logger.info("Using default preprocessing function(s).")
        map_fns = [
            parse_sentences,
        ]
    map_kwargs["input_columns"] = key

    for fn in map_fns:
        ds = ds.map(partial(batch_map, key=key, fn=fn), **map_kwargs)

    # convert iterable to standard dataset for saving
    logger.info("Processing...")
    ds = datasets.Dataset.from_generator(partial(gen_from_iterable_dataset, ds), features=ds.features)
    logger.info(f"After processing, {dataset}'s '{loader_kwargs['split']}' split has {ds.num_rows} records")

    # save to reload later
    logger.info("Saving...")
    ds.save_to_disk(Path(save_dir), num_shards=constants.N_SHARDS, num_proc=loader_kwargs["num_proc"])

    logger.info("Preprocessing completee!")
    return ds


def load_data(
    dataset: str,
    name: str = None,
    data_dir: Union[Path, str] = None,
):
    """Load preprocessing dataset from local files.

    NOTE: Does not include final normalization or tokenization.
    """
    if data_dir is None:
        save_prefix = f"{dataset}_{name}" if name else f"{dataset}"
        data_dir = DATA_DIR / f"{save_prefix}"
    else:
        data_dir = Path(data_dir)

    shards = Path(data_dir).glob("*.arrow")
    if any(shards):
        logger.info("Loading from saved work...")
        return datasets.load_from_disk(data_dir)

    else:
        raise FileNotFoundError(f"No preprocessed data found in {data_dir}.  Run `prep_data` first?")
