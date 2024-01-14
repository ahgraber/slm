# %%
from collections import defaultdict
from functools import partial
import logging
from pathlib import Path
import re
from typing import Any, Callable, Optional, Union

from nltk.tokenize import TreebankWordDetokenizer
from nltk.util import ngrams

import datasets
from tokenizers import (
    Regex,
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    processors,
)

from slm.data import constants
from slm.data.sentence_pretokenizer import SentencePreTokenizer
from slm.utils import flatten, get_project_root
from slm.word2vec.vocab import END_TOKEN, START_TOKEN, UNK_TOKEN, Vocab  # NOQA: E402

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


def w2v_tokenizer(vocab: Vocab, start=START_TOKEN, end=END_TOKEN):
    """Instantiate WordLevel tokenizer from vocab."""
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab.to_dict(), unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = word_splitter
    tokenizer.post_processor = processors.TemplateProcessing(  # add separators
        single=f"{start} $0 {end}",
        pair=f"{start} $A {start} {end} $B:1 {end}:1",  # unneeded for word2vec
        special_tokens=[
            (start, tokenizer.token_to_id(start)),
            (end, tokenizer.token_to_id(end)),
        ],
    )

    return tokenizer


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
) -> list[str]:
    """Apply standard text normalization and split blob into words.

    Useful for extracting similar WordLevel tokens if Tokenizer cannot be used due to lack of Vocabulary.
    """
    record = normalizer.normalize_str(record)
    return [word for word, _span in splitter.pre_tokenize_str(record)]


def parse_ngrams(
    record: str,
    n: int = 2,
    normalizer: normalizers.Normalizer = normalizer,
    splitter: pre_tokenizers.PreTokenizer = word_splitter,
) -> list[str]:
    """Apply standard text normalization and split blob into words, then identify ngrams.

    Useful for extracting similar WordLevel tokens if Tokenizer cannot be used due to lack of Vocabulary.
    """
    record = normalizer.normalize_str(record)
    words = [word for word, _span in splitter.pre_tokenize_str(record)]
    return [" ".join(gram) for gram in ngrams(words, 2)]


# %%
def tokenize(record: str, tokenizer: Tokenizer) -> dict[str, list[Any]]:
    """Convert Tokenizer output to dict."""
    output = tokenizer.encode(record)
    return {
        "ids": output.ids,
        "tokens": output.tokens,
        "attention_mask": output.attention_mask,
    }


def tokenize_batch(batch: list[str], tokenizer: Tokenizer) -> dict[str, list[list[Any]]]:
    """Convert batched tokenization to dict."""
    outbatch = defaultdict(list)
    for record in batch:
        output = tokenizer.encode(record)
        outbatch["ids"].append(output.ids)
        outbatch["tokens"].append(output.tokens)
        outbatch["attention_mask"].append(output.attention_mask)

    return dict(outbatch)


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
        save_prefix = f"{dataset}/{name}" if name else dataset
        save_dir = DATA_DIR / save_prefix
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
    ds = datasets.Dataset.from_generator(
        partial(gen_from_iterable_dataset, ds),
        features=ds.features,
    )
    logger.info(f"After processing, {dataset}'s '{loader_kwargs['split']}' split has {ds.num_rows} records")

    # save to reload later
    logger.info("Saving...")
    ds.save_to_disk(
        Path(save_dir),
        num_shards=constants.N_SHARDS,
        num_proc=loader_kwargs["num_proc"],
    )

    logger.info("Preprocessing completee!")
    return ds


def load_data(
    managed_ds: Optional[str] = None,
    data_dir: Optional[Union[Path, str]] = None,
) -> datasets.Dataset:
    """Load dataset from local files / managed datasets.

    NOTE: Does not include final normalization or tokenization.
    """
    if data_dir is None:
        path = constants.MANAGED_DATASETS[managed_ds]["path"]
        name = constants.MANAGED_DATASETS[managed_ds]["name"]

        save_prefix = f"{path}/{name}" if name else path
        data_dir = DATA_DIR / save_prefix

    else:
        data_dir = Path(data_dir)

    shards = Path(data_dir).glob("*.arrow")
    if any(shards):
        logger.info("Loading from saved work...")
        return datasets.load_from_disk(data_dir)

    else:
        raise FileNotFoundError(f"No preprocessed data found in {data_dir}.  Run `prep_data` first?")
