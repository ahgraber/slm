# %%
import logging
import re
import string
from typing import Callable, Optional, Union

from nltk.tokenize import PunktSentenceTokenizer
import spacy

from datasets import Dataset, IterableDataset
from tokenizers import (
    NormalizedString,
    PreTokenizedString,
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

# %%
logger = logging.getLogger(__name__)

# %%
SPACY_SENTER_KWARGS = {"exclude": ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]}

nlp = spacy.load("en_core_web_sm", **SPACY_SENTER_KWARGS)
nlp.enable_pipe("senter")  # fast sentence segmentation without dependency parses
# nlp.add_pipe("sentencizer")  # even faster, rule-based sentence segmentation


# %%
# we know wikipedia layout schema; we can remove any 'see also' links and references
# https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Layout#Standard_appendices_and_footers
WIKI_APPENDICES = [
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
]


def wiki_truncate_appedices(article: str) -> str:
    """Remove standard wikipedia appendices from article blob."""
    for heading in WIKI_APPENDICES:
        match = re.search(heading + r"\s\n", article)
        if match:
            article = article[: match.start()]

    return article


# this is wikipedia - we can remove headings that are more easily identified now in sentnce form
def wiki_remove_headings(article: str) -> str:
    """Remove headings.

    Headings defined as spans of 1-5 words between newlines without punctuation.
    """
    heading = re.compile(
        r"""[\n\r]{1}            # match single newline/carriage return
            \ *(\w+\ ?){1,5}\ *  # match 1-5 words, with optional preceeding/succeeding space
            [\n\r]{1}            # match single newline/carriage return
        """,
        re.X,
    )
    return heading.sub("\n\n", article)


# %%
# pre-token splitting (sentences)
# ref: https://github.com/huggingface/tokenizers/blob/v0.15.0/bindings/python/py_src/tokenizers/__init__.pyi
# use [spacy](https://spacy.io/) or [stanza](https://stanfordnlp.github.io/stanza/)
class SentencePreTokenizer:
    """Split documents into sentences.  Uses NLTK's Punkt model by default.

    To use a SpaCy model:

    ```py
    nlp = spacy.load("en_core_web_sm", **SPACY_SENTER_KWARGS)  # Disable parser, other unneeded pipeline components
    nlp.enable_pipe("senter")  # fast sentence segmentation without dependency parses
    presplit = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer(sentence_tokenizer=nlp))  ```
    """

    def __init__(self, sentence_tokenizer: Optional[Union[PunktSentenceTokenizer, spacy.Language]] = None):
        self.splitter = sentence_tokenizer if sentence_tokenizer else PunktSentenceTokenizer()

    def nltk_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        """Use nltk's Punkt model to split into sentences."""
        spans = self.splitter.span_tokenize(str(normalized_string))
        return [normalized_string[s1:s2] for (s1, s2) in spans]

    def spacy_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        """Use spacy to split into sentences."""
        doc = self.splitter(str(normalized_string))
        return [normalized_string[s.start_char : s.end_char] for s in doc.sents]

    def pre_tokenize(self, pretok_string: PreTokenizedString):
        """Split the PreTokenizedString."""
        if isinstance(self.splitter, spacy.Language):
            logging.debug("Using spacy nlp sentence splitter")
            logging.debug(self.splitter.analyze_pipes())
            pretok_string.split(self.spacy_split)
        elif isinstance(self.splitter, PunktSentenceTokenizer):
            logging.debug("Using nltk punkt sentence splitter")
            pretok_string.split(self.nltk_split)
        else:
            raise TypeError(
                "SentencePreTokenizer.sentence_tokenizer is not spacy Language or nltk PunktSentenceTokenizer"
            )


# %%
# prenorm = normalizers.Replace(Regex(r"[\p{Other}&&[^\n\t\r]]"), "\n")  # normalize control chars
# presplit = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer())  # use nltk punkt sentence splitter


def blob_to_sentences(
    blob: str,
    is_wiki: bool,
    normalizer: normalizers.Normalizer,
    splitter: pre_tokenizers.PreTokenizer,
) -> list[str]:
    """Split blob into sentences.

    If dataset is wikipedia, remove appendices and headings.
    """
    blob = normalizer.normalize_str(blob)
    if is_wiki:
        blob = wiki_truncate_appedices(blob)
        blob = wiki_remove_headings(blob)
    return [sentence for sentence, _span in splitter.pre_tokenize_str(blob)]


def all_punctuation(s: str):
    """Return True if string contains only punctuation."""
    return all(ch in string.punctuation for ch in s)


def sentences_to_words(
    sentences: list[str],
    normalizer: normalizers.Normalizer,
    splitter: pre_tokenizers.PreTokenizer,
) -> list[list[str]]:
    """Apply standard text normalization and split into words."""
    words = []
    for sentence in sentences:
        sentence = normalizer.normalize_str(sentence)
        words.append([word for word, _span in splitter.pre_tokenize_str(sentence) if not all_punctuation(word)])
    return words


def pipeline(batch: list[str], key: str, b2s_kwargs: dict, s2w_kwargs: dict) -> dict[str, list[list[str]]]:
    """Wrapper function for preprocessing pipeline."""
    batch = [blob_to_sentences(blob, **b2s_kwargs) for blob in batch]
    batch = [sentences_to_words(sentences, **s2w_kwargs) for sentences in batch]
    return {key: batch}
