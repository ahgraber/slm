# %%
from itertools import chain
import logging
from typing import Callable, Optional, Union

from nltk.tokenize import PunktSentenceTokenizer
import spacy

from tokenizers import (
    NormalizedString,
    PreTokenizedString,
    Regex,
    normalizers,
    pre_tokenizers,
)

from slm.utils import flatten

# %%
logger = logging.getLogger(__name__)

# %%
SPACY_SENTER_KWARGS = {"exclude": ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]}

nlp = spacy.load("en_core_web_sm", **SPACY_SENTER_KWARGS)
nlp.enable_pipe("senter")  # fast sentence segmentation without dependency parses
# nlp.add_pipe("sentencizer")  # even faster, rule-based sentence segmentation


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

    def __init__(
        self,
        sentence_tokenizer: Optional[Union[PunktSentenceTokenizer, spacy.Language]] = None,
    ):
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
nltk_sentence_splitter = pre_tokenizers.PreTokenizer.custom(
    SentencePreTokenizer(sentence_tokenizer=PunktSentenceTokenizer())
)
spacy_sentence_splitter = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer(sentence_tokenizer=nlp))

word_splitter = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ]
)


def parse_sentences(
    record: str,
    normalizer: normalizers.Normalizer = normalizer,
    splitter: pre_tokenizers.PreTokenizer = spacy_sentence_splitter,
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


def batch_map(
    batch: list[str],
    key: str,
    fn: Callable,
    **kwargs,
) -> dict[str, list[list[str]]]:
    """Allow function that handles single examples to operate over batch."""
    return {key: list(flatten([fn(record, **kwargs) for record in batch]))}


# %%
