# %%
import logging
from typing import Literal, Optional, Union

from nltk.tokenize import PunktSentenceTokenizer
import spacy

from tokenizers import (
    NormalizedString,
    PreTokenizedString,
)

from slm.utils import init_nltk, init_spacy  # NOQA: E402

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
    """Split documents into sentences.

    For use in huggingface tokenizers pipeline.
    Uses NLTK's Punkt model by default.  To use a SpaCy model:

    ```py
    nlp = spacy.load("en_core_web_sm", **SPACY_SENTER_KWARGS)  # Disable parser, other unneeded pipeline components
    nlp.enable_pipe("senter")  # fast sentence segmentation without dependency parses
    presplit = pre_tokenizers.PreTokenizer.custom(SentencePreTokenizer(sentence_tokenizer=nlp))  ```
    """

    def __init__(
        self,
        kind: Optional[Literal["nltk", "spacy"]] = None,
    ):
        self.splitter = kind

        # TODO: This feels hacky & like it encourages side-effects?
        init_nltk(model="punkt")
        init_spacy(model="en_core_web_sm")

    @property
    def splitter(self):
        """Define model used for sentence parsing."""
        return self._splitter

    @splitter.setter
    def splitter(self, kind: Literal["nltk", "spacy"] = "nltk"):
        if kind == "nltk":
            self._splitter = PunktSentenceTokenizer()
        elif kind == "spacy":
            nlp = spacy.load("en_core_web_sm", **SPACY_SENTER_KWARGS)
            nlp.enable_pipe("senter")  # fast sentence segmentation without dependency parses

            self._splitter = nlp
        else:
            raise ValueError("Kind must be in ['nltk','spacy'].")

    def _nltk(self, s: str) -> list[str]:
        """Identify sentences with nltk's Punkt model."""
        spans = self.splitter.span_tokenize(str(s))
        # need to return NormalizedStr if possible
        return [s[s1:s2] for (s1, s2) in spans]

    def _spacy(self, s: str) -> list[str]:
        """Identify sentences with spacy."""
        doc = self.splitter(str(s))
        # need to return NormalizedStr if possible
        return [s[sent.start_char : sent.end_char] for sent in doc.sents]

    def split(self, i: int, s: str):
        """Split into sentences.

        NOTE: int `i` required for huggingface tokenizer indexing.
        """
        if isinstance(i, str):
            raise TypeError("`split` expects (int, str) as args")
        if isinstance(self.splitter, PunktSentenceTokenizer):
            logging.debug("Using nltk punkt sentence splitter")
            return self._nltk(s)
        elif isinstance(self.splitter, spacy.Language):
            logging.debug("Using spacy nlp sentence splitter")
            logging.debug(self.splitter.analyze_pipes())
            return self._spacy(s)
        else:
            raise TypeError(
                "SentencePreTokenizer.sentence_tokenizer is not nltk PunktSentenceTokenizer or spacy Language"
            )

    def pre_tokenize(self, pretok_string: PreTokenizedString):
        """Split the PreTokenizedString.

        Wrapper for huggingface tokenizers integration.
        """
        pretok_string.split(self.split)
