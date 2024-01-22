# %%
# TODO: remove slm setup
from collections import Counter
from pathlib import Path
import sys

import pytest

import nltk
import spacy

from slm.data.preprocess import (
    clean_wiki_articles,
    parse_ngrams,
    parse_sentences,
    parse_words,
    split_sentences_nltk,
    split_sentences_spacy,
    treebank_detokenize,
)
from slm.data.sentence_pretokenizer import SentencePreTokenizer
from slm.word2vec.vocab import Vocab  # NOQA: E402


# %%
@pytest.fixture
def testcases():
    # ref https://gist.github.com/owens2727/b936168921d3468d88bb27d2016044c9
    testcases = {
        "paragraph": """This is a test. It is only a test.""",
        "sentences": [
            "This is a test.",
            "It is only a test.",
        ],
        "words": [
            "This",
            "is",
            "a",
            "test",
            ".",
            "It",
            "is",
            "only",
            "a",
            "test",
            ".",
        ],
        "bigrams": [
            "This is",
            "is a",
            "a test",
            # "test .",
            # ". It",
            "It is",
            "is only",
            "only a",
            "a test",
            # "test .",
        ],
    }

    return testcases


# %%
class TestSentencePreTokenizer:
    def test_nltk(self, testcases):
        spt = SentencePreTokenizer(kind="nltk")

        results = spt._nltk(testcases["paragraph"])

        for r, s in zip(results, testcases["sentences"]):
            assert r == s

    def test_spacy(self, testcases):
        spt = SentencePreTokenizer(kind="spacy")

        results = spt._spacy(testcases["paragraph"])

        for r, s in zip(results, testcases["sentences"]):
            assert str(r).strip() == s

    @pytest.mark.parametrize("kind", ["nltk", "spacy"], ids=["nltk", "spacy"])
    def test_split(self, kind, testcases):
        spt = SentencePreTokenizer(kind=kind)

        results = spt.split(0, testcases["paragraph"])

        for r, s in zip(results, testcases["sentences"]):
            assert str(r).strip() == s


class TestParseSentences:
    """Test `SentencePreTokenizer` in Huggingface tokenization pipeline via `parse_sentences`."""

    @pytest.mark.parametrize("splitter", [split_sentences_nltk, split_sentences_spacy], ids=["nltk", "spacy"])
    def test_split(self, splitter, testcases):
        results = parse_sentences(record=testcases["paragraph"], splitter=splitter)

        for r, s in zip(results, testcases["sentences"]):
            # pipeline includes normalization
            assert str(r).strip() == s.lower()


class TestParseWords:
    def test_split(self, testcases):
        results = parse_words(record=testcases["paragraph"])

        for r, s in zip(results, testcases["words"]):
            # pipeline includes normalization
            assert r == s.lower()


class TestParseNGrams:
    @pytest.fixture
    def vocab_set(self, testcases):
        vocabulary = Vocab(counter=Counter([word.lower() for word in testcases["words"]]))
        return set(vocabulary.vocab)

    def test_split(self, testcases, vocab_set):
        results = parse_ngrams(record=testcases["paragraph"], vocab_set=vocab_set, n=2)

        for r, s in zip(results, testcases["bigrams"]):
            # pipeline includes normalization
            assert r == s.lower()

    def test_filter(self, testcases, vocab_set):
        # add a nonsense word to confirm it's filtered out
        results = parse_ngrams(record=testcases["paragraph"] + "pqwern012385", vocab_set=vocab_set, n=2)

        for r, s in zip(results, testcases["bigrams"]):
            # pipeline includes normalization
            assert r == s.lower()


# %%
class TestTreebank:
    def test_treebank_detokenize(self, testcases):
        tbwt = nltk.tokenize.TreebankWordTokenizer()
        tokenized = tbwt.tokenize(testcases["paragraph"])

        assert testcases["paragraph"] == treebank_detokenize(" ".join(tokenized))

        assert testcases["paragraph"] == treebank_detokenize(testcases["paragraph"])


class TestWikiCleaner:
    def test_clean_wiki_articles(self):
        # from wikimedia/wikipedia dataset
        testcases = [
            """Keep this line\nGallery \nSee also \nBibliography \nIpsum lorem ...""",
            """Keep this line\nWorks cited\n\n work 1, work 2""",
            """Keep this line\nWorks Cited\r\nwork 1, work 2""",
        ]

        for test in testcases:
            assert clean_wiki_articles(test) == "Keep this line\n"
