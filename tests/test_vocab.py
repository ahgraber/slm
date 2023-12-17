# %%
from collections import Counter, OrderedDict
from functools import partial
import itertools
import logging
import os
from pathlib import Path
import random
import re
import string
import sys
from typing import Any, Callable, Iterable, Optional, Union
import unicodedata

import nltk
import pytest
import spacy
from tqdm import tqdm

import numpy as np

sys.path.insert(0, str(Path(__file__ + "/../src").resolve()))
from slm.word2vec.vocab import SEP_TOKEN, UNK_TOKEN, Vocab  # NOQA: E402


# %%
@pytest.fixture
def text() -> str:
    text = """
DO YOU LIKE GREEN EGGS AND HAM?

I DO NOT LIKE THEM,SAM-I-AM.
I DO NOT LIKE GREEN EGGS AND HAM.

WOULD YOU LIKE THEM HERE OR THERE?

I WOULD NOT LIKE THEM HERE OR THERE.
I WOULD NOT LIKE THEM ANYWHERE.
I DO NOT LIKE GREEN EGGS AND HAM.
I DO NOT LIKE THEM, SAM-I-AM.

WOULD YOU LIKE THEM IN A HOUSE?
WOULD YOU LIKE THEN WITH A MOUSE?

I DO NOT LIKE THEM IN A HOUSE.
I DO NOT LIKE THEM WITH A MOUSE.
I DO NOT LIKE THEM HERE OR THERE.
I DO NOT LIKE THEM ANYWHERE.
I DO NOT LIKE GREEN EGGS AND HAM.
I DO NOT LIKE THEM, SAM-I-AM.
"""
    return " ".join(text.splitlines())


@pytest.fixture
def counter(text: str) -> Counter:
    split = re.findall(r"\w+|[^\w\s]+", text)
    return Counter(split)


class TestVocab:
    def test_init(self):
        v = Vocab()

        assert isinstance(v.counter, Counter)
        assert len(v.counter) == 0
        assert isinstance(v.vocab, list)
        assert len(v.vocab) == 0
        assert isinstance(v._infrequent, set)
        assert len(v._infrequent) == 0

    def test_init_vs_update(self, counter):
        v1 = Vocab()
        v1.update(counter)
        v2 = Vocab(counter=counter)
        assert v1.vocab == v2.vocab

    @pytest.mark.parametrize("v", [Vocab(), Vocab(unk_token=UNK_TOKEN)])
    def test_unk(self, v, counter):
        assert v._unk_token == UNK_TOKEN
        assert UNK_TOKEN in v._specials

        # test if unk was added to vocab objects
        v.update(counter)
        assert UNK_TOKEN in v.vocab
        assert UNK_TOKEN in v
        # counter does not include special tokens
        assert len(v) == len(v.vocab) != len(v.counter)

        # token ids should be deterministic
        unk_id = 0
        assert v.word2id(UNK_TOKEN) == unk_id
        assert v[UNK_TOKEN] == unk_id
        assert v.id2word(unk_id) == UNK_TOKEN

    @pytest.mark.parametrize(
        "v",
        [
            Vocab(unk_token=None, sep_token=SEP_TOKEN),
            Vocab(unk_token=UNK_TOKEN, sep_token=SEP_TOKEN),
        ],
    )
    def test_sep(self, v, counter):
        assert v._sep_token == SEP_TOKEN
        assert SEP_TOKEN in v._specials

        # test if sep was added to vocab objects
        v.update(counter)
        assert SEP_TOKEN in v.vocab
        assert SEP_TOKEN in v
        # counter does not include special tokens
        assert len(v) == len(v.vocab) != len(v.counter)

        # token ids should be deterministic
        sep_id = 1 if v._unk_token else 0
        assert v.word2id(SEP_TOKEN) == sep_id
        assert v[SEP_TOKEN] == sep_id
        assert v.id2word(sep_id) == SEP_TOKEN

    @pytest.mark.parametrize(
        "v",
        [
            Vocab(unk_token=None, sep_token=None),
            Vocab(unk_token=None, sep_token=SEP_TOKEN),
            Vocab(unk_token=UNK_TOKEN, sep_token=None),
            Vocab(unk_token=UNK_TOKEN, sep_token=SEP_TOKEN),
        ],
    )
    def test_lookups(self, text, counter, v):
        v.update(counter)
        # 'like' is most-frequent, has index of 0 + n_specials
        assert v["LIKE"] == v.word2id("LIKE") == 0 + len(v._specials)
        assert v.id2word(0 + len(v._specials)) == "LIKE"

        for word in ["LIKE", "SAM", "GREEN"]:
            assert v.wordfreq(word) == text.count(word)

    @pytest.mark.parametrize("size", [6, 15, 30])
    def test_size(self, counter, size):
        v = Vocab(counter=counter, size=size, unk_token=None, sep_token=None)

        # assert len(counter) == 27
        if size < len(counter):
            assert len(v.vocab) == size
        else:
            assert len(v.vocab) == len(counter)

    @pytest.mark.parametrize("min_freq", [2, 6, 30])
    def test_min_freq(self, counter, min_freq):
        v = Vocab(counter=counter, min_freq=min_freq)

        # assert len(counter) == 27
        if min_freq == 2:
            assert len(v._infrequent) == len([word for word, count in v.counter.items() if count < min_freq])
            assert "THEN" in v._infrequent
        if min_freq == 6:
            assert len(v._infrequent) == len([word for word, count in v.counter.items() if count < min_freq])
            assert all(
                word in v._infrequent
                for word in [
                    "THEN",  # 1
                    "HERE",  # 3
                    "EGGS",  # 4
                    "WOULD",  # 5
                ]
            )
        if min_freq == 30:
            assert len(v._infrequent) == len(counter)
            assert all(
                word in v._infrequent
                for word in [
                    "THEN",  # 1
                    "EGGS",  # 4
                    "WOULD",  # 5
                    "THEM",  # 11
                    "LIKE",  # 16
                ]
            )

    def test_update(self, counter):
        v1 = Vocab(counter)

        v2 = Vocab(counter)
        v2.update(Counter(["I"] * 99))

        # check that counter and vocab have updated
        assert v1.counter != v2.counter
        assert v1.vocab != v2.vocab

        # check that index has shifted
        assert v1["I"] > v2["I"]

    def test_delete(self, counter):
        v = Vocab(counter)
        like_idx = v["LIKE"]
        then_idx = v["THEN"]

        # confirm exists
        word = "I"
        assert word in v.counter
        assert word in v.vocab
        assert word in v

        v.delete("I")
        assert word not in v.counter
        assert word not in v.vocab
        assert word not in v
        assert v[word] == v[UNK_TOKEN]

        assert like_idx == v["LIKE"]  # indices of words more frequent than deleted do not change
        assert like_idx != v["THEN"]  # indices of words less frequent than deleted _do_ change


# %%
