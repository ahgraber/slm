# %%
from collections import Counter
import re

import pytest

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
WOULD YOU LIKE THEM WITH A MOUSE?

I DO NOT LIKE THEM IN A HOUSE.
I DO NOT LIKE THEM WITH A MOUSE.
I DO NOT LIKE THEM HERE OR THERE.
I DO NOT LIKE THEM ANYWHERE.
I DO NOT LIKE GREEN EGGS AND HAM.
I DO NOT LIKE THEM, SAM-I-AM!
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
        tok_id = 0
        assert (v.word2id(UNK_TOKEN) == tok_id) and (v.id2word(tok_id) == UNK_TOKEN)
        assert (v[UNK_TOKEN] == tok_id) and (v[tok_id] == UNK_TOKEN)

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
        tok_id = 1 if v._unk_token else 0
        assert (v.word2id(SEP_TOKEN) == tok_id) and (v.id2word(tok_id) == SEP_TOKEN)
        assert (v[SEP_TOKEN] == tok_id) and (v[tok_id] == SEP_TOKEN)

    @pytest.mark.parametrize(
        "v",
        [
            Vocab(unk_token=None, sep_token=None),
            Vocab(unk_token=None, sep_token=SEP_TOKEN),
            Vocab(unk_token=UNK_TOKEN, sep_token=None),
            Vocab(unk_token=UNK_TOKEN, sep_token=SEP_TOKEN),
        ],
    )
    def test_lookups(self, counter, v):
        v.update(counter)
        # 'like' is most-frequent, has index of 0 + n_specials
        tok_id = 0 + len(v._specials)
        assert tok_id == v["LIKE"] == v.word2id("LIKE") == v[v[tok_id]]
        assert "LIKE" == v[tok_id] == v.id2word(tok_id) == v[v["LIKE"]]

    def test_size_gt_n_words(self, counter):
        v = Vocab(counter, size=1_000, min_freq=1, unk_token=None, sep_token=None)

        # len(counter) is ground truth (un-truncated) length
        assert len(counter) == len(v.counter) == len(v.vocab) == len(v)
        assert v.vocab[-1] == min(v.counter)

        # least frequent word must exist in counter and be accessible via lookups
        assert counter["!"] == v.counter["!"]
        assert v["!"] == v[v[-1]] == v[v.vocab[-1]]  # id == id
        assert v[v["!"]] == v[-1] == v.vocab[-1] == v[len(v.vocab) - 1]  # word == word
        with pytest.raises(IndexError):
            v[v.size]  # size > len(v.vocab)

    def test_size_lt_n_words(self, counter, size=10):
        v = Vocab(counter, size=size, min_freq=1, unk_token=None, sep_token=None)

        # len(counter) is ground truth (un-truncated) length
        assert len(counter) == len(v.counter)
        assert size == v.size == len(v) == len(v.vocab)
        assert len(v.counter) > len(v.vocab)  # counter is > truncated vocab

        # least frequent word must exist in counter, but _not_ accessible via lookups
        assert counter["!"] == v.counter["!"]
        assert v[-1] == v.vocab[-1] == v[v.size - 1] == v.vocab[v.size - 1]  # last item in vocab is idx size-1
        assert v[v[-1]] == v[v.vocab[-1]] == v[v[v.size - 1]] == v[v.vocab[v.size - 1]]
        with pytest.raises(ValueError):
            v["!"]  # truncated, error bc no unk_token
        with pytest.raises(IndexError):
            v[len(v.counter)]

    @pytest.mark.parametrize("min_freq", [2, 6, 30])
    def test_min_freq(self, counter, min_freq):
        v = Vocab(counter=counter, min_freq=min_freq, unk_token=None, sep_token=None)

        # assert len(counter) == 27
        if min_freq == 2:
            assert len(v._infrequent) == len([word for word, count in v.counter.items() if count < v.min_freq])
            assert "!" in v._infrequent
        if min_freq == 6:
            assert len(v._infrequent) == len([word for word, count in v.counter.items() if count < v.min_freq])
            assert all(
                word in v._infrequent
                for word in [
                    "!",  # 1
                    "HERE",  # 3
                    "EGGS",  # 4
                    "WOULD",  # 5
                ]
            )
        if min_freq == 30:
            assert len(v._infrequent) == len(v.counter)
            assert all(
                word in v._infrequent
                for word in [
                    "!",  # 1
                    "EGGS",  # 4
                    "WOULD",  # 5
                    "THEM",  # 11
                    "LIKE",  # 16
                ]
            )

    def test_update(self, counter):
        v1 = Vocab(counter, min_freq=1, unk_token=None, sep_token=None)
        v2 = Vocab(counter, min_freq=1, unk_token=None, sep_token=None)
        v2.update(Counter(["I"] * 99))

        # check that counter and vocab have updated
        assert v1.counter != v2.counter
        assert v1.vocab != v2.vocab

        # check that index has shifted
        assert v1["I"] > v2["I"]

    def test_delete_no_specials(self, counter):
        v = Vocab(counter, min_freq=1, unk_token=None, sep_token=None)
        like_idx = v["LIKE"]
        excl_idx = v["!"]

        # confirm exists
        word = "I"
        assert word in v.counter
        assert word in v.vocab
        assert word in v

        v.delete("I")
        assert word not in v.counter
        assert word not in v.vocab
        assert word not in v
        with pytest.raises(ValueError):
            v[word]

        assert like_idx == v["LIKE"]  # indices of words more frequent than deleted do not change
        assert excl_idx != v["!"]  # indices of words less frequent than deleted _do_ change

    def test_delete_with_specials(self, counter):
        v = Vocab(counter, min_freq=1, unk_token=UNK_TOKEN, sep_token=SEP_TOKEN)
        like_idx = v["LIKE"]
        excl_idx = v["!"]

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
        assert excl_idx != v["!"]  # indices of words less frequent than deleted _do_ change


# %%
