# %%
from collections import Counter

import pytest

from word2vec import loaders as dl

from slm.word2vec import vocab


# %%
class TestSlidingWindowContexts:
    def test_empty_sequence(self):
        # Arrange
        seq = []
        n = 5
        expected_output = []

        # Act & Assert
        output = dl.sliding_window_contexts(seq, n)
        assert output == expected_output, "Incorrect result for empty sequence"

    def test_sequence_shorterthan_window(self):
        # Arrange
        seq = [1, 2, 3]
        n = 20
        expected_output = []

        # Act & Assert
        output = dl.sliding_window_contexts(seq, n)
        assert output == expected_output, "Incorrect result for sequence outside window"

    def test_sequence_within_window(self):
        id_seq = list(range(10))
        expected_3 = [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8],
            [3, 4, 5, 6, 7, 8, 9],
        ]
        expected_4 = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        ]

        assert dl.sliding_window_contexts(id_seq, n=3) == expected_3, "Incorrect result for sequence within window"
        assert dl.sliding_window_contexts(id_seq, n=4) == expected_4, "Incorrect result for sequence within window"


def test_calc_subsample_weights():
    test_counter = Counter(["the"] * 50 + ["a"] * 15 + ["and"] * 5)
    test_vocab = vocab.Vocab(counter=test_counter)
    wt = dl.calc_subsample_weights(test_vocab, t=10e-5)

    # subsample_probabilities is a number that is used to determine whether the token is _retained_
    # more frequent tokens should have lower values
    assert wt["the"] < wt["a"] < wt["and"]


def test_subsample_freq_words():
    # fmt: off
    record = {
        "tokens": [
            "<S>","the","result",":","they","all","get","32","mandates",",","with","the","likud","pushed","to",
            "second","place","with","30",",","while","labor","falls","into","the","3",".","25","%","abyss","and",
            "goes","to","pole","paradise","with","a","<UNK>","3","seats","(","down","from","24",")",".","</S>"
        ],
    }
    # fmt: on

    # make the most-frequent word (the) so frequent nothing else will get dropped
    test_counter = Counter(record["tokens"] + ["the"] * 50000)
    test_vocab = vocab.Vocab(counter=test_counter, min_freq=0)
    word2sswt = dl.calc_subsample_weights(test_vocab, t=10e-5)

    # hack record with token ids from vocab
    record["ids"] = [test_vocab[w] for w in record["tokens"]]

    ids_seq = dl.subsample_freq_words(record, word2sswt=word2sswt)

    assert len(ids_seq) < len(record["ids"])
    assert test_vocab["the"] not in ids_seq
