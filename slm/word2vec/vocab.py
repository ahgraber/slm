# %%
from collections import Counter
import logging
from typing import Optional, Union

# %%
logger = logging.getLogger(__name__)

# %%
VOCAB_SIZE = 60_000
VOCAB_MIN_FREQ = 3
UNK_TOKEN = "<UNK>"
SEP_TOKEN = "<SEP>"


class Vocab:
    """Vocabulary must be able to map words:tokens (ids), tokens:words and words:counts.

    Word ids are not immutable; running Vocab.update() may alter the word:id mapping or remove infrequent words from the lexicon.  This means that Vocab instances must be tightly coupled with their tokenization pipelines.

    Notes
    -----
    `counter` is a much more stable representation of the lexicon than `vocab`;
    words do not get removed `counter` unless `delete` is specifically called,
    while `vocab` is influenced by both `size` limits and `min_freq` limits.
    """

    def __init__(
        self,
        counter: Optional[Counter] = None,
        size: Optional[int] = VOCAB_SIZE,
        min_freq: Optional[int] = VOCAB_MIN_FREQ,
        unk_token: Optional[str] = UNK_TOKEN,
        sep_token: Optional[str] = None,
    ):
        self.size = size
        self.min_freq = min_freq

        self.vocab = []

        self._unk_token = unk_token
        self._sep_token = sep_token
        self._specials = []
        if self._unk_token:
            self._specials.append(unk_token)
        if self._sep_token:
            self._specials.append(sep_token)

        self._infrequent = set()

        self.counter = Counter()
        if counter is not None:
            self.update(counter)

    def __len__(self) -> int:
        """Retrieve number of items in vocab."""
        return len(self.vocab)

    def __contains__(self, word: str) -> bool:
        """Check whether word exists in vocabulary."""
        if word in self._infrequent:  # handle deleted case
            return False

        return self.vocab.__contains__(word)

    def __getitem__(self, item: Union[int, str]) -> Union[int, str]:
        """Return word from id or id from word."""
        match item:
            case int():
                return self.id2word(item)
            case str():
                return self.word2id(item)

    def __sort_counter(self, count_tuples: tuple[str, int]) -> list[tuple[str, int]]:
        """Sort Counter by descending counts, then alphabetically for tiebreaking."""
        sorted_counts = sorted(
            count_tuples,
            key=lambda x: (-x[1], x[0]),
        )
        return sorted_counts

    def to_dict(self) -> dict[str, int]:
        """Return {word: id} for entire vocabulary of len `size` and with at least `min_freq` counts."""
        return {word: i for i, word in enumerate(self.vocab) if word not in self._infrequent}

    def word2id(self, word: str) -> int:
        """Return id of given word."""
        if (word in self._infrequent) or (word not in self.vocab):
            logger.debug(f"'{word}' is infrequent, returning `unk` if enabled")
            if self._unk_token:
                return self.vocab.index(self._unk_token)
            else:
                raise ValueError(f"{word} is not in vocab")

        return self.vocab.index(word)

    def id2word(self, idx: int):
        """Return word of given token id."""
        if idx > len(self.vocab):
            if self._unk_token:
                return self._unk_token
            else:
                raise IndexError(f"{idx} out of vocab range")

        return self.vocab[idx]

    def wordfreq(self, word: str):
        """Return count/frequency of given word."""
        return self.counter[word]

    def _update_vocab(self):
        """Update vocab and infrequent words based on current counts."""
        self.vocab = [word for word, _count in self.__sort_counter(self.counter.most_common(self.size))]
        self._infrequent = {word for word, count in self.counter.items() if count < self.min_freq}
        if len(self._specials) > 0:
            self.vocab = self._specials + self.vocab

    def update(self, counts: Counter):
        """Update Vocab object with new words/counts.

        `counts` will be added to existing counts, and new words appended to the vocabulary.
        This means that new words may be out of order vs. count frequency index.
        """
        logging.warning("Updating the vocabulary will alter word:token mapping!")
        self.counter.update(counts)
        self._update_vocab()

    def delete(self, word):
        """Delete word from vocabulary."""
        logging.warning("Deleting a word from the vocabulary _will_ alter word:token mapping!")
        _ = self.counter.pop(word)
        self._update_vocab()
