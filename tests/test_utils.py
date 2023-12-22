# %%
from collections import Counter, OrderedDict
from functools import partial
import itertools

import pytest

from slm.utils import flatten  # NOQA: E402


# %%
class TestFlatten:
    testcases = [
        ("This is a test.  It is only a test", ["This is a test.  It is only a test"]),
        (["This is a test.  It is only a test"], ["This is a test.  It is only a test"]),
        (
            ["This", "is", "a", "test.", "It", "is", "only", "a", "test."],
            ["This", "is", "a", "test.", "It", "is", "only", "a", "test."],
        ),
        (
            [["This", "is", "a", "test."], ["It", "is", "only", "a", "test."]],
            ["This", "is", "a", "test.", "It", "is", "only", "a", "test."],
        ),
        (b"This is a test.  It is only a test", [b"This is a test.  It is only a test"]),
        ([b"This is a test.  It is only a test"], [b"This is a test.  It is only a test"]),
        (
            [b"This", b"is", b"a", b"test.", b"It", b"is", b"only", b"a", b"test."],
            [b"This", b"is", b"a", b"test.", b"It", b"is", b"only", b"a", b"test."],
        ),
        (
            [[b"This", b"is", b"a", b"test."], [b"It", b"is", b"only", b"a", b"test."]],
            [b"This", b"is", b"a", b"test.", b"It", b"is", b"only", b"a", b"test."],
        ),
        (8675309, [8675309]),
        ([8675309], [8675309]),
        ([8, 6, 7, 5, 3, 0, 9], [8, 6, 7, 5, 3, 0, 9]),
        ([[8, 6, 7], [5, 3, 0, 9]], [8, 6, 7, 5, 3, 0, 9]),
    ]

    @pytest.mark.parametrize("testcase,expected", testcases)
    def test_flatten(self, testcase, expected):
        assert list(flatten(testcase)) == expected


# %%
