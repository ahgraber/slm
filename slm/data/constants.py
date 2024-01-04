# %%
# MANAGED_DATASETS = ["bookcorpus", "commoncrawl", "wikipedia"]
MANAGED_DATASETS = {
    "bookcorpus": {"path": "bookcorpus", "name": None},
    "commoncrawl": {"path": "c4", "name": "realnewslike"},
    "wikipedia": {"path": "wikimedia/wikipedia", "name": "20231101.en"},
}


SPLIT = "train"
KEY = "text"

N_SHARDS = 32
BATCH_SIZE = 64
BUFFER_SIZE = 1_000
LOADER_KWARGS = {
    "split": SPLIT,
    # ["force_redownload", "reuse_cache_if_exists", "reuse_dataset_if_exists"]
    "download_mode": "reuse_cache_if_exists",
    # ["all_checks", "basic_checks"]
    "verification_mode": "basic_checks",
    "num_proc": 8,
}

MAP_KWARGS = {
    "batched": True,
    "batch_size": BATCH_SIZE,
}


# %%
WIKI_HEADINGS = [
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
