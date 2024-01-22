"""Build vocab from preprocessed dataset."""

# %%
import argparse
import collections
import itertools
import logging
from pathlib import Path
import pickle
import sys

import tqdm

sys.path.insert(0, str(Path(__file__ + "/../../").resolve()))
from slm.data import constants  # NOQA: E402
from slm.data.preprocess import load_data, parse_ngrams, parse_words  # NOQA: E402
from slm.utils import get_project_root  # NOQA: E402
from slm.word2vec.vocab import Vocab  # NOQA: E402

# %%
logger = logging.getLogger(__name__)
LOG_FMT = "%(asctime)s - %(levelname)-8s - %(name)s - %(funcName)s:%(lineno)d - %(message)s"  # noqa: N806
logging.basicConfig(format=LOG_FMT)
logging.captureWarnings(True)
logger.setLevel(logging.INFO)

slm_logger = logging.getLogger("slm")
slm_logger.setLevel(logging.INFO)

# %%
ROOT_DIR = get_project_root()
ARTIFACT_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", required=True, choices=constants.MANAGED_DATASETS)
    parser.add_argument("-n", "--name", required=False, type=str, default=None)
    # parser.add_argument("-s", "--split", required=False, type=str, default="train")
    parser.add_argument("-k", "--key", required=False, type=str, default="text")
    parser.add_argument("-i", "--data_dir", required=False, type=Path, default=None)
    # parser.add_argument("-o", "--save_file", required=False, type=Path, default=None)

    args = parser.parse_args()
    dataset = args.dataset
    path = constants.MANAGED_DATASETS[dataset]["path"]
    name = args.name if args.name else constants.MANAGED_DATASETS[dataset]["name"]
    # split = args.split
    key = args.key

    data_dir = args.data_dir
    # save_file = args.save_file

    save_prefix = f"{path}/{name}" if name else path
    save_file = Path(ARTIFACT_DIR / "vocab" / f"{save_prefix.replace('/', '_')}_vocab.pkl")
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if save_file.exists():
        print("File exists at save path.  Continuing will remove and replace existing file.")
        response = str(input("Continue? [y/N]  "))
        if response.lower() not in ["y", "yes"]:
            print("Exiting at user request.")
            sys.exit(0)
        else:
            save_file.unlink()  # delete file

    map_kwargs = constants.MAP_KWARGS
    map_kwargs["input_columns"] = key

    dset = load_data(
        managed_ds=dataset,
        data_dir=data_dir,
    )
    dsamples = dset.num_rows
    logger.info(f"{dataset} has {dsamples} records")

    dset = dset.to_iterable_dataset(num_shards=constants.N_SHARDS)

    logger.info("Begin processing dataset & counting...")
    for n in [1, 2, 3]:
        iterds = iter(dset)
        counter = collections.Counter()
        if n == 1:
            for record in tqdm.tqdm(iterds, desc="Words", total=dsamples):
                counter.update(parse_words(record[key]))
        else:
            if save_file.exists():
                with save_file.open("rb") as f:
                    _v = pickle.load(f)  # NOQA: S301
                    vocab_set = set(_v.vocab)
                    del _v
            else:
                vocab_set = set()

            for record in tqdm.tqdm(iterds, desc=f"{n}_grams", total=dsamples):
                counter.update(parse_ngrams(record[key], vocab_set=vocab_set, n=n))

        logger.info("Creating vocab...")
        vocabulary = Vocab(collections.Counter(dict(itertools.takewhile(lambda itm: itm[1] > 1, counter.items()))))

        logger.info("Saving vocab...")
        if save_file.exists():
            with save_file.open("rb") as f:
                _v = pickle.load(f)  # NOQA: S301
                vocabulary.update(_v.counter)
        with save_file.open("wb") as f:
            pickle.dump(vocabulary, f)

    logger.info("Process complete.")
