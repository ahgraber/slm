# %%
import logging
import os
from pathlib import Path
import subprocess

# %%
logger = logging.getLogger(__name__)


# %%
def get_project_root():
    """Return top-level directory of (current) git repo."""
    git_call = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True)
    repo_path = Path(git_call.stdout.decode("utf-8").strip())
    return repo_path


def torch_device():
    """Detect and return best device for pytorch."""
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


# %%
def init_nltk(model: str = "punkt", savedir: Path = Path(".")):
    """Ensure NLTK model is downloaded/available."""
    import nltk

    savedir.mkdir(parents=True, exist_ok=True)

    # specify download dir for nltk by setting NLTK_DATA env var
    os.environ["NLTK_DATA"] = str(savedir.resolve())
    nltk.download(model)


def init_spacy(model: str = "en_core_web_sm"):
    """Ensure spaCy model model is downloaded/available."""
    import spacy

    try:
        _ = spacy.load(model)
    except OSError:
        spacy.cli.download(model)
