# %%
from pathlib import Path
import subprocess


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
