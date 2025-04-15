import subprocess
from pathlib import Path

def get_git_root():
    try:
        root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
        return Path(root)
    except subprocess.CalledProcessError:
        raise RuntimeError("Not a git repository")
