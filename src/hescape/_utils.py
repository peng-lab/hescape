import os
from pathlib import Path


def find_project_root(path: str = ".") -> str:
    """Recursively finds the project root by looking for a '.git' directory."""
    # Start from the current working directory
    current_dir = Path(os.getcwd()).resolve()
    while current_dir != current_dir.parent:
        if (current_dir / ".git").is_dir():
            return str(current_dir)
        current_dir = current_dir.parent
    # If .git is not found, raise an error
    raise FileNotFoundError("Could not find project root with .git directory.")
