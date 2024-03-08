from typing import List
from pathlib import Path


def create_output_paths(out_paths: List[Path] = None) -> None:

    for out_path in out_paths:
        if not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()
