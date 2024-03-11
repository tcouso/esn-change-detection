from pathlib import Path


def create_output_path(out_path: Path = None) -> None:

    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
