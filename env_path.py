import sys
from pathlib import Path


def _to_str_path(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths]


BASE_DIR = Path(__file__).parent

sys.path.extend(_to_str_path([
    BASE_DIR / "keypoint_models",
    BASE_DIR / "preprocess"
]))
