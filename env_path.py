import sys
from pathlib import Path

from typing import List


def _to_str_path(paths: List[Path]) -> List[str]:
    return [str(path) for path in paths]


BASE_DIR = Path(__file__).parent

sys.path.extend(_to_str_path([
    BASE_DIR / "keypoint_models",
    BASE_DIR / "preprocess"
]))
