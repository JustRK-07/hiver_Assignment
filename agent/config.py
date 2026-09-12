from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_taxonomy(cfg_path: Path | None = None) -> dict:
    """Load and cache the intent taxonomy. Pass cfg_path only in tests."""
    cfg = load_config(cfg_path)
    path = ROOT / cfg["data"]["taxonomy"]
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
