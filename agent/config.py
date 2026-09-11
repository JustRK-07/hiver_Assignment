from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_taxonomy(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    path = ROOT / cfg["data"]["taxonomy"]
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
