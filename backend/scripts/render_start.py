import sys
import os
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import DEFAULT_DATA_PATH, DEMO_DATA_PATH, MODEL_PATH  # noqa: E402
from scripts.generate_demo_data import build_demo_dataset  # noqa: E402
from train import train  # noqa: E402


def ensure_model() -> None:
    if MODEL_PATH.exists():
        return

    data_path = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else DEMO_DATA_PATH
    if not data_path.exists():
        build_demo_dataset(data_path)

    trials = 5 if data_path == DEMO_DATA_PATH else 20
    train(data_path, trials=trials)


if __name__ == "__main__":
    ensure_model()
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
