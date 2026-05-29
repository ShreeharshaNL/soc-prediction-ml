import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATASET = "jmohit19/soil-data"


def main() -> None:
    if not shutil.which("kaggle"):
        raise SystemExit(
            "Kaggle CLI is not installed or not on PATH. Install it with `pip install kaggle`, "
            "then place kaggle.json in your Kaggle config folder."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(DATA_DIR), "--unzip"],
        check=True,
    )
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise SystemExit("Dataset downloaded, but no CSV file was found in backend/data.")

    target = DATA_DIR / "soil_data.csv"
    if csv_files[0] != target:
        shutil.copy2(csv_files[0], target)
    print(f"Dataset ready: {target}")


if __name__ == "__main__":
    main()
