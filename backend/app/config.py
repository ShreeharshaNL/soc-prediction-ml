from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_PATH = MODEL_DIR / "soc_xgboost_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
SHAP_PATH = REPORT_DIR / "shap_beeswarm.png"
DEMO_DATA_PATH = DATA_DIR / "demo_soil_data.csv"
DEFAULT_DATA_PATH = DATA_DIR / "soil_data.csv"
