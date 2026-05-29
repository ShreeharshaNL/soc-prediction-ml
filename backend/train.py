import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from app.config import METRICS_PATH, MODEL_PATH, REPORT_DIR, SHAP_PATH

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_CANDIDATES = {
    "soc",
    "soc_percent",
    "soc_%",
    "soc_pct",
    "soil_organic_carbon",
    "soil_organic_carbon_percent",
    "organic_carbon",
    "organic_carbon_percent",
    "carbon",
}


def find_target_column(frame: pd.DataFrame, explicit_target: str | None = None) -> str:
    if explicit_target:
        if explicit_target not in frame.columns:
            raise ValueError(f"Target column '{explicit_target}' was not found in the dataset.")
        return explicit_target

    normalized = {column.strip().lower().replace(" ", "_").replace("%", "percent"): column for column in frame.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    soc_like = [column for column in frame.columns if "soc" in column.lower()]
    if soc_like:
        return soc_like[0]

    raise ValueError(
        "Could not detect SOC target column. Pass --target with the target column name, "
        "for example: python train.py --data data/soil_data.csv --target SOC_percent"
    )


def clean_frame(path: Path, target: str | None) -> tuple[pd.DataFrame, pd.Series, str]:
    frame = pd.read_csv(path)
    frame = frame.dropna(axis=1, how="all").drop_duplicates()
    target_column = find_target_column(frame, target)
    y = pd.to_numeric(frame[target_column], errors="coerce")
    X = frame.drop(columns=[target_column])
    valid_rows = y.notna()
    X = X.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    for column in X.columns:
        numeric = pd.to_numeric(X[column], errors="coerce")
        if numeric.notna().mean() >= 0.8:
            X[column] = numeric
        else:
            X[column] = X[column].astype("string")

    return X, y, target_column


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_features, categorical_features


def objective_factory(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: ColumnTransformer):
    cv = KFold(n_splits=3, shuffle=True, random_state=197)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=197,
            n_jobs=-1,
            tree_method="hist",
            **params,
        )
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        return float(-scores.mean())

    return objective


def make_feature_schema(X: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in numeric_features:
        series = pd.to_numeric(X[column], errors="coerce")
        schema.append(
            {
                "name": column,
                "kind": "number",
                "example": round(float(series.median()), 4) if series.notna().any() else 0,
                "min": round(float(series.min()), 4) if series.notna().any() else None,
                "max": round(float(series.max()), 4) if series.notna().any() else None,
            }
        )
    for column in categorical_features:
        values = X[column].dropna().astype(str)
        categories = sorted(values.unique().tolist())[:30]
        schema.append(
            {
                "name": column,
                "kind": "category",
                "example": categories[0] if categories else "",
                "categories": categories,
            }
        )
    return schema


def save_shap_plot(pipeline: Pipeline, X_sample_raw: pd.DataFrame) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    X_transformed = preprocessor.transform(X_sample_raw)
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_transformed)

    SHAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(values, X_transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(SHAP_PATH, dpi=180, bbox_inches="tight")
    plt.close()


def train(data_path: Path, target: str | None = None, trials: int = 30) -> dict[str, Any]:
    X, y, target_column = clean_frame(data_path, target)
    if len(X) < 20:
        raise ValueError("The dataset needs at least 20 valid rows for train/test splitting.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=197)
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    study = optuna.create_study(direction="minimize", study_name="soc_xgboost_rmse")
    study.optimize(objective_factory(X_train, y_train, preprocessor), n_trials=trials, show_progress_bar=False)

    best_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=197,
        n_jobs=-1,
        tree_method="hist",
        **study.best_params,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    metrics = {
        "dataset": str(data_path),
        "target_column": target_column,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "rmse": round(rmse, 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "best_params": study.best_params,
        "optuna_best_cv_rmse": round(float(study.best_value), 4),
    }

    sample_size = min(250, len(X_test))
    save_shap_plot(pipeline, X_test.sample(sample_size, random_state=197))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "raw_features": X.columns.tolist(),
        "feature_schema": make_feature_schema(X, numeric_features, categorical_features),
        "target_column": target_column,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SOC XGBoost model with Optuna tuning and SHAP output.")
    parser.add_argument("--data", type=Path, default=Path("data/soil_data.csv"), help="Path to CSV dataset.")
    parser.add_argument("--target", default=None, help="Target column name. Optional if SOC column can be detected.")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train(args.data, target=args.target, trials=args.trials)
    print(json.dumps(metrics, indent=2))
