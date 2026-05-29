from pathlib import Path

import numpy as np
import pandas as pd


def build_demo_dataset(path: Path, rows: int = 450) -> None:
    rng = np.random.default_rng(197)
    elevation = rng.normal(620, 160, rows).clip(120, 1600)
    slope = rng.gamma(2.0, 4.0, rows).clip(0, 45)
    rainfall = rng.normal(930, 210, rows).clip(350, 1800)
    temperature = rng.normal(23.5, 3.0, rows).clip(12, 35)
    ndvi = rng.normal(0.52, 0.17, rows).clip(0.05, 0.95)
    clay = rng.normal(32, 10, rows).clip(5, 65)
    ph = rng.normal(6.7, 0.55, rows).clip(4.8, 8.4)
    land_cover = rng.choice(["cropland", "forest", "grassland", "shrubland", "built-up"], rows)

    cover_effect = {
        "forest": 0.75,
        "grassland": 0.45,
        "shrubland": 0.18,
        "cropland": -0.05,
        "built-up": -0.35,
    }
    soc = (
        0.55
        + 0.0012 * rainfall
        - 0.030 * temperature
        + 1.35 * ndvi
        + 0.018 * clay
        - 0.020 * slope
        - 0.060 * (ph - 6.7) ** 2
        + np.vectorize(cover_effect.get)(land_cover)
        + rng.normal(0, 0.22, rows)
    ).clip(0.15, 8.0)

    frame = pd.DataFrame(
        {
            "elevation_m": elevation.round(2),
            "slope_deg": slope.round(2),
            "annual_rainfall_mm": rainfall.round(2),
            "mean_temperature_c": temperature.round(2),
            "ndvi": ndvi.round(3),
            "clay_percent": clay.round(2),
            "soil_ph": ph.round(2),
            "land_cover": land_cover,
            "SOC_percent": soc.round(3),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    build_demo_dataset(Path(__file__).resolve().parents[1] / "data" / "demo_soil_data.csv")
    print("Demo dataset written to backend/data/demo_soil_data.csv")
