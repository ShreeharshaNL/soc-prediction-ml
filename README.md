# Soil Organic Carbon Estimation with XGBoost

Predict soil organic carbon (SOC %) from terrain attributes, land cover, and climate normals using an XGBoost Regressor. The backend supports Bayesian hyperparameter optimisation with Optuna and generates a SHAP beeswarm plot for feature importance.

Student: Shreeharsha N L  
USN: 4NI23IS197  
Email: 2023is_shreeharshanl_d@nie.ac.in  
Project Type: Regression + Hyperparameter Tuning

## Dataset

Kaggle dataset:

```text
https://www.kaggle.com/datasets/jmohit19/soil-data
```

Download the dataset manually from Kaggle and place the main CSV file here:

```text
backend/data/soil_data.csv
```

If the target column has a different name, pass it during training:

```powershell
python train.py --data data/soil_data.csv --target YOUR_TARGET_COLUMN --trials 50
```

Common SOC target names are auto-detected, including `SOC`, `soc_percent`, `SOC_percent`, `soil_organic_carbon`, and `organic_carbon`.

## Project Structure

```text
soil-organic-carbon-xgboost/
  backend/      FastAPI backend, ML training pipeline, model files
  frontend/     React + Vite frontend
  render.yaml   Optional Render blueprint for backend + frontend
```

## Local Backend Setup

Open PowerShell:

```powershell
cd "C:\Users\shree\D Drive\shnl\random proj\soil-organic-carbon-xgboost\backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For real Kaggle data:

```powershell
python train.py --data data/soil_data.csv --trials 50
```

For demo testing without Kaggle data:

```powershell
python scripts/generate_demo_data.py
python train.py --data data/demo_soil_data.csv --trials 10
```

Run the backend:

```powershell
uvicorn app.main:app --reload
```

Backend local URL:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

## Local Frontend Setup

Open another PowerShell window:

```powershell
cd "C:\Users\shree\D Drive\shnl\random proj\soil-organic-carbon-xgboost\frontend"
npm install
```

Create `.env`:

```powershell
copy .env.example .env
```

Make sure `.env` contains:

```text
VITE_API_URL=http://localhost:8000
```

Run frontend locally:

```powershell
npm run dev
```

Frontend local URL:

```text
http://localhost:5173
```

## Frontend Build Command

Use this before deploying to Vercel:

```powershell
cd "C:\Users\shree\D Drive\shnl\random proj\soil-organic-carbon-xgboost\frontend"
npm install
npm run build
```

Build output folder:

```text
frontend/dist
```

## Deploy Frontend on Vercel

This project is configured so Vercel deploys only the React frontend from the repository root. The Python ML backend is ignored by Vercel through `.vercelignore` because it can exceed Vercel size limits.

Recommended Vercel settings:

```text
Framework Preset: Vite
Root Directory: .
Install Command: cd frontend && npm install
Build Command: cd frontend && npm run build
Output Directory: frontend/dist
```

Add this environment variable in Vercel:

```text
VITE_API_URL=https://YOUR-BACKEND-URL.onrender.com
```

Manual CLI deployment:

```powershell
cd "C:\Users\shree\D Drive\shnl\random proj\soil-organic-carbon-xgboost"
npx vercel login
npx vercel --prod
```

If you deploy from the Vercel dashboard, import the full repository but keep the root directory as `.`. Do not select the `backend` folder for Vercel.

## Deploy Backend on Render

Recommended Render settings:

```text
Service Type: Web Service
Root Directory: backend
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: python scripts/render_start.py
```

Environment variables:

```text
PYTHON_VERSION=3.11.9
CORS_ORIGINS=*
```

The backend starts on Render using the `PORT` variable automatically.

If `backend/models/soc_xgboost_pipeline.joblib` does not exist, `scripts/render_start.py` trains a small demo model so the API can start. For real predictions, train using `backend/data/soil_data.csv`.

## API Endpoints

```text
GET  /api/status
GET  /api/features
POST /api/predict
GET  /api/metrics
GET  /api/artifacts/shap-beeswarm
```

Prediction request example:

```json
{
  "features": {
    "elevation_m": 620,
    "slope_deg": 7.2,
    "annual_rainfall_mm": 930,
    "mean_temperature_c": 23.5,
    "ndvi": 0.52,
    "clay_percent": 32,
    "soil_ph": 6.7,
    "land_cover": "cropland"
  }
}
```

## Generated Model Files

After training, these files are created:

```text
backend/models/soc_xgboost_pipeline.joblib
backend/reports/metrics.json
backend/reports/shap_beeswarm.png
```

## Notes

Vercel is best for the React frontend. The Python ML backend uses heavy packages such as XGBoost, Optuna, and SHAP, so Render is the better deployment target for the backend.
