# Soil Organic Carbon Estimation with XGBoost

Predict soil organic carbon (SOC %) from terrain attributes, land cover, and climate normals using XGBoost Regressor. The project includes Bayesian hyperparameter optimization with Optuna, SHAP beeswarm explainability, a FastAPI backend, and a React frontend ready for Render deployment.

Student: Shreeharsha N L  
USN: 4NI23IS197  
Email: 2023is_shreeharshanl_d@nie.ac.in

## Project Structure

```text
backend/   FastAPI API, training pipeline, model artifacts
frontend/  React + Vite web app
render.yaml Render blueprint for backend and frontend
```

## Dataset

Dataset link: https://www.kaggle.com/datasets/jmohit19/soil-data

Kaggle blocks unauthenticated downloads in many environments. Download the dataset ZIP or CSV from Kaggle and place the main CSV at:

```text
backend/data/soil_data.csv
```

If your CSV has a different target column name, the training script auto-detects common SOC names such as `SOC`, `soc_percent`, `soil_organic_carbon`, `organic_carbon`, and `soc`.

With Kaggle API credentials configured, you can also run:

```powershell
cd backend
pip install kaggle
python scripts/download_kaggle_dataset.py
```

## Run Locally

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_demo_data.py
python train.py --data data/demo_soil_data.csv --trials 10
uvicorn app.main:app --reload
```

Backend URL: `http://localhost:8000`

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:5173`

Create `frontend/.env` for local API configuration if needed:

```text
VITE_API_URL=http://localhost:8000
```

## Render Deployment

1. Push this folder to GitHub.
2. In Render, choose **New +** then **Blueprint**.
3. Select the repository containing this project.
4. Render will read `render.yaml` and create:
   - `soc-xgboost-backend`
   - `soc-xgboost-frontend`
5. Add the backend URL as `VITE_API_URL` for the frontend if Render does not auto-fill it.

The backend startup command uses `scripts/render_start.py`. If no trained model exists, it trains a small demo model so the deployed API starts successfully. For real results, train with Kaggle data locally or during build by placing the CSV in `backend/data/soil_data.csv`.

## API Endpoints

- `GET /api/status` - model status, metrics, artifact availability
- `GET /api/features` - raw feature schema for the prediction form
- `POST /api/predict` - predict SOC %
- `GET /api/metrics` - model metrics from the latest training run
- `GET /api/artifacts/shap-beeswarm` - SHAP beeswarm PNG

## Training Command

```powershell
cd backend
python train.py --data data/soil_data.csv --trials 50
```

Outputs:

- `models/soc_xgboost_pipeline.joblib`
- `reports/metrics.json`
- `reports/shap_beeswarm.png`
