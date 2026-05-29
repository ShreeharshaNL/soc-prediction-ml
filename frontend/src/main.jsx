import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, BarChart3, FlaskConical, Leaf, RefreshCcw, Send } from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const fallbackFeatures = [
  { name: "elevation_m", kind: "number", example: 620 },
  { name: "slope_deg", kind: "number", example: 7.2 },
  { name: "annual_rainfall_mm", kind: "number", example: 930 },
  { name: "mean_temperature_c", kind: "number", example: 23.5 },
  { name: "ndvi", kind: "number", example: 0.52 },
  { name: "clay_percent", kind: "number", example: 32 },
  { name: "soil_ph", kind: "number", example: 6.7 },
  { name: "land_cover", kind: "category", example: "cropland", categories: ["cropland", "forest", "grassland", "shrubland", "built-up"] }
];

function App() {
  const [status, setStatus] = useState(null);
  const [features, setFeatures] = useState(fallbackFeatures);
  const [form, setForm] = useState(() => seedForm(fallbackFeatures));
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [imageVersion, setImageVersion] = useState(Date.now());

  useEffect(() => {
    loadApiState();
  }, []);

  async function loadApiState() {
    setError("");
    try {
      const [statusResponse, featureResponse] = await Promise.all([
        fetch(`${API_URL}/api/status`),
        fetch(`${API_URL}/api/features`)
      ]);
      if (statusResponse.ok) {
        setStatus(await statusResponse.json());
      }
      if (featureResponse.ok) {
        const remoteFeatures = await featureResponse.json();
        if (remoteFeatures.length) {
          setFeatures(remoteFeatures);
          setForm(seedForm(remoteFeatures));
        }
      }
      setImageVersion(Date.now());
    } catch {
      setError("Backend is not reachable. Start FastAPI locally or set VITE_API_URL on Render.");
    }
  }

  async function handlePredict(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);
    try {
      const payload = {};
      for (const feature of features) {
        const value = form[feature.name];
        payload[feature.name] = feature.kind === "number" ? Number(value) : value;
      }
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: payload })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }
      setPrediction(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const metrics = status?.metrics || {};
  const numericSummary = useMemo(
    () => [
      ["RMSE", metrics.rmse],
      ["MAE", metrics.mae],
      ["R2", metrics.r2],
      ["Rows", metrics.rows]
    ],
    [metrics]
  );

  return (
    <main className="app-shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Regression + Hyperparameter Tuning</p>
          <h1>Soil Organic Carbon Estimation</h1>
          <p className="subtitle">XGBoost regressor with Optuna tuning and SHAP explainability for SOC percentage prediction.</p>
        </div>
        <div className="identity">
          <strong>4NI23IS197</strong>
          <span>Shreeharsha N L</span>
        </div>
      </section>

      <section className="status-strip">
        <StatusBadge icon={<Activity size={18} />} label="Backend" value={status?.model_ready ? "Model ready" : "Waiting"} />
        <StatusBadge icon={<FlaskConical size={18} />} label="Model" value="XGBoost" />
        <StatusBadge icon={<BarChart3 size={18} />} label="Optuna CV RMSE" value={metrics.optuna_best_cv_rmse ?? "N/A"} />
      </section>

      <section className="workspace">
        <form className="predictor" onSubmit={handlePredict}>
          <div className="panel-heading">
            <div>
              <h2>Prediction Inputs</h2>
              <p>Enter terrain, climate, soil, and land cover values.</p>
            </div>
            <button type="button" className="icon-button" onClick={loadApiState} title="Refresh model state">
              <RefreshCcw size={18} />
            </button>
          </div>

          <div className="field-grid">
            {features.map((feature) => (
              <label key={feature.name} className="field">
                <span>{labelize(feature.name)}</span>
                {feature.kind === "category" ? (
                  <select value={form[feature.name] ?? ""} onChange={(event) => setForm({ ...form, [feature.name]: event.target.value })}>
                    {(feature.categories?.length ? feature.categories : [feature.example]).map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="number"
                    step="any"
                    value={form[feature.name] ?? ""}
                    onChange={(event) => setForm({ ...form, [feature.name]: event.target.value })}
                  />
                )}
              </label>
            ))}
          </div>

          <button className="primary-button" type="submit" disabled={loading}>
            <Send size={18} />
            {loading ? "Predicting" : "Predict SOC"}
          </button>

          {error && <div className="message error">{error}</div>}
          {prediction && (
            <div className="result">
              <Leaf size={28} />
              <div>
                <span>Predicted SOC</span>
                <strong>{prediction.soc_percent.toFixed(3)}%</strong>
              </div>
            </div>
          )}
        </form>

        <aside className="insights">
          <div className="panel-heading">
            <div>
              <h2>Model Metrics</h2>
              <p>Latest training run summary.</p>
            </div>
          </div>
          <div className="metric-grid">
            {numericSummary.map(([label, value]) => (
              <div className="metric" key={label}>
                <span>{label}</span>
                <strong>{value ?? "N/A"}</strong>
              </div>
            ))}
          </div>
          <div className="shap-panel">
            <h2>SHAP Beeswarm</h2>
            <img
              src={`${API_URL}/api/artifacts/shap-beeswarm?v=${imageVersion}`}
              alt="SHAP beeswarm variable importance plot"
              onError={(event) => {
                event.currentTarget.style.display = "none";
              }}
            />
          </div>
        </aside>
      </section>
    </main>
  );
}

function StatusBadge({ icon, label, value }) {
  return (
    <div className="status-badge">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function seedForm(items) {
  return Object.fromEntries(items.map((feature) => [feature.name, feature.example ?? ""]));
}

function labelize(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (match) => match.toUpperCase());
}

createRoot(document.getElementById("root")).render(<App />);
