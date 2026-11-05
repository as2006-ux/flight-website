"""Flight delay project backend: serves recorded results and a NumPy least-squares predictor."""
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE = Path(__file__).parent
app = Flask(__name__, static_folder=str(BASE.parent / "frontend"), static_url_path="")
RESULTS = json.loads((BASE / "results.json").read_text())
COLS = ["MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_DEPARTURE", "DISTANCE", "ARRIVAL_DELAY"]


def features(month, day, dow, distance, hour):
    m, d, h = (np.asarray(v, float) for v in (month, dow, hour))
    return np.column_stack([m, np.asarray(day, float), d, np.asarray(distance, float), h,
        np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), np.sin(2*np.pi*m/12),
        np.cos(2*np.pi*m/12), np.sin(2*np.pi*d/7), np.cos(2*np.pi*d/7)])


def train():
    """Least-squares fit on scheduled-only features, as in Unit 2. Needs backend/data/flights.csv."""
    path = Path(os.environ.get("FLIGHTS_CSV", BASE / "data" / "flights.csv"))
    if not path.exists():
        return None
    df = pd.read_csv(path, usecols=COLS, low_memory=False).replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sample(n=min(len(df), 1_000_000), random_state=42)
    hour = (df["SCHEDULED_DEPARTURE"].astype(int) // 100).clip(0, 23)
    X = features(df.MONTH, df.DAY, df.DAY_OF_WEEK, df.DISTANCE, hour)
    y = df["ARRIVAL_DELAY"].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.linalg.solve(Xs.T @ Xs, Xs.T @ y)
    return {"w": w, "mu": mu, "sd": sd, "rows": len(df)}


MODEL = train()


@app.get("/api/health")
def health():
    return jsonify(status="ok", predictor_ready=MODEL is not None)


@app.get("/api/results")
def results():
    return jsonify(RESULTS)


@app.post("/api/predict")
def predict():
    if MODEL is None:
        return jsonify(error="Predictor not trained. Put flights.csv in backend/data/ and restart."), 503
    body = request.get_json(silent=True) or {}
    ranges = {"month": (1, 12), "day": (1, 31), "day_of_week": (1, 7), "hour": (0, 23), "distance": (1, 5000)}
    vals = {}
    for key, (lo, hi) in ranges.items():
        try:
            v = float(body[key])
        except (KeyError, TypeError, ValueError):
            return jsonify(error=f"'{key}' is required and must be a number."), 400
        if not lo <= v <= hi:
            return jsonify(error=f"'{key}' must be between {lo} and {hi}."), 400
        vals[key] = v
    x = features([vals["month"]], [vals["day"]], [vals["day_of_week"]], [vals["distance"]], [vals["hour"]])
    xs = np.concatenate([[1.0], ((x - MODEL["mu"]) / MODEL["sd"])[0]])
    minutes = float(xs @ MODEL["w"])
    return jsonify(predicted_arrival_delay_minutes=round(minutes, 1), trained_on_rows=MODEL["rows"],
        caveat="Least-squares R² was about 0.01 in Unit 2, so this estimate is weak and for demonstration only.")


@app.get("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
