# app.py
import os
import io
import json
from datetime import datetime, timedelta

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd

# Only import tf/keras once
import tensorflow as tf
from tensorflow.keras.models import load_model

import yfinance as yf
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- Load model once ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.keras")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Put model.keras in project root.")

model = load_model(MODEL_PATH)
model.summary()  # optional: prints model architecture to console

# --- Utility: fetch data ---
def fetch_history(ticker: str, period_days: int = 365):
    """
    Fetch historical price data for ticker using yfinance.
    Returns a pandas DataFrame indexed by date.
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=period_days)
    # yfinance sometimes uses network; ensure DNS/network is available
    df = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), progress=False)
    if df.empty:
        raise RuntimeError(f"No data fetched for {ticker}")
    return df

# --- Utility: prepare input for model ---
def prepare_input(df: pd.DataFrame, window: int = 60):
    """
    Example: create normalized sliding windows from close prices
    Adapt this to match how you trained the model in the notebook.
    """
    # Use 'Close' column
    close = df['Close'].values.astype('float32')
    # simple normalization using last window mean/std (adjust per your training)
    if len(close) < window:
        raise ValueError("Not enough historical points for prediction.")
    seq = close[-window:]
    # reshape -> (1, window, 1)
    seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-9)
    seq = seq.reshape((1, window, 1))
    return seq

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json() or {}
        ticker = payload.get("ticker", "BTC-USD")
        days = int(payload.get("history_days", 365))
        predict_steps = int(payload.get("predict_steps", 7))

        # fetch
        df = fetch_history(ticker, period_days=days)

        # prepare input (ensure window equals what your model expects)
        window = 60  # change to the window size used in training
        x = prepare_input(df, window=window)

        # predict (model output shape depends on your model)
        preds = model.predict(x)  # shape depends on model

        # Postprocess preds (this is model-specific)
        # If your model predicts normalized values, convert back using same scaler
        preds_list = preds.flatten().tolist()

        # respond with price history and preds
        resp = {
            "ticker": ticker,
            "history": {
                "dates": df.index.strftime("%Y-%m-%d").tolist()[-200:],  # trim if large
                "close": df['Close'].round(6).tolist()[-200:]
            },
            "predictions": preds_list
        }
        return jsonify(resp)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Optional: route to serve a plot image (PNG)
@app.route("/plot/<ticker>.png")
def plot_png(ticker):
    try:
        df = fetch_history(ticker, period_days=365)
        plt.figure(figsize=(8,4))
        plt.plot(df.index, df['Close'])
        plt.title(f"{ticker} Closing Price")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plt.close()
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        return f"Error: {e}", 500

if __name__ == "__main__":
    # For local dev:
    app.run(host="127.0.0.1", port=5000, debug=True)
