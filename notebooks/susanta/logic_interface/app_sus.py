import streamlit as st
import requests
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Grid Intelligence", layout="centered")

st.title("⚡ Grid Intelligence Forecast Dashboard")

st.markdown("Forecast electricity prices using ML model")

# -----------------------------
# USER INPUT
# -----------------------------
steps = st.slider("Prediction steps", 1, 10, 3)

run = st.button("Run Prediction")

# -----------------------------
# CALL API
# -----------------------------
if run:
    try:
        response = requests.get(API_URL, params={"steps": steps})
        data = response.json()

        preds = data["prediction"]
        mae = data["mae"]

        # -----------------------------
        # METRICS
        # -----------------------------
        st.subheader("📊 Model Performance")
        st.metric("MAE", round(mae, 4))

        # -----------------------------
        # PREDICTION DATAFRAME
        # -----------------------------
        df_plot = pd.DataFrame({
            "Step": list(range(1, len(preds) + 1)),
            "Predicted Price": preds
        })

        df_plot = df_plot.set_index("Step")

        # -----------------------------
        # PLOT
        # -----------------------------
        st.subheader("🔮 Forecast")
        st.line_chart(df_plot)

        # -----------------------------
        # RAW OUTPUT
        # -----------------------------
        st.subheader("Raw Output")
        st.write(data)

    except Exception as e:
        st.error(f"Error: {e}")
