"""
Sydney Housing Price Predictor — Streamlit deployment (SIT720 8.1 Distinction Task, Part 6)

Run with:
    streamlit run app.py

Loads the trained model pipeline exported by the project notebook
(SIT720_8.1_Distinction_Task.ipynb) and lets a user enter property features to
get an instant predicted sale price, with a short explanation of the prediction.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "best_model.joblib"
META_PATH = APP_DIR / "model_meta.json"

st.set_page_config(
    page_title="Sydney Housing Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Load model + metadata (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


if not MODEL_PATH.exists():
    st.error(
        "No trained model found (`best_model.joblib`). Run the project notebook "
        "`SIT720_8.1_Distinction_Task.ipynb` end-to-end first — its final cell "
        "saves the model into this `app/` folder."
    )
    st.stop()

model, meta = load_model()

SUBURBS = ["Mosman", "Parramatta", "Mount Druitt"]
PROPERTY_TYPES = ["House", "Unit", "Townhouse"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 Sydney Housing Price Predictor")
st.caption(
    "SIT720 Machine Learning — Mini Project decision-support prototype. "
    f"Powered by a **{meta['best_model_name']}** model "
    f"(held-out test R² = {meta['test_r2']:.2f}, MAE ≈ ${meta['test_mae']:,.0f})."
)
st.info(
    "⚠️ This model was trained on **115 real sold properties** manually collected from "
    "domain.com.au's sold-listings pages for Mosman, Parramatta and Mount Druitt (see the "
    "project report, Part 1). It only uses the fields genuinely available from a sold-listing "
    "search result (no condition, view or agent-description text) — treat predictions as a "
    "decision-support starting point, not a substitute for a professional valuation.",
    icon="ℹ️",
)

tab_manual, tab_batch = st.tabs(["🔢 Enter a single property", "📄 Upload a CSV (batch)"])

# ---------------------------------------------------------------------------
# Tab 1: manual single-property entry
# ---------------------------------------------------------------------------
with tab_manual:
    st.subheader("Property details")

    col1, col2 = st.columns(2)
    with col1:
        suburb = st.selectbox("Suburb", SUBURBS)
        property_type = st.selectbox("Property type", PROPERTY_TYPES)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=12, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=12, value=2)

    with col2:
        car_spaces = st.number_input(
            "Car spaces (leave at 0 if not listed)", min_value=0, max_value=10, value=1
        )
        land_size_sqm = st.number_input(
            "Land size (sqm) — leave at 0 for a unit (no individual land size)",
            min_value=0.0, max_value=2000.0, value=400.0, step=10.0,
        )
        sale_quarter = st.selectbox("Sale quarter (for seasonality)", ["1", "2", "3", "4"], index=2)

    if st.button("Predict sale price", type="primary"):
        bed_bath_ratio = bedrooms / max(bathrooms, 1)

        row = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "car_spaces": car_spaces if car_spaces > 0 else np.nan,
            "land_size_sqm": np.nan if (property_type == "Unit" or land_size_sqm == 0) else land_size_sqm,
            "bed_bath_ratio": bed_bath_ratio,
            "suburb": suburb,
            "property_type": property_type,
            "sale_quarter": sale_quarter,
        }])

        pred_log = model.predict(row)[0]
        pred_price = float(np.exp(pred_log)) if meta.get("target_transform") == "log" else float(pred_log)

        st.success(f"### Predicted sale price: **${pred_price:,.0f}**")
        st.caption(
            f"Approximate range (± typical model error): "
            f"${max(0, pred_price - meta['test_mae']):,.0f} – ${pred_price + meta['test_mae']:,.0f}"
        )

        with st.expander("Show the exact feature row sent to the model"):
            st.dataframe(row.T.rename(columns={0: "value"}))

# ---------------------------------------------------------------------------
# Tab 2: batch CSV upload
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.write(
        "Upload a CSV with the same columns used to train the model "
        f"({', '.join(meta['numeric_features'] + meta['categorical_features'])}). "
        "Missing numeric columns will be median/most-frequent imputed automatically by the pipeline."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            needed_cols = meta["numeric_features"] + meta["categorical_features"]
            missing_cols = [c for c in needed_cols if c not in batch_df.columns]
            if missing_cols:
                st.error(f"CSV is missing required columns: {missing_cols}")
            else:
                preds_log = model.predict(batch_df[needed_cols])
                preds = np.exp(preds_log) if meta.get("target_transform") == "log" else preds_log
                batch_df["predicted_sale_price"] = preds.round(0)
                st.dataframe(batch_df)
                st.download_button(
                    "Download predictions as CSV",
                    batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="predicted_prices.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Could not process file: {e}")

st.divider()
st.caption(
    "Built with Streamlit for the SIT720 Machine Learning unit (Deakin University). "
    "Model trained in `SIT720_8.1_Distinction_Task.ipynb`; see the project report for full methodology."
)
