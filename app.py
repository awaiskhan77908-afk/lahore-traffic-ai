import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Lahore Traffic AI", page_icon="🚦")

st.title("Lahore Traffic AI")
st.subheader("Traffic Congestion Prediction System")

model_path = Path(__file__).parent / "model_path = Path(__file__).parent / "traffic_model (1).joblib""

if not model_path.exists():
    st.error("traffic_model.joblib file missing hai.")
    st.stop()

bundle = joblib.load(model_path)
model = bundle["model"]
road_mapping = bundle["road_mapping"]

road_ids = list(road_mapping.keys())
road = st.selectbox("Select Road ID", road_ids)

prediction_time = st.datetime_input(
    "Prediction date and time",
    value=datetime.now()
)

speed = st.number_input(
    "Latest observed speed (km/h)",
    min_value=0.0,
    max_value=150.0,
    value=30.0
)

if st.button("Predict Traffic"):
    data = pd.DataFrame([{
        "hour": prediction_time.hour,
        "day_of_week": prediction_time.weekday(),
        "road_code": road_mapping[road],
        "previous_speed": speed
    }])

    predicted_speed = max(0.0, float(model.predict(data)[0]))

    if predicted_speed < 20:
        level = "High Congestion"
    elif predicted_speed < 40:
        level = "Medium Congestion"
    else:
        level = "Low Congestion"

    st.metric("Predicted Speed", f"{predicted_speed:.2f} km/h")
    st.subheader(level)
    st.caption("Model estimate, not live traffic data.")
