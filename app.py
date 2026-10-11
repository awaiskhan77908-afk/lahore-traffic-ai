
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Lahore Traffic AI",
    page_icon="🚦",
    layout="wide"
)

st.title("Lahore Traffic AI")
st.write("AI-Based Traffic Congestion Prediction System")

model_path = Path(__file__).parent / "traffic_model.joblib"

if not model_path.exists():
    st.error("traffic_model.joblib file GitHub par nahi mili.")
    st.info("Model file ko app.py ke saath same folder mein upload karein.")
    st.stop()

bundle = joblib.load(model_path)
model = bundle["model"]
road_mapping = bundle["road_mapping"]

st.success("AI model loaded successfully!")

roads = list(road_mapping.keys())
selected_road = st.selectbox("Select Road ID", roads)

prediction_time = st.datetime_input(
    "Select prediction date and time",
    value=datetime.now()
)

previous_speed = st.number_input(
    "Latest observed speed (km/h)",
    min_value=0.0,
    max_value=150.0,
    value=30.0
)

if st.button("Predict Traffic"):
    input_data = pd.DataFrame([{
        "hour": prediction_time.hour,
        "day_of_week": prediction_time.weekday(),
        "road_code": road_mapping[selected_road],
        "previous_speed": previous_speed
    }])

    predicted_speed = max(
        0.0, float(model.predict(input_data)[0])
    )

    if predicted_speed < 20:
        congestion = "High Congestion"
    elif predicted_speed < 40:
        congestion = "Medium Congestion"
    else:
        congestion = "Low Congestion"

    st.metric(
        "Predicted Next Speed",
        f"{predicted_speed:.2f} km/h"
    )
    st.subheader("Congestion Level: " + congestion)
    st.caption(
        "This is a model-based estimate, not live traffic data."
    )
