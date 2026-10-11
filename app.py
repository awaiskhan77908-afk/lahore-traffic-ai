import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Lahore Traffic AI", page_icon="🚦")

st.title("Lahore Traffic AI")
st.subheader("Traffic Congestion Prediction System")

model_path = Path(__file__).parent /  "traffic_model (1).joblib"

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

st.title("🚦 Lahore Traffic AI")
st.subheader("Traffic Congestion Prediction & Road Comparison")

model_path = Path(__file__).parent / "traffic_model (1).joblib"

if not model_path.exists():
    st.error("Model file missing: " + model_path.name)
    st.stop()

try:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    road_mapping = bundle["road_mapping"]
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.success("AI model loaded successfully!")

roads = list(road_mapping.keys())

if not roads:
    st.error("No roads found in the model.")
    st.stop()

st.header("1. Predict Traffic")

road = st.selectbox("Select road", roads)

prediction_clock = st.time_input(
    "Prediction time",
    value=datetime.now().time().replace(second=0, microsecond=0)
)



speed = st.number_input(
    "Latest observed speed (km/h)",
    min_value=0.0,
    max_value=150.0,
    value=30.0,
    key="latest_observed_speed"
)



if st.button("Predict Traffic", key="predict_traffic_button"):

    input_data = pd.DataFrame([{
        "hour": prediction_time.hour,
        "day_of_week": prediction_time.weekday(),
        "road_code": road_mapping[road],
        "previous_speed": speed
    }])

    try:
        predicted_speed = max(
            0.0, float(model.predict(input_data)[0])
        )

        if predicted_speed < 20:
            level = "High Congestion"
        elif predicted_speed < 40:
            level = "Medium Congestion"
        else:
            level = "Low Congestion"

        st.metric("Predicted Speed", f"{predicted_speed:.2f} km/h")
        st.subheader("Traffic Level: " + level)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.header("2. Compare Roads")

st.write(
    "Historical average speed comparison. "
    "This does not represent live traffic."
)

csv_path = Path(__file__).parent / "lahore_traffic_demo.csv"

if csv_path.exists():
    try:
        df = pd.read_csv(csv_path)

        if {"road_id", "average_speed_kmh"}.issubset(df.columns):
            comparison = (
                df.groupby("road_id")["average_speed_kmh"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )

            comparison.columns = [
                "Road ID", "Average Speed (km/h)"
            ]

            st.dataframe(comparison, use_container_width=True)

            if not comparison.empty:
                best = comparison.iloc[0]
                st.success(
                    f"Highest historical average speed: "
                    f"{best['Road ID']} "
                    f"({best['Average Speed (km/h)']:.2f} km/h)"
                )

            st.bar_chart(
                comparison.set_index("Road ID")[
                    "Average Speed (km/h)"
                ]
            )
        else:
            st.warning(
                "CSV columns road_id and average_speed_kmh "
                "were not found. Check your dataset column names."
            )

    except Exception as e:
        st.error(f"Could not read traffic dataset: {e}")
else:
    st.warning(
        "Dataset CSV is missing. Upload "
        "lahore_traffic_demo.csv beside app.py to enable comparison."
    )

st.caption(
    "Academic prototype: predictions and recommendations "
    "depend on the available dataset, not live traffic feeds."
)
