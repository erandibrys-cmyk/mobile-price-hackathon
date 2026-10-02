import subprocess
import sys

import pandas as pd
import streamlit as st

from charts import show_charts
from model import predict_prices, train_model
from model_ui import BINARY_COLUMNS, FEATURE_COLUMNS

# Names shown on the page. The keys stay as the CSV column names.
FIELD_NAMES = {
    "battery_power": "Battery capacity (mAh)",
    "blue": "Bluetooth",
    "clock_speed": "Clock speed (GHz)",
    "dual_sim": "Dual SIM",
    "fc": "Front camera (MP)",
    "four_g": "4G",
    "int_memory": "Internal memory (GB)",
    "m_dep": "Phone depth (cm)",
    "mobile_wt": "Phone weight (g)",
    "n_cores": "Number of processor cores",
    "pc": "Primary camera (MP)",
    "px_height": "Screen height (pixels)",
    "px_width": "Screen width (pixels)",
    "ram": "RAM (MB)",
    "sc_h": "Screen height (cm)",
    "sc_w": "Screen width (cm)",
    "talk_time": "Talk time (hours)",
    "three_g": "3G",
    "touch_screen": "Touch screen",
    "wifi": "Wi-Fi",
    "price_range": "Price range",
}

# The file has no dollar price. These four words are the only price groups.
PRICE_GROUPS = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Very high",
}

st.set_page_config(page_title="Mobile Price Classification", layout="wide")
st.title("Mobile Price Classification")

# Run Kim's file. It saves processed_data/train_cleaned.csv.
subprocess.run([sys.executable, "data_processing.py"], check=True)
clean_data = pd.read_csv("processed_data/train_cleaned.csv")

st.subheader("Cleaned data")
st.write("Rows:", len(clean_data))
st.write("There is no dollar price in this dataset. Each phone belongs to one group: Low, Medium, High, or Very high.")
st.dataframe(clean_data.rename(columns=FIELD_NAMES))

show_charts(clean_data)

st.subheader("Phone details")
st.write("Each box starts with the average of that column in the cleaned data.")

phone = {}

with st.form("phone_form"):
    for column in FEATURE_COLUMNS:
        column_average = float(clean_data[column].mean())

        if column in BINARY_COLUMNS:
            # 0 or 1. Pick 1 when the average is at least half.
            selected = 0
            if column_average >= 0.5:
                selected = 1
            phone[column] = st.selectbox(
                FIELD_NAMES[column],
                options=[0, 1],
                index=selected,
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
        else:
            phone[column] = st.number_input(
                FIELD_NAMES[column],
                min_value=float(clean_data[column].min()),
                max_value=float(clean_data[column].max()),
                value=column_average,
            )

    submitted = st.form_submit_button("Predict")

# Train once, then reuse the same model when the button is clicked again.
@st.cache_resource
def get_trained_model():
    features = pd.read_csv("processed_data/X.csv")
    answers = pd.read_csv("processed_data/y.csv")["price_range"]
    return train_model(features, answers)

trained_model, metrics = get_trained_model()

if submitted:
    # Richard owns this prediction. He edits model.py, not this button.
    features = pd.read_csv("processed_data/X.csv")
    one_phone = pd.DataFrame([phone])
    one_phone = one_phone[features.columns]
    label = int(predict_prices(trained_model, one_phone)[0])
    st.success("Predicted price group: " + PRICE_GROUPS[label])
    st.write("This is not a dollar amount.")
    correct_percent = round(metrics["accuracy"] * 100, 1)
    st.write("On phones the model had not seen before, it picked the right group", correct_percent, "% of the time.")
