"""Prediction contract for the mobile price app.

The Streamlit page imports this module only. It does not import
data_processing.py, so the interface can run while that file is still
being written.

Richard replaces the body of predict_price_range with the trained model.
The function name, the argument, and the return values stay the same.
"""

FEATURE_COLUMNS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]

BINARY_COLUMNS = [
    "blue",
    "dual_sim",
    "four_g",
    "three_g",
    "touch_screen",
    "wifi",
]


def predict_price_range(phone):
    """Return 0 low, 1 medium, 2 high, or 3 very high.

    phone is a dictionary whose keys are FEATURE_COLUMNS.
    price_range must not be one of those keys.
    """
    row = [phone[column] for column in FEATURE_COLUMNS]
    if len(row) != len(FEATURE_COLUMNS):
        raise ValueError("Phone row is incomplete.")
    # Temporary answer so the page can be demonstrated before training.
    return 1
