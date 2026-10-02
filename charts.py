"""Charts for the mobile price page.

Erandi can change this file. The page calls show_charts with the cleaned table.
"""

import streamlit as st


def show_charts(clean_data):
    st.subheader("Charts")
    chart_one, chart_two, chart_three = st.columns(3)

    with chart_one:
        st.write("Phones in each price range")
        phone_counts = clean_data["price_range"].value_counts().sort_index()
        phone_counts.index = ["Low", "Medium", "High", "Very high"]
        st.bar_chart(phone_counts)

    with chart_two:
        st.write("Average RAM by price range")
        average_ram = clean_data.groupby("price_range")["ram"].mean()
        average_ram.index = ["Low", "Medium", "High", "Very high"]
        st.bar_chart(average_ram)

    with chart_three:
        st.write("Average battery power by price range")
        average_battery = clean_data.groupby("price_range")["battery_power"].mean()
        average_battery.index = ["Low", "Medium", "High", "Very high"]
        st.bar_chart(average_battery)
