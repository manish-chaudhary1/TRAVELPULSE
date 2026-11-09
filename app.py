
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="TravelPulse",
    page_icon="✈️",
    layout="wide"
)

# Project title
st.title("✈️ TravelPulse")
st.subheader("AI-Powered Travel Inventory Intelligence & Optimization")

st.markdown("""
TravelPulse helps travel businesses forecast demand,
identify inventory opportunities, optimize procurement,
and evaluate financial risk.
""")

# Load data safely
@st.cache_data
def load_data():

    files = {
        "forecast": "outputs/forecast_results.csv",
        "opportunities": "outputs/opportunity_results.csv",
        "optimizer": "outputs/optimal_procurement_plan.csv",
        "scenarios": "outputs/scenario_results.csv"
    }

    data = {}

    for key, path in files.items():
        if Path(path).exists():
            data[key] = pd.read_csv(path)
        else:
            data[key] = pd.DataFrame()

    return data


data = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Overview",
        "Demand Forecasting",
        "Opportunity Detection",
        "Inventory Optimization",
        "Scenario Analysis"
    ]
)

# OVERVIEW
if page == "Overview":

    st.header("Business Overview")

    forecast = data["forecast"]
    optimizer = data["optimizer"]
    scenarios = data["scenarios"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Hotels Analyzed",
        forecast["hotel_id"].nunique()
        if not forecast.empty else 0
    )

    col2.metric(
        "Total Predicted Demand",
        round(forecast["predicted_demand"].sum(), 0)
        if not forecast.empty else 0
    )

    col3.metric(
        "Rooms to Purchase",
        round(optimizer["rooms_to_purchase"].sum(), 0)
        if not optimizer.empty else 0
    )

    st.divider()

    st.subheader("Project Pipeline")

    st.markdown("""
    1. Historical booking data
    2. Demand forecasting
    3. Opportunity detection
    4. Inventory procurement optimization
    5. Scenario and risk analysis
    """)

# DEMAND FORECASTING
elif page == "Demand Forecasting":

    st.header("Demand Forecasting")

    df = data["forecast"]

    if df.empty:
        st.warning("Forecast results not found.")
    else:

        st.write("Predicted versus actual demand")

        fig = px.line(
            df,
            x="travel_date",
            y=["predicted_demand", "actual_demand"],
            title="Demand Forecast vs Actual Demand"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Data")

        st.dataframe(df, use_container_width=True)

# OPPORTUNITY DETECTION
elif page == "Opportunity Detection":

    st.header("Inventory Opportunities")

    df = data["opportunities"]

    if df.empty:
        st.warning("Opportunity results not found.")
    else:

        if "opportunity_level" in df.columns:

            fig = px.histogram(
                df,
                x="opportunity_level",
                title="Opportunity Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

# INVENTORY OPTIMIZATION
elif page == "Inventory Optimization":

    st.header("Optimal Procurement Plan")

    df = data["optimizer"]

    if df.empty:
        st.warning("Optimization results not found.")
    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Procurement",
            round(df["procurement_amount"].sum(), 2)
        )

        col2.metric(
            "Expected Profit",
            round(df["expected_profit"].sum(), 2)
        )

        col3.metric(
            "Expected Rooms Sold",
            round(df["expected_rooms_sold"].sum(), 0)
        )

        fig = px.bar(
            df,
            x="hotel_id",
            y="rooms_to_purchase",
            color="supplier_id",
            title="Procurement Allocation by Hotel and Supplier"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

# SCENARIO ANALYSIS
elif page == "Scenario Analysis":

    st.header("Scenario Analysis & Risk")

    df = data["scenarios"]

    if df.empty:
        st.warning("Scenario results not found.")
    else:

        fig = px.bar(
            df,
            x="scenario",
            y="profit",
            color="scenario",
            title="Profit Under Different Demand Conditions"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df, use_container_width=True)

        if "probability" in df.columns:

            expected_profit = (
                df["profit"] * df["probability"]
            ).sum()

            st.metric(
                "Probability-Weighted Expected Profit",
                round(expected_profit, 2)
            )

st.sidebar.divider()
st.sidebar.caption("TravelPulse | Hackathon Prototype")
