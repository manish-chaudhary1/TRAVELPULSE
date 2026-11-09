
import pandas as pd
import numpy as np

# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

forecast = pd.read_csv("outputs/forecast_results.csv")

inventory = pd.read_csv("data/supplier_inventory.csv")

# -----------------------------------------
# 2. AGGREGATE INVENTORY BY HOTEL
# -----------------------------------------

hotel_inventory = (
    inventory.groupby(["hotel_id", "destination"])
    .agg(
        available_rooms=("available_rooms", "sum"),
        procurement_cost=("procurement_cost", "mean"),
        selling_price=("selling_price", "mean"),
        recovery_value=("recovery_value", "mean")
    )
    .reset_index()
)

# -----------------------------------------
# 3. AGGREGATE FORECASTS
# -----------------------------------------

hotel_forecast = (
    forecast.groupby(["hotel_id", "destination"])
    .agg(
        predicted_demand=("predicted_demand", "mean"),
        actual_demand=("net_bookings", "mean"),
        baseline_demand=("baseline_prediction", "mean")
    )
    .reset_index()
)

# -----------------------------------------
# 4. COMBINE FORECAST AND INVENTORY
# -----------------------------------------

df = hotel_forecast.merge(
    hotel_inventory,
    on=["hotel_id", "destination"],
    how="inner"
)

# -----------------------------------------
# 5. CALCULATE BUSINESS METRICS
# -----------------------------------------

df["demand_gap"] = (
    df["predicted_demand"] - df["available_rooms"]
)

df["demand_pressure"] = (
    df["predicted_demand"] /
    df["available_rooms"].replace(0, np.nan)
)

df["profit_per_room"] = (
    df["selling_price"] - df["procurement_cost"]
)

df["profit_margin"] = (
    df["profit_per_room"] /
    df["selling_price"].replace(0, np.nan)
)

df["expected_revenue"] = (
    df["predicted_demand"] * df["selling_price"]
)

df["expected_profit"] = (
    df["predicted_demand"] * df["profit_per_room"]
)

# -----------------------------------------
# 6. CALCULATE OPPORTUNITY SCORE
# -----------------------------------------

# Normalize demand pressure between 0 and 1
pressure_score = (
    df["demand_pressure"].fillna(0).clip(0, 2) / 2
)

# Profit margin score
margin_score = df["profit_margin"].clip(0, 1)

# Demand gap score
gap_score = (
    df["demand_gap"].clip(lower=0) /
    df["predicted_demand"].replace(0, np.nan)
).fillna(0).clip(0, 1)

df["opportunity_score"] = (
    0.40 * pressure_score
    + 0.35 * margin_score
    + 0.25 * gap_score
) * 100

# -----------------------------------------
# 7. CLASSIFY OPPORTUNITIES
# -----------------------------------------

def classify_opportunity(row):

    if row["demand_gap"] <= 0:
        return "Low Priority"

    elif row["profit_per_room"] <= 0:
        return "Not Profitable"

    elif row["opportunity_score"] >= 65:
        return "High Priority"

    elif row["opportunity_score"] >= 45:
        return "Medium Priority"

    else:
        return "Low Priority"


df["opportunity_status"] = df.apply(
    classify_opportunity,
    axis=1
)

# -----------------------------------------
# 8. SORT OPPORTUNITIES
# -----------------------------------------

df = df.sort_values(
    "opportunity_score",
    ascending=False
)

# -----------------------------------------
# 9. DISPLAY RESULTS
# -----------------------------------------

print("\nTOP INVENTORY OPPORTUNITIES")
print("---------------------------")

print(
    df[
        [
            "hotel_id",
            "destination",
            "predicted_demand",
            "available_rooms",
            "demand_gap",
            "profit_per_room",
            "opportunity_score",
            "opportunity_status"
        ]
    ].head(10).to_string(index=False)
)

# -----------------------------------------
# 10. SAVE RESULTS
# -----------------------------------------

df.to_csv(
    "outputs/opportunity_results.csv",
    index=False
)

print("\nOpportunity detection completed!")
print("Saved: outputs/opportunity_results.csv")
