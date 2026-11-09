
import pandas as pd
import numpy as np

# -----------------------------------------
# 1. LOAD THE DATASETS
# -----------------------------------------

bookings = pd.read_csv("data/historical_bookings.csv")
calendar = pd.read_csv("data/calendar.csv")

bookings["travel_date"] = pd.to_datetime(bookings["travel_date"])
calendar["date"] = pd.to_datetime(calendar["date"])

# -----------------------------------------
# 2. AGGREGATE HOTEL-WISE DEMAND
# -----------------------------------------

# Multiple booking records can refer to the same
# hotel and travel date. Combine them.

daily_demand = (
    bookings.groupby(["hotel_id", "destination", "travel_date"])
    .agg(
        net_bookings=("net_bookings", "sum"),
        avg_selling_price=("selling_price", "mean"),
        avg_procurement_cost=("procurement_cost", "mean")
    )
    .reset_index()
)

# -----------------------------------------
# 3. CREATE A COMPLETE HOTEL-DATE GRID
# -----------------------------------------

hotels = bookings[
    ["hotel_id", "destination"]
].drop_duplicates()

grid = hotels.merge(calendar, how="cross")

grid = grid.rename(columns={"date": "travel_date"})

# Add observed demand to the complete date grid
df = grid.merge(
    daily_demand,
    on=["hotel_id", "destination", "travel_date"],
    how="left"
)

# No recorded bookings means zero observed bookings
df["net_bookings"] = df["net_bookings"].fillna(0)

# Fill missing prices using each hotel's historical median
for column in ["avg_selling_price", "avg_procurement_cost"]:
    df[column] = df.groupby("hotel_id")[column].transform(
        lambda series: series.fillna(series.median())
    )

# -----------------------------------------
# 4. CREATE CALENDAR FEATURES
# -----------------------------------------

df["month"] = df["travel_date"].dt.month
df["day_of_week"] = df["travel_date"].dt.dayofweek
df["day_of_month"] = df["travel_date"].dt.day

df["weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# -----------------------------------------
# 5. CREATE HISTORICAL DEMAND FEATURES
# -----------------------------------------

df = df.sort_values(
    ["hotel_id", "travel_date"]
).reset_index(drop=True)

# Demand 7 and 14 days earlier
df["lag_7"] = (
    df.groupby("hotel_id")["net_bookings"].shift(7)
)

df["lag_14"] = (
    df.groupby("hotel_id")["net_bookings"].shift(14)
)

# Previous 7-day average
df["rolling_mean_7"] = (
    df.groupby("hotel_id")["net_bookings"]
    .transform(
        lambda series: series.shift(1).rolling(7).mean()
    )
)

# Previous 30-day average
df["rolling_mean_30"] = (
    df.groupby("hotel_id")["net_bookings"]
    .transform(
        lambda series: series.shift(1).rolling(30).mean()
    )
)

# -----------------------------------------
# 6. REMOVE ROWS WITHOUT HISTORICAL FEATURES
# -----------------------------------------

df = df.dropna(
    subset=[
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_30"
    ]
)

# -----------------------------------------
# 7. SAVE THE FINAL DATASET
# -----------------------------------------

df.to_csv(
    "data/modeling_dataset.csv",
    index=False
)

# -----------------------------------------
# 8. DISPLAY RESULTS
# -----------------------------------------

print("FEATURE ENGINEERING COMPLETED!")

print("\nDataset preview:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nFeatures created:")
print(df.columns.tolist())

print("\nSaved: data/modeling_dataset.csv")
