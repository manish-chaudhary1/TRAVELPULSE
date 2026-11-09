
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

df = pd.read_csv("data/modeling_dataset.csv")

df["travel_date"] = pd.to_datetime(df["travel_date"])

df = df.sort_values(
    ["travel_date", "hotel_id"]
).reset_index(drop=True)


# -----------------------------------------
# 2. SELECT FEATURES AND TARGET
# -----------------------------------------

features = [
    "month",
    "day_of_week",
    "day_of_month",
    "weekend",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_30"
]

target = "net_bookings"

X = df[features]
y = df[target]


# -----------------------------------------
# 3. CHRONOLOGICAL TRAIN/TEST SPLIT
# -----------------------------------------

# Never randomly shuffle time-series data.
# Earlier dates train the model.
# Later dates test its performance.

unique_dates = np.sort(df["travel_date"].unique())

train_end = int(len(unique_dates) * 0.70)
validation_end = int(len(unique_dates) * 0.85)

train_dates = unique_dates[:train_end]
validation_dates = unique_dates[train_end:validation_end]
test_dates = unique_dates[validation_end:]

train_mask = df["travel_date"].isin(train_dates)
validation_mask = df["travel_date"].isin(validation_dates)
test_mask = df["travel_date"].isin(test_dates)

X_train = X[train_mask]
y_train = y[train_mask]

X_validation = X[validation_mask]
y_validation = y[validation_mask]

X_test = X[test_mask]
y_test = y[test_mask]


# -----------------------------------------
# 4. TRAIN RANDOM FOREST
# -----------------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -----------------------------------------
# 5. PREDICT DEMAND
# -----------------------------------------

predictions = model.predict(X_test)

# Demand cannot be negative
predictions = np.maximum(predictions, 0)


# -----------------------------------------
# 6. BASELINE COMPARISON
# -----------------------------------------

# Baseline: predict using previous 7-day average

baseline_predictions = X_test["rolling_mean_7"].to_numpy()


# -----------------------------------------
# 7. EVALUATE BOTH MODELS
# -----------------------------------------

def evaluate(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    wape = (
        np.sum(np.abs(actual - predicted))
        / max(np.sum(np.abs(actual)), 1e-9)
    ) * 100

    return mae, rmse, wape


rf_mae, rf_rmse, rf_wape = evaluate(
    y_test.to_numpy(), predictions
)

base_mae, base_rmse, base_wape = evaluate(
    y_test.to_numpy(), baseline_predictions
)


# -----------------------------------------
# 8. DISPLAY RESULTS
# -----------------------------------------

print("\nRANDOM FOREST RESULTS")
print("---------------------")

print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("WAPE:", round(rf_wape, 2), "%")


print("\nBASELINE RESULTS")
print("----------------")

print("MAE:", round(base_mae, 2))
print("RMSE:", round(base_rmse, 2))
print("WAPE:", round(base_wape, 2), "%")


# -----------------------------------------
# 9. SAVE MODEL
# -----------------------------------------

Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/demand_model.pkl"
)


# -----------------------------------------
# 10. SAVE TEST PREDICTIONS
# -----------------------------------------

results = df.loc[test_mask, [
    "travel_date",
    "hotel_id",
    "destination",
    "net_bookings"
]].copy()

results["predicted_demand"] = predictions

results["baseline_prediction"] = baseline_predictions

results.to_csv(
    "outputs/forecast_results.csv",
    index=False
)


# -----------------------------------------
# 11. FEATURE IMPORTANCE
# -----------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)


print("\nForecasting completed successfully!")
print("Model saved in models/demand_model.pkl")
print("Predictions saved in outputs/forecast_results.csv")

print("\nTEST DATA DIAGNOSTICS")

print("Actual demand total:", y_test.sum())
print("Average actual demand:", y_test.mean())
print("Zero-demand records:", (y_test == 0).sum())
print("Total test records:", len(y_test))