
import pandas as pd
import numpy as np
from pathlib import Path

# Create output directory if it doesn't exist
Path("outputs").mkdir(parents=True, exist_ok=True)

# Load the optimal procurement plan
plan = pd.read_csv("outputs/optimal_procurement_plan.csv")

# Check the columns
print("Available columns:")
print(plan.columns.tolist())

# Define demand scenarios
scenarios = {
    "Low Demand": 0.80,
    "Normal Demand": 1.00,
    "High Demand": 1.20
}

scenario_results = []

# Evaluate the procurement plan under each scenario

for scenario_name, demand_multiplier in scenarios.items():

    temp = plan.copy()

    temp["scenario_demand"] = (
        temp["predicted_demand"] * demand_multiplier
    )

    temp["scenario_sold"] = np.minimum(
        temp["rooms_to_purchase"],
        temp["scenario_demand"]
    )

    temp["scenario_unsold"] = (
        temp["rooms_to_purchase"] - temp["scenario_sold"]
    )

    temp["scenario_revenue"] = (
        temp["scenario_sold"] * temp["selling_price"]
    )

    temp["scenario_recovery"] = (
        temp["scenario_unsold"] * temp["recovery_value"]
    )

    temp["scenario_cost"] = (
        temp["rooms_to_purchase"] * temp["procurement_cost"]
    )

    temp["scenario_profit"] = (
        temp["scenario_revenue"]
        + temp["scenario_recovery"]
        - temp["scenario_cost"]
    )

    scenario_results.append({
        "scenario": scenario_name,
        "demand_multiplier": demand_multiplier,
        "total_demand": temp["scenario_demand"].sum(),
        "rooms_purchased": temp["rooms_to_purchase"].sum(),
        "rooms_sold": temp["scenario_sold"].sum(),
        "unsold_rooms": temp["scenario_unsold"].sum(),
        "revenue": temp["scenario_revenue"].sum(),
        "recovery": temp["scenario_recovery"].sum(),
        "procurement_cost": temp["scenario_cost"].sum(),
        "profit": temp["scenario_profit"].sum()
    })

# Convert results into a DataFrame
results = pd.DataFrame(scenario_results)

# Calculate expected profit using scenario probabilities
probabilities = {
    "Low Demand": 0.25,
    "Normal Demand": 0.50,
    "High Demand": 0.25
}

results["probability"] = results["scenario"].map(probabilities)

expected_profit = (
    results["profit"] * results["probability"]
).sum()

worst_case_profit = results["profit"].min()

# Display results
print("\nSCENARIO ANALYSIS")
print(results.to_string(index=False))

print("\nRISK ANALYSIS")
print("Expected Profit:", round(expected_profit, 2))
print("Worst Case Profit:", round(worst_case_profit, 2))

# Save results
results.to_csv(
    "outputs/scenario_results.csv",
    index=False
)

print("\nScenario results saved successfully!")
