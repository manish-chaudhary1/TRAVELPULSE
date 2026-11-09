
import pandas as pd
import numpy as np
import pulp
from pathlib import Path
import math

# -----------------------------------------
# 1. CONFIGURATION
# -----------------------------------------

BUDGET = 100000  # Total procurement budget in rupees

Path("outputs").mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# 2. LOAD DATA
# -----------------------------------------

forecast = pd.read_csv("outputs/forecast_results.csv")
inventory = pd.read_csv("data/supplier_inventory.csv")

# -----------------------------------------
# 3. CALCULATE TOTAL DEMAND
# -----------------------------------------

# Sum predictions across the forecast horizon.
# This is total predicted demand, not daily average demand.

hotel_demand = (
    forecast.groupby(["hotel_id", "destination"])
    .agg(
        predicted_demand=("predicted_demand", "sum"),
        actual_demand=("net_bookings", "sum")
    )
    .reset_index()
)

# -----------------------------------------
# 4. PREPARE SUPPLIER OFFERS
# -----------------------------------------

offers = (
    inventory.groupby(
        ["hotel_id", "destination", "supplier_id"]
    )
    .agg(
        capacity=("available_rooms", "sum"),
        procurement_cost=("procurement_cost", "mean"),
        selling_price=("selling_price", "mean"),
        recovery_value=("recovery_value", "mean")
    )
    .reset_index()
)

# Attach hotel demand to every supplier offer
offers = offers.merge(
    hotel_demand,
    on=["hotel_id", "destination"],
    how="inner"
)

# Keep only valid offers
offers = offers[
    (offers["capacity"] > 0)
    & (offers["procurement_cost"] > 0)
    & (offers["selling_price"] > 0)
].reset_index(drop=True)

if offers.empty:
    raise ValueError("No valid supplier offers were found.")

# -----------------------------------------
# 5. CREATE OPTIMIZATION PROBLEM
# -----------------------------------------

problem = pulp.LpProblem(
    "TravelPulse_Inventory_Optimization",
    pulp.LpMaximize
)

# -----------------------------------------
# 6. DECISION VARIABLES
# -----------------------------------------

# x = rooms purchased
# y = rooms sold
# u = rooms left unsold

indices = range(len(offers))

purchase = pulp.LpVariable.dicts(
    "Purchase",
    indices,
    lowBound=0,
    cat="Integer"
)

sold = pulp.LpVariable.dicts(
    "Sold",
    indices,
    lowBound=0,
    cat="Integer"
)

unsold = pulp.LpVariable.dicts(
    "Unsold",
    indices,
    lowBound=0,
    cat="Integer"
)

# -----------------------------------------
# 7. OBJECTIVE FUNCTION
# -----------------------------------------

problem += pulp.lpSum(
    offers.loc[i, "selling_price"] * sold[i]
    + offers.loc[i, "recovery_value"] * unsold[i]
    - offers.loc[i, "procurement_cost"] * purchase[i]
    for i in indices
)

# -----------------------------------------
# 8. BUDGET CONSTRAINT
# -----------------------------------------

problem += pulp.lpSum(
    offers.loc[i, "procurement_cost"] * purchase[i]
    for i in indices
) <= BUDGET

# -----------------------------------------
# 9. SUPPLIER CAPACITY CONSTRAINTS
# -----------------------------------------

for i in indices:

    problem += (
        purchase[i] <= int(offers.loc[i, "capacity"])
    )

# -----------------------------------------
# 10. SOLD AND UNSOLD CONSTRAINTS
# -----------------------------------------

for i in indices:

    problem += sold[i] + unsold[i] == purchase[i]

# -----------------------------------------
# 11. HOTEL DEMAND CONSTRAINTS
# -----------------------------------------

for (hotel_id, destination), group in offers.groupby(
    ["hotel_id", "destination"]
):

    hotel_indices = group.index.tolist()

    demand = group["predicted_demand"].iloc[0]

    # Total rooms sold cannot exceed predicted demand
    problem += pulp.lpSum(
        sold[i] for i in hotel_indices
    ) <= demand

    # Avoid procuring more than predicted hotel demand
    problem += pulp.lpSum(
        purchase[i] for i in hotel_indices
    ) <= math.ceil(demand)

# -----------------------------------------
# 12. SOLVE THE PROBLEM
# -----------------------------------------

solver = pulp.PULP_CBC_CMD(msg=False)

import shutil
import pulp

cbc_path = shutil.which("cbc")

if not cbc_path:
    raise RuntimeError("CBC was not found. Check your Homebrew installation.")

print("Using CBC solver:", cbc_path)

solver = pulp.COIN_CMD(
    path=cbc_path,
    msg=True
)

problem.solve(solver)

print("Solver status:", pulp.LpStatus[problem.status])
status = pulp.LpStatus[problem.status]

print("\nOPTIMIZATION STATUS:", status)

if status != "Optimal":
    raise RuntimeError(
        f"Optimizer did not find an optimal solution: {status}"
    )

# -----------------------------------------
# 13. EXTRACT RESULTS
# -----------------------------------------

offers["rooms_to_purchase"] = [
    int(round(pulp.value(purchase[i]) or 0))
    for i in indices
]

offers["expected_rooms_sold"] = [
    int(round(pulp.value(sold[i]) or 0))
    for i in indices
]

offers["expected_unsold_rooms"] = [
    int(round(pulp.value(unsold[i]) or 0))
    for i in indices
]

offers["procurement_amount"] = (
    offers["rooms_to_purchase"]
    * offers["procurement_cost"]
)

offers["expected_revenue"] = (
    offers["expected_rooms_sold"]
    * offers["selling_price"]
)

offers["recovery_amount"] = (
    offers["expected_unsold_rooms"]
    * offers["recovery_value"]
)

offers["expected_profit"] = (
    offers["expected_revenue"]
    + offers["recovery_amount"]
    - offers["procurement_amount"]
)

# -----------------------------------------
# 14. DISPLAY RESULTS
# -----------------------------------------

selected = offers[
    offers["rooms_to_purchase"] > 0
].copy()

print("\nOPTIMAL PROCUREMENT PLAN")
print("------------------------")

if selected.empty:
    print("No inventory should be purchased.")
else:
    print(
        selected[
            [
                "hotel_id",
                "destination",
                "supplier_id",
                "rooms_to_purchase",
                "expected_rooms_sold",
                "expected_unsold_rooms",
                "procurement_amount",
                "expected_profit"
            ]
        ].to_string(index=False)
    )

# -----------------------------------------
# 15. FINANCIAL SUMMARY
# -----------------------------------------

total_investment = offers["procurement_amount"].sum()
total_revenue = offers["expected_revenue"].sum()
total_recovery = offers["recovery_amount"].sum()
total_profit = offers["expected_profit"].sum()

print("\nFINANCIAL SUMMARY")
print("-----------------")

print("Budget:", BUDGET)
print("Investment:", round(total_investment, 2))
print("Unused Budget:", round(BUDGET - total_investment, 2))
print("Expected Revenue:", round(total_revenue, 2))
print("Recovery Value:", round(total_recovery, 2))
print("Expected Profit:", round(total_profit, 2))

if total_investment > 0:
    roi = (total_profit / total_investment) * 100
else:
    roi = 0

print("Expected ROI:", round(roi, 2), "%")

# -----------------------------------------
# 16. SAVE PROCUREMENT PLAN
# -----------------------------------------

offers.to_csv(
    "outputs/optimal_procurement_plan.csv",
    index=False
)

print("\nSaved: outputs/optimal_procurement_plan.csv")
print("INVENTORY OPTIMIZATION COMPLETED!")
