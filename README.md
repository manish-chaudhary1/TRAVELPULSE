# TravelPulse – AI-Powered Travel Inventory Intelligence & Optimization Engine

**TravelPulse** is a data-driven travel inventory intelligence platform designed to help travel businesses make smarter hotel procurement decisions.

It combines demand forecasting, opportunity detection, mathematical optimization, and scenario analysis to recommend how much hotel inventory should be procured while considering demand, supplier capacity, costs, and potential revenue.

## Problem Statement

Travel businesses face uncertainty when purchasing hotel inventory. Underestimating demand can result in missed revenue opportunities, while overestimating demand can lead to unsold rooms and financial losses.

TravelPulse addresses this challenge by combining predictive analytics with optimization techniques to support more informed inventory procurement decisions.

## Proposed Solution

TravelPulse provides an integrated workflow:

1. **Data Generation & Processing:** Creates and processes historical booking, supplier inventory, and calendar datasets.
2. **Exploratory Data Analysis:** Examines booking patterns, demand trends, cancellations, and supplier information.
3. **Feature Engineering:** Generates time-based and historical demand features.
4. **Demand Forecasting:** Uses machine learning to estimate future hotel demand.
5. **Opportunity Detection:** Identifies potential procurement opportunities based on demand, inventory, and profitability.
6. **Inventory Optimization:** Uses Linear Programming to determine procurement quantities under supplier capacity and budget constraints.
7. **Scenario Analysis:** Evaluates procurement decisions under low-, normal-, and high-demand conditions.
8. **Interactive Dashboard:** Presents forecasts, opportunities, procurement recommendations, and scenario results through Streamlit.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Optimization | PuLP (Linear Programming) |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Storage | Joblib |

## Project Structure

```text
TravelPulse/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── historical_bookings.csv
│   ├── supplier_inventory.csv
│   ├── calendar.csv
│   └── modeling_dataset.csv
│
├── src/
│   ├── data_generation.py
│   ├── data_analysis.py
│   ├── feature_engineering.py
│   ├── demand_forecasting.py
│   ├── opportunity_detection.py
│   ├── inventory_optimizer.py
│   └── scenario_generation.py
│
├── models/
│   └── demand_model.pkl
│
└── outputs/
    ├── forecast_results.csv
    ├── opportunity_results.csv
    ├── optimal_procurement_plan.csv
    └── scenario_results.csv
```

## Optimization Approach

The inventory optimization module uses Linear Programming to maximize expected procurement profit.

The objective considers:

- Revenue from rooms sold.
- Recovery value from unsold rooms.
- Supplier procurement costs.

The optimization model also considers:

- Supplier inventory capacity.
- Total procurement budget.
- Forecasted demand constraints.

## Scenario Analysis

TravelPulse evaluates procurement decisions under three
