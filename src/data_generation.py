import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# 1. BASIC CONFIGURATION
# --------------------------------------------------

rng = np.random.default_rng(42)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

START_DATE = "2025-01-01"
DAYS = 365

DESTINATIONS = {
    "Delhi": 1.3,
    "Goa": 1.5,
    "Jaipur": 1.1,
    "Manali": 1.2,
    "Mumbai": 1.4
}

SUPPLIERS = ["Supplier_A", "Supplier_B", "Supplier_C"]


# --------------------------------------------------
# 2. CREATE HOTEL INFORMATION
# --------------------------------------------------

hotels = []

hotel_number = 1

for destination, demand_factor in DESTINATIONS.items():

    for i in range(4):

        hotels.append({
            "hotel_id": f"H{hotel_number:03d}",
            "destination": destination,
            "hotel_name": f"{destination}_Hotel_{i + 1}",
            "demand_factor": demand_factor * rng.uniform(0.8, 1.2),
            "room_capacity": int(rng.integers(30, 101)),
            "base_price": int(rng.integers(2500, 7001))
        })

        hotel_number += 1

hotel_df = pd.DataFrame(hotels)


# --------------------------------------------------
# 3. CALENDAR FEATURES
# --------------------------------------------------

calendar_dates = pd.date_range(
    start=START_DATE,
    periods=DAYS + 60,
    freq="D"
)

calendar = pd.DataFrame({
    "date": calendar_dates
})

calendar["month"] = calendar["date"].dt.month
calendar["day_of_week"] = calendar["date"].dt.dayofweek
calendar["weekend"] = (
    calendar["day_of_week"] >= 5
).astype(int)

# Illustrative Indian holidays for the synthetic dataset
holidays = pd.to_datetime([
    "2025-01-26",
    "2025-03-14",
    "2025-08-15",
    "2025-10-02",
    "2025-10-20",
    "2025-12-25",
    "2026-01-26"
])

calendar["holiday"] = calendar["date"].isin(holidays).astype(int)


def get_season(month):

    if month in [3, 4, 5, 6]:
        return "Summer"

    elif month in [7, 8, 9]:
        return "Monsoon"

    elif month in [10, 11, 12, 1]:
        return "Peak"

    else:
        return "Winter"


calendar["season"] = calendar["month"].apply(get_season)


# --------------------------------------------------
# 4. GENERATE HISTORICAL BOOKINGS
# --------------------------------------------------

booking_dates = pd.date_range(
    start=START_DATE,
    periods=DAYS,
    freq="D"
)

season_factors = {
    "Summer": 1.15,
    "Monsoon": 0.75,
    "Peak": 1.40,
    "Winter": 1.00
}

historical_records = []

for hotel in hotels:

    for booking_date in booking_dates:

        # Random number of days between booking and travel
        lead_days = int(rng.integers(1, 61))

        travel_date = (
            booking_date + pd.Timedelta(days=lead_days)
        )

        month = travel_date.month
        season = get_season(month)

        # Seasonal and weekend demand
        season_factor = season_factors[season]

        weekend_factor = (
            1.25 if travel_date.dayofweek >= 5 else 1.0
        )

        # Gradual demand growth
        day_number = (booking_date - booking_dates[0]).days
        trend_factor = 1 + (day_number / DAYS) * 0.12

        # Expected search volume
        expected_searches = (
            40
            * hotel["demand_factor"]
            * season_factor
            * weekend_factor
            * trend_factor
        )

        searches = int(
            rng.poisson(expected_searches)
        )

        # Conversion rate: proportion of searches becoming bookings
        conversion_rate = rng.uniform(0.06, 0.22)

        bookings = int(
            rng.binomial(searches, conversion_rate)
        )

        # Cancellation rate
        cancellation_rate = rng.uniform(0.03, 0.15)

        cancellations = int(
            rng.binomial(bookings, cancellation_rate)
        )

        net_bookings = bookings - cancellations

        # Selling price with small random variation
        selling_price = round(
            hotel["base_price"] * rng.uniform(0.90, 1.15),
            2
        )

        procurement_cost = round(
            selling_price * rng.uniform(0.60, 0.82),
            2
        )

        sold_out = int(
            net_bookings >= hotel["room_capacity"]
        )

        historical_records.append({

            "date": booking_date,
            "travel_date": travel_date,
            "destination": hotel["destination"],
            "hotel_id": hotel["hotel_id"],
            "searches": searches,
            "bookings": bookings,
            "cancellations": cancellations,
            "net_bookings": net_bookings,
            "sold_out": sold_out,
            "selling_price": selling_price,
            "procurement_cost": procurement_cost,
            "room_capacity": hotel["room_capacity"]

        })


historical_bookings = pd.DataFrame(historical_records)


# --------------------------------------------------
# 5. GENERATE SUPPLIER INVENTORY
# --------------------------------------------------

inventory_records = []

for hotel in hotels:

    for supplier in SUPPLIERS:

        selling_price = hotel["base_price"]

        procurement_cost = round(
            selling_price * rng.uniform(0.58, 0.82),
            2
        )

        available_rooms = int(
            rng.integers(5, 51)
        )

        recovery_value = round(
            procurement_cost * rng.uniform(0.10, 0.40),
            2
        )

        inventory_records.append({

            "hotel_id": hotel["hotel_id"],
            "destination": hotel["destination"],
            "supplier_id": supplier,
            "available_rooms": available_rooms,
            "procurement_cost": procurement_cost,
            "selling_price": selling_price,
            "recovery_value": recovery_value

        })


supplier_inventory = pd.DataFrame(inventory_records)


# --------------------------------------------------
# 6. SAVE ALL THREE CSV FILES
# --------------------------------------------------

historical_bookings.to_csv(
    OUTPUT_DIR / "historical_bookings.csv",
    index=False
)

supplier_inventory.to_csv(
    OUTPUT_DIR / "supplier_inventory.csv",
    index=False
)

calendar.to_csv(
    OUTPUT_DIR / "calendar.csv",
    index=False
)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("DATA GENERATION COMPLETED!")

print("\nHistorical Bookings:")
print(historical_bookings.head())

print("\nSupplier Inventory:")
print(supplier_inventory.head())

print("\nCalendar:")
print(calendar.head())

print("\nDataset Shapes:")

print("Historical Bookings:", historical_bookings.shape)
print("Supplier Inventory:", supplier_inventory.shape)
print("Calendar:", calendar.shape)

print("\nCSV files saved inside the data folder.")