
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("data/historical_bookings.csv")

# Convert date columns into datetime format
df["date"] = pd.to_datetime(df["date"])
df["travel_date"] = pd.to_datetime(df["travel_date"])

# 2. Basic information
print("\n--- DATASET INFORMATION ---")
print(df.info())

print("\n--- FIRST FIVE RECORDS ---")
print(df.head())

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 3. Check missing values and duplicates
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

# 4. Calculate important business metrics
df["conversion_rate"] = (
    df["bookings"] / df["searches"].replace(0, float("nan"))
)

df["cancellation_rate"] = (
    df["cancellations"] / df["bookings"].replace(0, float("nan"))
)

df["net_bookings"] = df["bookings"] - df["cancellations"]

# 5. Destination-wise analysis
destination_analysis = (
    df.groupby("destination")
    .agg(
        total_searches=("searches", "sum"),
        total_bookings=("bookings", "sum"),
        total_cancellations=("cancellations", "sum"),
        net_bookings=("net_bookings", "sum"),
        avg_conversion=("conversion_rate", "mean")
    )
    .sort_values("net_bookings", ascending=False)
)

print("\n--- DESTINATION ANALYSIS ---")
print(destination_analysis)

# 6. Hotel-wise analysis
hotel_analysis = (
    df.groupby(["destination", "hotel_id"])
    .agg(
        bookings=("bookings", "sum"),
        cancellations=("cancellations", "sum"),
        net_bookings=("net_bookings", "sum")
    )
    .sort_values("net_bookings", ascending=False)
)

print("\n--- TOP 10 HOTELS ---")
print(hotel_analysis.head(10))

# 7. Monthly demand analysis
df["month"] = df["travel_date"].dt.month

monthly_demand = (
    df.groupby("month")["net_bookings"].sum()
)

print("\n--- MONTHLY DEMAND ---")
print(monthly_demand)

# 8. Plot destination-wise bookings
destination_analysis["net_bookings"].plot(
    kind="bar",
    figsize=(9, 5),
    title="Net Bookings by Destination"
)

plt.ylabel("Net Bookings")
plt.xlabel("Destination")
plt.tight_layout()
plt.show()

# 9. Plot monthly demand
monthly_demand.plot(
    kind="line",
    marker="o",
    figsize=(10, 5),
    title="Monthly Travel Demand"
)

plt.ylabel("Net Bookings")
plt.xlabel("Month")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Plot searches vs bookings
plt.figure(figsize=(9, 5))

plt.scatter(
    df["searches"],
    df["bookings"],
    alpha=0.4
)

plt.title("Searches vs Bookings")
plt.xlabel("Searches")
plt.ylabel("Bookings")
plt.tight_layout()
plt.show()

print("\nEDA COMPLETED!")
