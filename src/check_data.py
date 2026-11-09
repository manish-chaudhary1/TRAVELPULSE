import pandas as pd

df = pd.read_csv("data/historical_bookings.csv")

print("First five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBookings by destination:")
print(df.groupby("destination")["bookings"].sum())

print("\nAverage cancellation rate:")
print(
    df["cancellations"].sum() /
    df["bookings"].sum()
)

print("\nSold-out records:")
print(df["sold_out"].sum())