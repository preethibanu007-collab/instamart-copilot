import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")

completed = orders[
    orders["status"] == "completed"
]

completed["order_ts"] = pd.to_datetime(
    completed["order_ts"]
)

snapshot_date = pd.Timestamp(
    "2025-10-01 00:00:00"
)

rfm = completed.groupby(
    "customer_id"
).agg(
    recency=(
        "order_ts",
        lambda x: (snapshot_date - x.max()).days
    ),
    frequency=(
        "order_id",
        "nunique"
    ),
    monetary=(
        "order_amount",
        "sum"
    )
).reset_index()

print(rfm.head())
print("\nCustomers:", len(rfm))
import os

os.makedirs("outputs", exist_ok=True)

rfm.to_csv(
    "outputs/rfm.csv",
    index=False
)

print("RFM Saved!")