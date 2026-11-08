import os
import pandas as pd
import numpy as np

CUT = pd.Timestamp("2025-09-01")
END = pd.Timestamp("2025-10-01")

o = pd.read_csv("data/orders_clean.csv", parse_dates=["order_ts"])
c = pd.read_csv("data/customers_clean.csv", parse_dates=["signup_date"])
p = pd.read_csv("data/raw/products.csv")
o = o.merge(p[["product_id", "category"]], on="product_id", how="left")

before = o[o["order_ts"] < CUT]
comp = before[before["status"] == "completed"]

sept = o[(o["status"] == "completed") & (o["order_ts"] >= CUT) & (o["order_ts"] < END)]
active = set(sept["customer_id"])

g = comp.groupby("customer_id")
f = pd.DataFrame({
    "recency": (CUT - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount": g["discount_pct"].mean(),
    "avg_delivery": g["delivery_minutes"].mean(),
    "fav_category": g["category"].agg(lambda s: s.value_counts().index[0]),
})
fin = before[before["status"].isin(["completed", "cancelled"])]
canc = fin.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())
f["cancel_rate"] = canc.reindex(f.index).fillna(0)

f = f.reset_index().merge(
    c[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
f["tenure_days"] = (CUT - f["signup_date"]).dt.days
f = f.drop(columns="signup_date")
f["lapsed"] = (~f["customer_id"].isin(active)).astype(int)

os.makedirs("outputs", exist_ok=True)
f.to_csv("outputs/lapse_features.csv", index=False)

print("cohort:", len(f), "| lapsed:", f["lapsed"].sum(),
      "| rate:", round(f["lapsed"].mean() * 100, 2), "%")
print(f.isna().sum())