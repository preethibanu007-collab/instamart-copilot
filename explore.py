import pandas as pd

o = pd.read_csv("data/raw/orders.csv")
c = pd.read_csv("data/raw/customers.csv")

print("orders rows:", len(o))
print("duplicate order_ids:", o["order_id"].duplicated().sum())
print(o["status"].value_counts())
print("bad amounts:", pd.to_numeric(o["order_amount"], errors="coerce").isna().sum())
print("negative delivery:", (o["delivery_minutes"] < 0).sum())
print(c["city"].str.strip().str.title().value_counts())
print("impossible ages:", ((c["age"] < 18) | (c["age"] > 100)).sum())