import os, sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
o = pd.read_sql_query("SELECT * FROM orders", con)
c = pd.read_sql_query("SELECT * FROM customers", con)
done = o[o["status"] == "completed"]
canc = o[o["status"] == "cancelled"]

checks = {
    "orders rows = 12309": len(o) == 12309,
    "one row per order_id": o["order_id"].nunique() == 12309,
    "completed = 11110": len(done) == 11110,
    "cancelled = 1199": len(canc) == 1199,
    "no pending rows": (o["status"] == "pending").sum() == 0,
    "0 missing order_amount": o["order_amount"].isna().sum() == 0,
    "0 missing delivery (completed)": done["delivery_minutes"].isna().sum() == 0,
    "1199 missing delivery (cancelled)": canc["delivery_minutes"].isna().sum() == 1199,
    "completed revenue = 2998668.80": abs(done["order_amount"].sum() - 2998668.80) < 0.01,
    "six cities with right counts": c["city"].value_counts().to_dict() == {
        "Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191,
        "Hyderabad": 191, "Mumbai": 184},
    "genders 610 / 560 / 30": c["gender"].value_counts().to_dict() == {
        "female": 610, "male": 560, "unknown": 30},
    "0 missing ages, all 18-60": c["age"].isna().sum() == 0 and c["age"].between(18, 60).all(),
    "data/clean/orders.csv exists with 12309 rows":
        os.path.exists("data/clean/orders.csv") and len(pd.read_csv("data/clean/orders.csv")) == 12309,
    "data/clean/customers.csv exists with 1200 rows":
        os.path.exists("data/clean/customers.csv") and len(pd.read_csv("data/clean/customers.csv")) == 1200,
}
bad = [k for k, v in checks.items() if not v]
for k in bad:
    print("FAILED:", k)
if not bad:
    print("ALL CHECKPOINTS PASSED")