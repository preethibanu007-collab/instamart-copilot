import re, sqlite3
import numpy as np
import pandas as pd

RAW = "data/raw"
def load(n):
    return pd.read_csv(f"{RAW}/{n}.csv", keep_default_na=False, na_values=[""])

print("=== AUDIT OF RAW DATA ===")
o, c, p = load("orders"), load("customers"), load("products")
for name, df in [("orders", o), ("customers", c), ("products", p)]:
    print(f"\n{name}: shape={df.shape}")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)}))
age = pd.to_numeric(c["age"], errors="coerce")
audit = {
    "raw order rows": len(o),
    "exact duplicate rows": int(o.duplicated().sum()),
    "pending rows": int((o["status"] == "pending").sum()),
    "distinct order_id": o["order_id"].nunique(),
    "non-numeric order_amount": int(pd.to_numeric(o["order_amount"], errors="coerce").isna().sum()
                                    - o["order_amount"].isna().sum()),
    "negative delivery_minutes": int((pd.to_numeric(o["delivery_minutes"], errors="coerce") < 0).sum()),
    "blank delivery_minutes": int(o["delivery_minutes"].isna().sum()),
    "blank age": int(age.isna().sum()),
    "age outside 10-100": int(((age < 10) | (age > 100)).sum()),
    "distinct raw city spellings": c["city"].nunique(),
    "distinct non-blank gender spellings": c["gender"].dropna().nunique(),
    "blank gender": int(c["gender"].isna().sum()),
}
print("\nAUDIT FIGURES (reference: 13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30)")
for k, v in audit.items():
    print(f"  {k}: {v}")

print("\n=== SQL QUERIES ===")
con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
parts = re.split(r"(?m)^-- (Q\d+)\s*$", text)
res = {parts[i]: pd.read_sql_query(parts[i + 1].strip().rstrip(";"), con)
       for i in range(1, len(parts), 2)}
for q, df in res.items():
    print(f"{q}: {len(df)} rows")
print("Q1 customers (ref 30):", len(res["Q1"]))
print("Q2 rows (ref 30), total (ref 2998668.80):", len(res["Q2"]), round(res["Q2"]["revenue"].sum(), 2))
print("Q2 largest cell (ref Bengaluru/Electronics 233909.45):", res["Q2"].iloc[0].to_dict())
print("Q3 products (ref 11):", len(res["Q3"]))
print("Q4 customers with no completed order (ref 25):", len(res["Q4"]))
print("Q5 customers above average (ref 550) / average (ref 2552.06):",
      len(res["Q5"]), res["Q5"]["avg_total"].iloc[0])
q6 = res["Q6"]
print("Q6 top per city:\n", q6[q6["rnk"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("Q6 rows where RANK != ROW_NUMBER (ref 8):", int((q6["rnk"] != q6["row_num"]).sum()))
q7 = res["Q7"]
print("Q7 daily rows (ref 183) / last cumulative (ref 2998668.80):",
      len(q7), q7["cumulative_revenue"].iloc[-1])

print("\n=== PANDAS CROSS-CHECK ===")
d = pd.read_sql_query(
    "SELECT o.order_amount, o.discount_pct, o.quantity, o.delivery_minutes, c.city, p.category "
    "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
    "JOIN products p ON o.product_id = p.product_id WHERE o.status = 'completed'", con)
pv = pd.pivot_table(d, values="order_amount", index="city", columns="category",
                    aggfunc="sum", margins=True)
print("pivot grand total:", round(pv.loc["All", "All"], 2), "| SQL total:", round(res["Q2"]["revenue"].sum(), 2))
print(d.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"),
                            count=("order_amount", "count")).round(2))

print("\n=== EXPLORATORY STATISTICS (completed order_amount) ===")
s = d["order_amount"]
q1, q2, q3 = s.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"mean {s.mean():.2f} | median {q2:.2f} | skew {s.skew():.2f}")
print(f"Q1 {q1:.2f} | Q3 {q3:.2f} | lower fence {lo:.2f} | upper fence {hi:.2f}")
print("outliers below:", int((s < lo).sum()), "| above:", int((s > hi).sum()))
cm = d[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(cm.round(3))
cols = list(cm.columns)
pairs = [(cols[i], cols[j], cm.iloc[i, j]) for i in range(4) for j in range(i + 1, 4)]
best = max(pairs, key=lambda t: abs(t[2]))
print("largest |correlation|:", best[0], "vs", best[1], round(best[2], 3))