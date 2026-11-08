import pandas as pd

orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

assert len(orders) == 12309
assert len(customers) == 1200

assert (orders["status"] == "completed").sum() == 11110
assert (orders["status"] == "cancelled").sum() == 1199

assert orders["order_amount"].isna().sum() == 0

revenue = round(
    orders.loc[
        orders["status"] == "completed",
        "order_amount"
    ].sum(),
    2
)

assert revenue == 2998668.80

print("ALL CHECKPOINTS PASSED")
