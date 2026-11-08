import pandas as pd
import sqlite3

# Load the datasets
orders = pd.read_csv("data/raw/orders.csv")
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
)
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")

# Check the datasets
print("Orders loaded:", len(orders))
print("Customers loaded:", len(customers))

# Check missing values
print("\nMissing values in Orders:")
print(orders.isnull().sum())

print("\nMissing values in Customers:")
print(customers.isnull().sum())
print("\nPercent missing delivery_minutes:", orders["delivery_minutes"].isnull().mean() * 100)
print(orders[orders["delivery_minutes"].isnull()]["status"].value_counts())
# Step 1: flag the real problem rows
bad = orders[(orders["status"] == "completed") & (orders["delivery_minutes"].isnull())]
print("Completed but missing time:", len(bad))
# Step 2: fix completed orders with missing delivery time
completed_ok = orders[(orders["status"] == "completed") & (orders["delivery_minutes"].notnull())]
print("Completed with time:", len(completed_ok))

median_time = completed_ok["delivery_minutes"].median()
print("Median delivery minutes:", median_time)

mask = (orders["status"] == "completed") & (orders["delivery_minutes"].isnull())
orders.loc[mask, "delivery_minutes"] = median_time

print("Completed but missing time after fix:", orders[mask].shape[0] - orders.loc[mask, "delivery_minutes"].notnull().sum())
# Step 3: fix missing age and gender
median_age = customers["age"].median()
print("Median age:", median_age)

customers["age"] = customers["age"].fillna(median_age)
customers["gender"] = customers["gender"].fillna("Unknown")

print("\nMissing values in Customers after fix:")
print(customers.isnull().sum())
# Step 4: quality checks
print("\n--- Step 4: Quality checks ---")

# Duplicates
print("Duplicate orders:", orders.duplicated().sum())
print("Duplicate customers:", customers.duplicated().sum())

# Spelling / category check
print("\nStatus values:", orders["status"].unique())
print("Cities:", customers["city"].unique())
print("Gender values:", customers["gender"].unique())

# Unusual values
print("\nAge range:", customers["age"].min(), "to", customers["age"].max())
print("Delivery minutes range:", orders["delivery_minutes"].min(), "to", orders["delivery_minutes"].max())

# Date format
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
print("\nSignup date type:", customers["signup_date"].dtype)
# Step 5: clean text, age and delivery time
print("\n--- Step 5: Fixing problems ---")

# Remove duplicates
orders = orders.drop_duplicates()
# Keep only latest record for each order
orders = orders.sort_values("order_ts")

orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
)
# Add product prices
orders = orders.merge(
    products[["product_id", "price_inr"]],
    on="product_id",
    how="left"
)

# Fill missing order amounts
mask = orders["order_amount"].isna()

orders.loc[mask, "order_amount"] = (
    orders.loc[mask, "price_inr"]
    * orders.loc[mask, "quantity"]
    * (1 - orders.loc[mask, "discount_pct"] / 100)
).round(2)
customers = customers.drop_duplicates()

# City: remove spaces and make capitalisation the same
customers["city"] = customers["city"].str.strip().str.title()

# Gender: make everything Male / Female / Unknown
customers["gender"] = customers["gender"].str.strip().str.lower()
customers["gender"] = customers["gender"].replace({"m": "male", "f": "female"})
customers["gender"] = customers["gender"].fillna("unknown")

# Age: impossible values become missing, then filled with the median
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = None
customers["age"] = customers["age"].fillna(customers["age"].median())

# Delivery minutes: zero or negative becomes missing, then completed orders get the median
orders.loc[orders["delivery_minutes"] <= 0, "delivery_minutes"] = None
fix_time = orders.loc[orders["status"] == "completed", "delivery_minutes"].median()
mask2 = (orders["status"] == "completed") & (orders["delivery_minutes"].isnull())
orders.loc[mask2, "delivery_minutes"] = fix_time

# Check again
print("Cities:", customers["city"].unique())
print("Gender values:", customers["gender"].unique())
print("\nGender Counts:")
print(customers["gender"].value_counts())
print("Age range:", customers["age"].min(), "to", customers["age"].max())
print("Delivery minutes range:", orders["delivery_minutes"].min(), "to", orders["delivery_minutes"].max())
print("Orders:", len(orders), "| Customers:", len(customers))
# Step 6: save cleaned data
import os

os.makedirs("data/clean", exist_ok=True)

orders.to_csv(
    "data/clean/orders.csv",
    index=False
)

customers.to_csv(
    "data/clean/customers.csv",
    index=False
)

products.to_csv(
    "data/clean/products.csv",
    index=False
)
print("\nCleaned files saved in the data folder.")
# Create SQLite database
conn = sqlite3.connect("data/instamart.db")

customers.to_sql(
    "customers",
    conn,
    if_exists="replace",
    index=False
)

products.to_sql(
    "products",
    conn,
    if_exists="replace",
    index=False
)

orders.to_sql(
    "orders",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Database created: data/instamart.db")
# Check database tables
conn = sqlite3.connect("data/instamart.db")

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)

print("\nDatabase Tables:")
print(tables)

conn.close()
print("\nOrder Status Counts:")
print(orders["status"].value_counts())
print("Orders:", len(orders), "| Customers:", len(customers))
revenue = orders.loc[
    orders["status"] == "completed",
    "order_amount"
].sum()

print("\nCompleted Revenue:", revenue)
print("\nMissing Order Amounts:")
print(orders["order_amount"].isna().sum())
