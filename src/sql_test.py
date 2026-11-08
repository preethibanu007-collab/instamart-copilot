import sqlite3
import pandas as pd

conn = sqlite3.connect("data/instamart.db")

query = """
WITH customer_spend AS (
    SELECT
        c.city,
        o.customer_id,
        ROUND(SUM(o.order_amount),2) AS total_spend
    FROM orders o
    JOIN customers c
      ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)

SELECT
    city,
    customer_id,
    total_spend,

    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS rnk,

    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_rnk,

    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS row_num

FROM customer_spend
ORDER BY city, total_spend DESC
"""

df = pd.read_sql_query(query, conn)

print(df.head())
print("\nRows:", len(df))

conn.close()