# Instamart Customer Intelligence & Support Copilot

All data is synthetic. All model calls use the offline `mock_llm.py`. No API key, network or paid service is used.

## Reproduce
Python 3.11+. From the repository root, run in order:
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python src/clean.py
python src/validate.py
python src/sql_test.py
python src/rfm.py
python src/clustering.py
python src/segment_profile.py
python src/lapse_data.py
python src/lapse_model.py
python src/lapse_extra.py
python src/lapse_check.py
python src/triage.py
python src/rag.py
python src/agent.py
```
`generate_data.py` must print `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `generate_data.py` and `mock_llm.py` are used unmodified.

## Part A - Data Foundation
### Setup
Repository on branch `main`; virtual environment created with `python -m venv venv`; libraries installed with pip and written with `pip freeze` to `requirements.txt`; `.gitignore` lists `venv/`, `.env`, `__pycache__/`, `*.pyc` and `charts/`. The two supplied files were copied unmodified and `python generate_data.py` was run (see Reproduce).

### Audit of the raw data
Shapes: orders (13084, 9), customers (1200, 5), products (40, 4). Missing values: `orders.delivery_minutes` 2,541 (19.42%); `customers.age` 61 (5.08%); `customers.gender` 30 (2.50%); every other column has 0 missing. `order_amount` is read as text in the raw file because some entries are not numbers.

| audit figure | value |
|---|---|
| raw orders rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id | 12,309 |
| order_amount entries that cannot be converted to a number | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| blank age | 61 |
| age outside 10-100 | 6 |
| distinct raw city spellings | 24 |
| distinct non-blank raw gender spellings | 10 |
| blank gender | 30 |

### Cleaning rules and what each one changed
Customers:
1. City: leading/trailing spaces removed and title case applied; 24 raw spellings became 6 cities.
2. Gender: all 10 raw spellings mapped to `male` or `female`; the 30 blank values became `unknown`.
3. Age: the 6 values outside 10-100 were set to missing; together with the 61 blank ages, 67 missing ages were filled with the median age of the customer's city (groupby + transform). Result: 0 missing ages, all between 18 and 60.
4. `signup_date` converted to a datetime.

Orders, in this order:
1. `order_ts` converted to a datetime.
2. Exact duplicate rows removed: 119 rows (13,084 to 12,965).
3. For each `order_id`, only the record with the latest `order_ts` was kept. This removed the 656 earlier `pending` records (12,965 to 12,309 rows, one per order).
4. `order_amount` converted to numeric; the 120 unparseable raw entries became missing and were filled from the product price: `round(price_inr x quantity x (1 - discount_pct / 100), 2)`. Result: 0 missing amounts.
5. The 106 negative `delivery_minutes` were set to missing. For completed orders, every missing delivery time was filled with the median delivery time of that order's product category among completed orders (groupby + transform). For cancelled orders it was left missing, because a cancelled order was never delivered and a filled value would be invented data.

After cleaning: 12,309 rows; 11,110 completed and 1,199 cancelled; no pending rows; 0 missing `order_amount`; 0 missing `delivery_minutes` among completed orders and 1,199 among cancelled orders; completed order_amount sums to INR 2,998,668.80. Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Genders: female 610, male 560, unknown 30.

### Storage and validation
The clean tables are written to `data/clean/` as CSV files and to `data/instamart.db` as tables `customers`, `products` and `orders` (`order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/validate.py` asserts 14 reference checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query` in `src/part_a_report.py`)
| query | technique | result |
|---|---|---|
| Q1 | WHERE then HAVING | 30 customers with completed orders over INR 500 whose total exceeds INR 8,000 |
| Q2 | three-table join | 30 city/category rows; revenues add up to INR 2,998,668.80; largest cell Bengaluru / Electronics, INR 233,909.45 |
| Q3 | scalar subquery | 11 products priced above the average catalogue price |
| Q4 | LEFT JOIN | 25 customers with no completed order |
| Q5 | CTE | 550 customers above the average total spend of INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER | top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 | running total | 183 daily rows; last cumulative revenue INR 2,998,668.80 |

In Q6, customers with exactly the same total spend in a city share one RANK, but ROW_NUMBER still gives each of them a different number, so the two columns differ for tied customers and for the rows after them.

### pandas cross-check
`pd.pivot_table(..., margins=True)` of revenue by city and category has a grand total of INR 2,998,668.80, equal to the SQL total. Completed order_amount by city:

| city | sum (INR) | mean (INR) | count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

### Exploratory analysis of completed order_amount
Mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00; the 1.5 x IQR fences are -283.50 and 645.30, with 0 outliers below and 897 above.
- **What is the data about?** Six-month quick-commerce orders (April to September 2025) from 1,200 customers in six Indian cities, with amounts in INR.
- **Distributions and statistics:** order values are strongly right-skewed: the mean (269.91) is about twice the median (135.00), because many small grocery baskets sit beside a few expensive electronics orders.
- **Outliers and anomalies:** 897 orders exceed the upper fence of INR 645.30; they are mostly real high-value electronics purchases rather than errors. The raw data had planted problems (duplicates, pending records, text amounts, negative delivery times, impossible ages), all handled above.
- **Relationships:** the largest correlation in absolute value is order_amount with discount_pct at about -0.095 (order_amount with quantity is +0.087). Correlations this small are very weak, and correlation of this size does not establish causation.

## Part B - Customer Segments and Lapse Prediction

### RFM
Snapshot 2025-10-01. 1,175 customers have at least one completed order (the other 25 have none and are not segmented).

### K-means
Features standardised with StandardScaler; KMeans(n_init=10, random_state=42).
| K | inertia | silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Elbow: inertia falls steeply up to K=4 (1268 to 743) and much more slowly afterwards (743 to 617 to 503).

**K = 4 chosen.** It has the highest silhouette (0.526), and inertia drops sharply from K=3 (1268) to K=4 (743), then flattens (K=5: 617). Four segments are also easy for the business to act on.
The inertia of K=8 is always lower than that of K=2 because more clusters always shrink the distances to the nearest centre, even when they only split a real group in two. So lower inertia alone cannot pick K; we need the elbow, the silhouette score and business meaning.

### Segment profile (cluster numbers are only labels, with no ordering)
| segment_id | name | size | mean recency (days) | mean frequency | mean monetary (INR) | action |
|---|---|---|---|---|---|---|
| 0 | Occasional Buyers | 436 | 28.9 | 3.8 | 1,057.6 | Light promotions to raise order frequency |
| 1 | Premium High-Spenders (Premium) | 138 | 24.1 | 6.8 | 7,129.7 | Early access to electronics offers, priority support |
| 2 | Frequent Loyalists | 419 | 8.8 | 19.4 | 3,410.0 | Loyalty rewards, protect with fast delivery |
| 3 | Lapsing Customers | 182 | 109.0 | 2.2 | 686.0 | Win-back discounts before they are lost |

Premium is the segment with the highest mean monetary value (segment 1).
PCA (2 components) on the scaled RFM matrix: explained variance 0.634 and 0.218, total 0.852.
Output: `outputs/customer_segments.csv`.

### Lapse problem
Cutoff 2025-09-01. Cohort: customers with at least one completed order before the cutoff = 1,156. `lapsed` = 1 if there is no completed order from 2025-09-01 to 2025-09-30: 443 lapsed (38.32%). Features use only orders before the cutoff plus the clean customer table: recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure days. Split: `train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 train and 232 test rows.
Baseline (always "not lapsed") on the test set: accuracy 0.616, lapsed-class F1 0.

### Preprocessing without leakage
A Pipeline with a median imputer and StandardScaler for numeric columns and OneHotEncoder(drop="first") for city, gender and favourite category. Every encoder and scaler is fitted on training rows only. The scaler's learned mea