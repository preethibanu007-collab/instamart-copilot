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
<<PART A: to be filled in - cleaning rules, SQLite tables, query results>>

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