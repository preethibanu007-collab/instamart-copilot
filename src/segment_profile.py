import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rfm = pd.read_csv("outputs/rfm.csv")
cols = ["recency", "frequency", "monetary"]

X = StandardScaler().fit_transform(rfm[cols])
rfm["segment_id"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)

m = rfm.groupby("segment_id")[cols].mean()
names = {}
premium = m["monetary"].idxmax()
names[premium] = "Premium High-Spenders"
rest = m.drop(index=premium)
loyal = rest["frequency"].idxmax()
names[loyal] = "Frequent Loyalists"
rest = rest.drop(index=loyal)
dormant = rest["recency"].idxmax()
names[dormant] = "Lapsing Customers"
for s in rest.index:
    if s != dormant:
        names[s] = "Occasional Buyers"

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = (rfm["segment_id"] == premium).astype(int)

out_cols = ["customer_id"] + cols + ["segment_id", "segment_name", "is_premium"]
rfm[out_cols].to_csv("outputs/customer_segments.csv", index=False)

print(rfm.groupby(["segment_id", "segment_name"]).size())
print(m.round(1))
pca = PCA(n_components=2).fit(X)
print("PCA ratios:", pca.explained_variance_ratio_.round(3),
      "total:", pca.explained_variance_ratio_.sum().round(3))