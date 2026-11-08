import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

rfm = pd.read_csv("outputs/rfm.csv")
cols = ["recency", "frequency", "monetary"]

X = StandardScaler().fit_transform(rfm[cols])
model = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm["segment"] = model.fit_predict(X)

rfm.to_csv("outputs/segments.csv", index=False)

print("Segment sizes:")
print(rfm["segment"].value_counts().sort_index())
print("\nAverage RFM per segment:")
print(rfm.groupby("segment")[cols].mean().round(1))