import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

rfm = pd.read_csv("outputs/rfm.csv")

scaler = StandardScaler()

X = scaler.fit_transform(
    rfm[["recency", "frequency", "monetary"]]
)

for k in range(2, 9):

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X)

    print(
        "K =", k,
        "Inertia =", round(model.inertia_, 1),
        "Silhouette =", round(
            silhouette_score(X, labels),
            3
        )
    )