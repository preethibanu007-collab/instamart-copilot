import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("outputs/lapse_features.csv")
df["gender"] = df["gender"].fillna("unknown")
y = df["lapsed"]
X = df.drop(columns=["lapsed"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
ids_te = Xte["customer_id"].values
Xtr = Xtr.drop(columns="customer_id")
Xte = Xte.drop(columns="customer_id")

num = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
       "avg_delivery", "age", "tenure_days"]
cat = ["city", "gender", "fav_category"]

def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])

def lr_model():
    return Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000, C=10))])

def scores(m):
    p = m.predict(Xte)
    return (round(precision_score(yte, p, zero_division=0), 3),
            round(recall_score(yte, p, zero_division=0), 3),
            round(f1_score(yte, p, zero_division=0), 3))

plain = lr_model().fit(Xtr, ytr)
print("LR unbalanced (P,R,F1):", scores(plain))

# oversample lapsed rows in TRAIN only
rng_idx = ytr[ytr == 1].sample(n=int((ytr == 0).sum() - (ytr == 1).sum()),
                               replace=True, random_state=42).index
Xb = pd.concat([Xtr, Xtr.loc[rng_idx]])
yb = pd.concat([ytr, ytr.loc[rng_idx]])
print("balanced train counts:", yb.value_counts().to_dict())
bal = lr_model().fit(Xb, yb)
print("LR balanced   (P,R,F1):", scores(bal))

# interpretation
rf = joblib.load("outputs/lapse_model.joblib")
names = rf.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf.named_steps["m"].feature_importances_, index=names)
print("\nTop 5 RF features:\n", imp.sort_values(ascending=False).head(5).round(3))
coef = pd.Series(plain.named_steps["m"].coef_[0],
                 index=plain.named_steps["prep"].get_feature_names_out())
for k in ["num__recency", "num__frequency", "num__cancel_rate"]:
    print(k, "coef:", round(coef[k], 3))

# retention list (Premium customers in the test set)
seg = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "is_premium"]]
t = pd.DataFrame({"customer_id": ids_te,
                  "lapse_probability": rf.predict_proba(Xte)[:, 1],
                  "actually_lapsed": yte.values}).merge(seg, on="customer_id")
t = t[t["is_premium"] == 1].sort_values("lapse_probability", ascending=False).head(10)
t[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(
    "outputs/retention_list.csv", index=False)
print("\nretention list rows:", len(t), "| truly lapsed:", int(t["actually_lapsed"].sum()))