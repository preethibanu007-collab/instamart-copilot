import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

df = pd.read_csv("outputs/lapse_features.csv")
y = df["lapsed"]
Xtr, Xte, ytr, yte = train_test_split(df.drop(columns=["lapsed"]), y,
                                      test_size=0.2, stratify=y, random_state=42)
num = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
       "avg_delivery", "age", "tenure_days"]
imp = SimpleImputer(strategy="median").fit(Xtr[num])
sc = StandardScaler().fit(imp.transform(Xtr[num]))
print("scaler mean frequency (train only):", round(sc.mean_[num.index("frequency")], 3),
      "| full cohort mean:", round(df["frequency"].mean(), 3))
miss = (pd.DataFrame(imp.transform(Xtr[num])).isna().sum().sum()
        + pd.DataFrame(imp.transform(Xte[num])).isna().sum().sum()
        + df[["city", "fav_category"]].isna().sum().sum())
print("missing values left in features after imputation:", int(miss))