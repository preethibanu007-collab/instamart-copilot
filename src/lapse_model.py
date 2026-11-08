import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

df = pd.read_csv("outputs/lapse_features.csv")
y = df["lapsed"]
X = df.drop(columns=["lapsed", "customer_id"])

num = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate",
       "avg_delivery", "age", "tenure_days"]
cat = ["city", "gender", "fav_category"]
X["gender"] = X["gender"].fillna("unknown")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("train/test:", len(Xtr), len(Xte), "| lapsed:", int(y.sum()), f"({y.mean()*100:.2f}%)")

base = np.zeros(len(yte), dtype=int)
print("baseline acc:", round(accuracy_score(yte, base), 3),
      "F1:", f1_score(yte, base, zero_division=0))

def prep():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat),
    ])

def report(name, model):
    p = model.predict(Xte)
    print(f"\n{name}\n", confusion_matrix(yte, p))
    print("precision", round(precision_score(yte, p, zero_division=0), 3),
          "recall", round(recall_score(yte, p, zero_division=0), 3),
          "F1", round(f1_score(yte, p, zero_division=0), 3))

lr = GridSearchCV(
    Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000))]),
    {"m__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr.fit(Xtr, ytr)
print("\nLR best:", lr.best_params_, "CV F1:", round(lr.best_score_, 3))

rf = GridSearchCV(
    Pipeline([("prep", prep()),
              ("m", RandomForestClassifier(n_estimators=100, random_state=42))]),
    {"m__max_depth": [3, 5, 8], "m__min_samples_leaf": [5, 10, 20]},
    cv=5, scoring="f1")
rf.fit(Xtr, ytr)
print("RF best:", rf.best_params_, "CV F1:", round(rf.best_score_, 3))

report("Logistic Regression (test)", lr.best_estimator_)
report("Random Forest (test)", rf.best_estimator_)

# selected model: random forest (higher test F1 expected)
os.makedirs("outputs", exist_ok=True)
joblib.dump(rf.best_estimator_, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
print("\nreloaded predictions identical:",
      bool((re.predict(Xte) == rf.best_estimator_.predict(Xte)).all()))