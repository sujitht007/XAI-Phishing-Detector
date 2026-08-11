import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.phone_features import PHONE_FEATURE_COLUMNS, extract_features_from_phone

SEED_PATH = os.path.join(BASE_DIR, "data", "phone_seeds.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "phishing_phone_dataset_20000.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models", "phone_saved_models.pkl")
X_TRAIN_PATH = os.path.join(BASE_DIR, "models", "phone_X_train.pkl")


def build_dataset_from_phones() -> pd.DataFrame:
    seeds = pd.read_csv(SEED_PATH)
    rows = []

    for _, row in seeds.iterrows():
        features = extract_features_from_phone(row["phone"])
        features["label"] = int(row["label"])
        rows.append(features)

        raw = row["phone"]
        variants = [
            raw.replace(" ", "-"),
            raw.replace("+", ""),
            raw.replace("-", " "),
            f"{raw} ext 123",
        ]
        for variant in variants:
            variant_features = extract_features_from_phone(variant)
            variant_features["label"] = int(row["label"])
            rows.append(variant_features)

    df = pd.DataFrame(rows)
    legit = df[df["label"] == 0]
    phishing = df[df["label"] == 1]
    target_size = 20000

    legit_samples = legit.sample(target_size // 2, replace=True, random_state=42)
    phishing_samples = phishing.sample(target_size // 2, replace=True, random_state=42)
    balanced = pd.concat([legit_samples, phishing_samples], ignore_index=True)
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    df = build_dataset_from_phones()
    df.to_csv(OUTPUT_PATH, index=False)

    X = df[PHONE_FEATURE_COLUMNS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(eval_metric="logloss", random_state=42)),
        ]),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"{name} trained (test accuracy: {accuracy:.3f})")

    joblib.dump(models, MODELS_PATH)
    joblib.dump(X_train, X_TRAIN_PATH)
    print("Phone models saved!")


if __name__ == "__main__":
    main()
