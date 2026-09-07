import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.features import FEATURE_COLUMNS, extract_features_from_url

SEED_PATH = os.path.join(BASE_DIR, "data", "url_seeds.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "phishing_dataset_20000.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models", "saved_models.pkl")
X_TRAIN_PATH = os.path.join(BASE_DIR, "models", "X_train.pkl")


def build_dataset_from_urls() -> pd.DataFrame:
    seeds = pd.read_csv(SEED_PATH)
    rows = []

    for _, row in seeds.iterrows():
        features = extract_features_from_url(row["url"])
        features["label"] = int(row["label"])
        rows.append(features)

        url = row["url"]
        variants = []
        if url.startswith("https://"):
            variants.append(url.replace("https://", "http://", 1))
        if "www." in url:
            variants.append(url.replace("www.", ""))
        elif "://" in url and "www." not in url:
            parts = url.split("://", 1)
            variants.append(f"{parts[0]}://www.{parts[1]}")

        for variant in variants:
            variant_features = extract_features_from_url(variant)
            variant_features["label"] = int(row["label"])
            rows.append(variant_features)

    df = pd.DataFrame(rows)

    legit = df[df["label"] == 0]
    phishing = df[df["label"] == 1]
    target_size = 20000

    legit_samples = legit.sample(target_size // 2, replace=True, random_state=42)
    phishing_samples = phishing.sample(target_size // 2, replace=True, random_state=42)
    balanced = pd.concat([legit_samples, phishing_samples], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced


def main():
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    df = build_dataset_from_urls()
    df.to_csv(OUTPUT_PATH, index=False)

    X = df[FEATURE_COLUMNS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter distributions for randomized search
    param_distributions = {
        "RandomForest": {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10],
            "class_weight": [None, "balanced"],
        },
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear"],
            "class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "scale_pos_weight": [1, max(1, int(y.value_counts().get(0,1) / max(1, y.value_counts().get(1,1))))],
        },
    }

    base_models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42),
    }

    best_models = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, model in base_models.items():
        print(f"Tuning {name}...")
        params = param_distributions.get(name, {})
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=5,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=1,
            verbose=0,
        )

        # Fit the search on the training set
        try:
            search.fit(X_train, y_train)
            best = search.best_estimator_
        except Exception:
            # Fallback: fit the base model if search fails
            model.fit(X_train, y_train)
            best = model

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
        print(f"{name} best params: {getattr(search, 'best_params_', {})}")
        print(classification_report(y_test, y_pred, digits=4))
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                print(f"ROC AUC: {auc:.4f}")
            except Exception:
                pass

        best_models[name] = best

    joblib.dump(best_models, MODELS_PATH)
    joblib.dump(X_train, X_TRAIN_PATH)
    print("Models saved!")


if __name__ == "__main__":
    main()
