import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.email_features import (
    EMAIL_FEATURE_COLUMNS,
    extract_features_from_email,
    EMAIL_SENDER_FEATURE_COLUMNS,
    extract_features_from_sender,
)

SEED_PATH = os.path.join(BASE_DIR, "data", "email_seeds.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "phishing_email_dataset_20000.csv")
MODELS_PATH = os.path.join(BASE_DIR, "models", "email_saved_models.pkl")
X_TRAIN_PATH = os.path.join(BASE_DIR, "models", "email_X_train.pkl")


def build_dataset_from_emails() -> pd.DataFrame:
    seeds = pd.read_csv(SEED_PATH)
    rows = []

    for _, row in seeds.iterrows():
        # Build sender-only feature set so models can predict from email address alone
        features = extract_features_from_sender(row["sender"])
        features["label"] = int(row["label"])
        rows.append(features)

        # No augmentation for sender-only features

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

    df = build_dataset_from_emails()
    df.to_csv(OUTPUT_PATH, index=False)

    # Train on sender-only feature columns
    X = df[EMAIL_SENDER_FEATURE_COLUMNS]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipelines
    pipelines = {
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
            ("model", XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)),
        ]),
    }

    param_distributions = {
        "RandomForest": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__class_weight": [None, "balanced"],
        },
        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
            "model__class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [3, 6, 10],
            "model__learning_rate": [0.01, 0.05, 0.1],
        },
    }

    best_models = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for name, pipeline in pipelines.items():
        print(f"Tuning {name}...")
        params = param_distributions.get(name, {})
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=8,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        try:
            search.fit(X_train, y_train)
            best = search.best_estimator_
        except Exception:
            pipeline.fit(X_train, y_train)
            best = pipeline

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
    print("Email models saved!")


if __name__ == "__main__":
    main()
