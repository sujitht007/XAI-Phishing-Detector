import os
import tempfile

import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluator import run_evaluation
from src.explainability import ExplainerCache


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'phishing_email_dataset_20000.csv')


def load_sample(n=500):
    df = pd.read_csv(DATA_PATH)
    if len(df) > n:
        df = df.sample(n, random_state=42)
    X = df.drop(columns=['label'])
    y = df['label']
    return X, y


def test_train_and_predict_quick():
    X, y = load_sample(300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    # Quick sanity checks
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    probs = model.predict_proba(X_test)
    assert probs.shape[0] == len(X_test)


def test_run_evaluation_pipeline():
    X, y = load_sample(300)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    models = {"TestLR": model}

    # Use first test row as example features
    sample_row = X_test.iloc[0]
    features = sample_row.to_dict()
    feature_columns = list(X.columns)

    # Build explainer cache (small)
    cache = ExplainerCache(models, X_train)

    result = run_evaluation(
        models=models,
        x_train=X_train,
        explainer_cache=cache,
        features=features,
        feature_columns=feature_columns,
        input_label="Sender",
        input_value="tester@example.com",
        trusted_override=False,
        trusted_note="",
        input_type="email",
    )

    assert isinstance(result, dict)
    assert 'best_prediction' in result
    assert result['best_prediction'] in ('Phishing', 'Legit')
    assert 'best_confidence' in result
    assert isinstance(result['best_confidence'], float)
