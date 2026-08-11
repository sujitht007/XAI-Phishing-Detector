import os
import sys

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.evaluation import compare_explainability_methods, compare_ml_models
from src.features import FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models", "saved_models.pkl")
X_TRAIN_PATH = os.path.join(BASE_DIR, "models", "X_train.pkl")

models = joblib.load(MODELS_PATH)
x_train = joblib.load(X_TRAIN_PATH)

sample = {
    "url_length": 120,
    "has_at_symbol": 1,
    "has_https": 0,
    "num_dots": 5,
    "has_ip": 1,
    "domain_age_days": 100,
    "has_suspicious_words": 1,
    "num_subdomains": 3,
    "is_trusted_tld": 0,
}

x = pd.DataFrame([sample])

model_results, best_model = compare_ml_models(models, x)
print("ML Algorithm Comparison:")
for item in model_results:
    print(
        f"  {item['name']} -> Robustness: {item['robustness']}, "
        f"Complexity: {item['complexity']}, Confidence: {item['confidence']}, "
        f"Score: {item['score']}, Prediction: {item['prediction']}"
    )
print(f"\nBest ML algorithm: {best_model['name']}")

explainer_results, best_explainer = compare_explainability_methods(
    models[best_model["name"]],
    x_train,
    x,
)
print("\nExplainability Method Comparison:")
for item in explainer_results:
    print(
        f"  {item['name']} -> Robustness: {item['robustness']}, "
        f"Complexity: {item['complexity']}, Score: {item['score']}"
    )
print(f"\nBest explainability method: {best_explainer['name']}")
