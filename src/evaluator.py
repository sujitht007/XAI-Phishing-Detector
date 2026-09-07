import os

import joblib
import pandas as pd
from collections import Counter

from src.evaluation import compare_explainability_methods, compare_ml_models
from src.explainability import ExplainerCache


def run_evaluation(
    models: dict,
    x_train: pd.DataFrame,
    explainer_cache: ExplainerCache,
    features: dict,
    feature_columns: list[str],
    input_label: str,
    input_value: str,
    trusted_override: bool,
    trusted_note: str,
    input_type: str,
):
    x = pd.DataFrame([features], columns=feature_columns)

    model_results, best_model = compare_ml_models(models, x)

    # For email inputs prefer the majority prediction when at least two models agree.
    if input_type.lower() == "email":
        preds = [item["prediction"] for item in model_results]
        if preds:
            most_common, cnt = Counter(preds).most_common(1)[0]
            if cnt >= 2:
                # choose best model among those that made the majority prediction
                candidate = max((item for item in model_results if item["prediction"] == most_common), key=lambda i: i["score"])
                best_model = {
                    **candidate,
                    "prediction": most_common,
                    "confidence": candidate["confidence"],
                }
                ensemble_note = " Ensemble majority-vote applied for email inputs."

    heuristic_flag = False
    heuristic_note = ""
    ensemble_note = ""
    if input_type.lower() == "url":
        from src.features import is_clearly_phishing, resolve_url_prediction

        if is_clearly_phishing(features):
            heuristic_flag = True
            heuristic_note = (
                " Heuristic override: malformed URL, unknown institutional domain, "
                "or suspicious path on a trusted-looking TLD."
            )
        else:
            prediction, confidence, best_model = resolve_url_prediction(model_results, features)
            best_model = {
                **best_model,
                "prediction": prediction,
                "confidence": round(confidence, 3),
            }
            ensemble_note = (
                " Ensemble decision applied to reduce false positives on well-formed URLs."
                if prediction == "Legit" and any(item["prediction"] == "Phishing" for item in model_results)
                else ""
            )

    if heuristic_flag:
        for item in model_results:
            item["prediction"] = "Phishing"
            item["confidence"] = max(item["confidence"], 0.95)
        best_model = {
            **best_model,
            "prediction": "Phishing",
            "confidence": max(best_model["confidence"], 0.95),
        }

    if trusted_override:
        best_model = {
            **best_model,
            "prediction": "Legit",
            "confidence": max(best_model["confidence"], 0.95),
        }
        for item in model_results:
            if item["name"] == best_model["name"]:
                item["prediction"] = "Legit"
                item["confidence"] = best_model["confidence"]

    reference_model = models[best_model["name"]]
    explanation_results, best_explainer = compare_explainability_methods(
        reference_model,
        x_train,
        x,
        cache=explainer_cache,
        model_name=best_model["name"],
    )

    explanation = (
        f"For this {input_type}, {best_model['name']} is the best ML algorithm "
        f"(score: {best_model['score']}, prediction: {best_model['prediction']}, "
        f"confidence: {best_model['confidence']}). "
        f"{best_explainer['name']} is the best explainability method "
        f"(score: {best_explainer['score']}, robustness: {best_explainer['robustness']}, "
        f"complexity: {best_explainer['complexity']}s)."
        f"{trusted_note}{heuristic_note}{ensemble_note}"
    )

    return {
        "analysis_type": input_type,
        "input_label": input_label,
        "input_value": input_value,
        "features": features,
        "feature_columns": feature_columns,
        "best_model": best_model["name"],
        "best_prediction": best_model["prediction"],
        "best_confidence": best_model["confidence"],
        "model_results": [
            {
                "name": item["name"],
                "robustness": item["robustness"],
                "complexity": item["complexity"],
                "confidence": item["confidence"],
                "score": item["score"],
                "prediction": item["prediction"],
            }
            for item in model_results
        ],
        "best_explainer": best_explainer["name"],
        "explainer_results": [
            {
                "name": item["name"],
                "robustness": item["robustness"],
                "complexity": item["complexity"],
                "score": item["score"],
                "top_features": item["top_features"],
            }
            for item in explanation_results
        ],
        "selected_top_features": best_explainer["top_features"],
        "explanation": explanation,
        "trusted_override": trusted_override,
    }


def load_model_bundle(base_dir: str, prefix: str = ""):
    name = f"{prefix}saved_models.pkl" if prefix else "saved_models.pkl"
    train_name = f"{prefix}X_train.pkl" if prefix else "X_train.pkl"
    models_path = os.path.join(base_dir, "models", name)
    x_train_path = os.path.join(base_dir, "models", train_name)
    models = joblib.load(models_path)
    x_train = joblib.load(x_train_path)
    cache = ExplainerCache(models, x_train)
    return models, x_train, cache
