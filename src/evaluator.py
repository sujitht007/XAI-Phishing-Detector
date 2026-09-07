
import os
import time
from collections import Counter
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from src.evaluation import (
    attribution_similarity,
    compute_explanation_score,
    compute_model_score,
)
from src.explainability import (
    ExplainerCache,
    available_explainers,
    generate_explanation,
)


def model_robustness_score(
    model,
    x: pd.DataFrame,
    trials: int = 2,
    noise_scale: float = 0.01,
) -> float:
    """
    Check whether the model gives the same prediction
    when a small amount of noise is added.
    """

    base_pred = int(model.predict(x)[0])
    stable = 0

    for _ in range(trials):
        noise = np.random.normal(
            0,
            noise_scale,
            x.values.shape,
        )

        noisy_sample = pd.DataFrame(
            x.values + noise,
            columns=x.columns,
        )

        try:
            noisy_pred = int(model.predict(noisy_sample)[0])

            if noisy_pred == base_pred:
                stable += 1

        except Exception:
            # If a model cannot handle the perturbed input,
            # don't allow one failure to crash the whole request.
            pass

    return (
        stable / trials
        if trials > 0
        else 1.0
    )


def model_complexity_score(
    model,
    x: pd.DataFrame,
) -> float:
    """Measure prediction time for one sample."""

    start = time.time()

    model.predict(x)

    return time.time() - start


def compare_ml_models(
    models: dict,
    x: pd.DataFrame,
) -> Tuple[list, dict]:
    """
    Compare all available ML models.

    Each model is evaluated using:
    - robustness
    - prediction complexity
    - confidence
    """

    results = []

    if not models:
        raise ValueError("No ML models available.")

    for name, model in models.items():

        try:
            robustness = model_robustness_score(
                model,
                x,
                trials=2,
            )

            complexity = model_complexity_score(
                model,
                x,
            )

            prediction = int(
                model.predict(x)[0]
            )

            probabilities = model.predict_proba(x)[0]

            confidence = float(
                np.max(probabilities)
            )

            results.append(
                {
                    "name": name,
                    "robustness": round(
                        float(robustness),
                        3,
                    ),
                    "complexity": round(
                        float(complexity),
                        3,
                    ),
                    "confidence": round(
                        confidence,
                        3,
                    ),
                    "prediction": (
                        "Phishing"
                        if prediction == 1
                        else "Legit"
                    ),
                }
            )

        except Exception as exc:
            print(
                f"Model '{name}' failed during prediction: {exc}"
            )

    if not results:
        raise RuntimeError(
            "All ML models failed during prediction."
        )

    max_complexity = (
        max(
            item["complexity"]
            for item in results
        )
        or 1.0
    )

    for item in results:

        item["score"] = round(
            compute_model_score(
                item["robustness"],
                item["complexity"],
                item["confidence"],
                max_complexity,
            ),
            3,
        )

    best = max(
        results,
        key=lambda item: item["score"],
    )

    return results, best


def select_best_explainer(
    results: list,
) -> dict:
    """Return the explainability method with the highest score."""

    if not results:
        raise ValueError(
            "No explainability results available."
        )

    return max(
        results,
        key=lambda item: item["score"],
    )


def compare_explainability_methods(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache=None,
    model_name=None,
    robustness_trials: int = 2,
    methods=None,
) -> Tuple[list, dict]:
    """
    Compare explainability methods.

    Phone analysis should call this with:

        methods=["Permutation"]
        robustness_trials=0

    This avoids expensive SHAP/LIME calculations on Render.
    """

    if methods is None:
        methods = available_explainers(model)

    results = []

    if not methods:
        raise ValueError(
            "No explainability methods available."
        )

    for method_name in methods:

        try:
            start = time.time()

            attributions = generate_explanation(
                method_name,
                model,
                x_train,
                x_sample,
                cache=cache,
                model_name=model_name,
            )

            complexity = time.time() - start

            # For lightweight mode we don't perform
            # additional robustness explanation calls.
            if robustness_trials <= 0:

                robustness = 1.0

            else:

                stable = 0.0

                for _ in range(robustness_trials):

                    noise = np.random.normal(
                        0,
                        0.01,
                        x_sample.values.shape,
                    )

                    noisy_sample = pd.DataFrame(
                        x_sample.values + noise,
                        columns=x_sample.columns,
                    )

                    noisy_attrs = generate_explanation(
                        method_name,
                        model,
                        x_train,
                        noisy_sample,
                        cache=cache,
                        model_name=model_name,
                    )

                    stable += attribution_similarity(
                        attributions,
                        noisy_attrs,
                    )

                robustness = (
                    stable / robustness_trials
                )

            results.append(
                {
                    "name": method_name,
                    "robustness": round(
                        float(robustness),
                        3,
                    ),
                    "complexity": round(
                        float(complexity),
                        3,
                    ),
                    "attributions": attributions,
                }
            )

        except Exception as exc:

            print(
                f"Explainer '{method_name}' failed: {exc}"
            )

    if not results:
        raise RuntimeError(
            "All explainability methods failed."
        )

    max_complexity = (
        max(
            item["complexity"]
            for item in results
        )
        or 1.0
    )

    for item in results:

        item["score"] = round(
            compute_explanation_score(
                item["robustness"],
                item["complexity"],
                max_complexity,
            ),
            3,
        )

        ranked = sorted(
            item["attributions"].items(),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )

        item["top_features"] = [
            {
                "feature": feature,
                "importance": round(
                    float(value),
                    4,
                ),
            }
            for feature, value in ranked[:5]
        ]

    best = select_best_explainer(
        results
    )

    return results, best


def evaluate_url(url: str):
    """
    Evaluate a URL using the URL models.
    """

    from src.features import (
        FEATURE_COLUMNS,
        extract_features_from_url,
        is_clearly_legitimate,
    )

    features = extract_features_from_url(url)

    trusted_override = is_clearly_legitimate(
        features
    )

    trusted_note = (
        " This URL matches a known legitimate domain "
        "with no phishing indicators."
        if trusted_override
        else ""
    )

    return run_evaluation(
        url_models,
        url_x_train,
        url_explainer_cache,
        features,
        FEATURE_COLUMNS,
        "URL",
        url,
        trusted_override,
        trusted_note,
        "URL",
    )


def evaluate_email(sender: str):
    """
    Evaluate an email sender.

    If email ML models are unavailable, use the
    dedicated email-address heuristic pipeline.
    """

    from src.email_features import (
        EMAIL_SENDER_FEATURE_COLUMNS,
        analyze_email_address,
        extract_features_from_sender,
    )

    if email_models_loaded:

        features = extract_features_from_sender(
            sender
        )
        sender_local_typo = features.get("sender_local_typo") == 1
        trusted_override = features.get("is_trusted_sender") == 1 and not sender_local_typo

        return run_evaluation(
            email_models,
            email_x_train,
            email_explainer_cache,
            features,
            EMAIL_SENDER_FEATURE_COLUMNS,
            "Sender",
            sender,
            trusted_override,
            " Sender ID does not exactly match the verified recruitment mailbox." if sender_local_typo else "",
            "email",
            forced_prediction="Phishing" if sender_local_typo else None,
        )

    result = analyze_email_address(
        sender
    )

    label = result.get(
        "label",
        "Invalid Email Address",
    )

    prediction_map = {
        "Invalid Email Address": "Invalid",
        "Suspicious Email Address": "Suspicious",
        "Likely Legitimate Email Address": "Legit",
    }

    mapped_prediction = prediction_map.get(
        label,
        "Suspicious",
    )

    confidence = float(
        result.get(
            "confidence",
            0.0,
        )
    )

    explanation = result.get(
        "explanation",
        "",
    )

    features = result.get(
        "features",
        {},
    )

    return {
        "analysis_type": "email",
        "input_label": "Sender",
        "input_value": sender,
        "features": features,
        "feature_columns": result.get(
            "feature_columns",
            EMAIL_SENDER_FEATURE_COLUMNS,
        ),
        "best_model": "EmailAddressPipeline",
        "best_prediction": mapped_prediction,
        "best_confidence": confidence,
        "model_results": [],
        "best_explainer": "",
        "explainer_results": [],
        "selected_top_features": [],
        "explanation": explanation,
        "trusted_override": (
            mapped_prediction == "Legit"
        ),
    }


def evaluate_phone(phone_number: str):
    """
    Evaluate a phone number.

    Phone analysis deliberately uses a lightweight
    explainability pipeline to prevent Render worker
    timeout / memory problems.
    """

    from src.phone_features import (
        PHONE_FEATURE_COLUMNS,
        extract_features_from_phone,
        is_clearly_legitimate_phone,
        is_valid_phone,
    )

    features = extract_features_from_phone(
        phone_number
    )

    # -------------------------------------------------
    # STEP 1: Validate phone format
    # -------------------------------------------------

    if not is_valid_phone(phone_number):

        return {
            "analysis_type": "phone",
            "input_label": "Phone",
            "input_value": phone_number,
            "features": features,
            "feature_columns": PHONE_FEATURE_COLUMNS,
            "best_model": "Heuristic",
            "best_prediction": "Phishing",
            "best_confidence": 0.99,
            "model_results": [],
            "best_explainer": "",
            "explainer_results": [],
            "selected_top_features": [],
            "explanation": (
                "Input rejected: phone number format is "
                "invalid. Marked as Phishing by heuristic."
            ),
            "trusted_override": False,
        }

    # -------------------------------------------------
    # STEP 2: Legitimate phone heuristic
    # -------------------------------------------------

    trusted_override = is_clearly_legitimate_phone(
        features
    )

    trusted_note = (
        " This phone number looks like a valid "
        "international contact number with no "
        "phishing flags, so the system treats it "
        "as legitimate."
        if trusted_override
        else ""
    )

    # -------------------------------------------------
    # STEP 3: Lightweight ML + XAI
    # -------------------------------------------------

    try:

        result = run_evaluation(
            phone_models,
            phone_x_train,
            phone_explainer_cache,
            features,
            PHONE_FEATURE_COLUMNS,
            "Phone",
            phone_number,
            trusted_override,
            trusted_note,
            "phone",
        )

        return result

    except Exception as exc:

        print(
            f"Phone ML evaluation failed: {exc}"
        )

        # If the ML pipeline fails, return a safe
        # heuristic result instead of crashing Flask.
        return {
            "analysis_type": "phone",
            "input_label": "Phone",
            "input_value": phone_number,
            "features": features,
            "feature_columns": PHONE_FEATURE_COLUMNS,
            "best_model": "Heuristic",
            "best_prediction": (
                "Legit"
                if trusted_override
                else "Suspicious"
            ),
            "best_confidence": 0.80,
            "model_results": [],
            "best_explainer": "Rule-based",
            "explainer_results": [],
            "selected_top_features": [],
            "explanation": (
                "The phone number passed basic validation, "
                "but the ML analysis was unavailable. "
                "A lightweight heuristic assessment was "
                "used instead."
            ),
            "trusted_override": trusted_override,
        }


def run_evaluation(
    models: dict,
    x_train: pd.DataFrame,
    explainer_cache,
    features: dict,
    feature_columns: list,
    input_label: str,
    input_value: str,
    trusted_override: bool,
    trusted_note: str,
    input_type: str,
    forced_prediction: str | None = None,
):
    """
    Main evaluation pipeline.
    """

    x = pd.DataFrame(
        [features],
        columns=feature_columns,
    )

    # -------------------------------------------------
    # STEP 1: Compare ML models
    # -------------------------------------------------

    model_results, best_model = compare_ml_models(
        models,
        x,
    )

    ensemble_note = ""

    # -------------------------------------------------
    # STEP 2: Email majority voting
    # -------------------------------------------------

    if input_type.lower() == "email":

        predictions = [
            item["prediction"]
            for item in model_results
        ]

        if predictions:

            most_common, count = (
                Counter(predictions)
                .most_common(1)[0]
            )

            if count >= 2:

                candidates = [
                    item
                    for item in model_results
                    if item["prediction"]
                    == most_common
                ]

                candidate = max(
                    candidates,
                    key=lambda item: item["score"],
                )

                best_model = {
                    **candidate,
                    "prediction": most_common,
                    "confidence": candidate[
                        "confidence"
                    ],
                }

                ensemble_note = (
                    " Ensemble majority-vote applied "
                    "for email inputs."
                )

    # -------------------------------------------------
    # STEP 3: URL-specific heuristics
    # -------------------------------------------------

    heuristic_flag = False
    heuristic_note = ""

    if input_type.lower() == "url":

        from src.features import (
            is_clearly_phishing,
            resolve_url_prediction,
        )

        if is_clearly_phishing(features):

            heuristic_flag = True

            heuristic_note = (
                " Heuristic override: malformed URL, "
                "unknown institutional domain, or "
                "suspicious path on a trusted-looking TLD."
            )

        else:

            prediction, confidence, selected_model = (
                resolve_url_prediction(
                    model_results,
                    features,
                )
            )

            best_model = {
                **selected_model,
                "prediction": prediction,
                "confidence": round(
                    confidence,
                    3,
                ),
            }

            if (
                prediction == "Legit"
                and any(
                    item["prediction"]
                    == "Phishing"
                    for item in model_results
                )
            ):
                ensemble_note = (
                    " Ensemble decision applied to "
                    "reduce false positives on "
                    "well-formed URLs."
                )

    # -------------------------------------------------
    # STEP 4: URL phishing override
    # -------------------------------------------------

    if heuristic_flag:

        for item in model_results:

            item["prediction"] = "Phishing"

            item["confidence"] = max(
                item["confidence"],
                0.95,
            )

        best_model = {
            **best_model,
            "prediction": "Phishing",
            "confidence": max(
                best_model["confidence"],
                0.95,
            ),
        }

    # -------------------------------------------------
    # STEP 5: Trusted legitimate override
    # -------------------------------------------------

    if trusted_override:

        best_model = {
            **best_model,
            "prediction": "Legit",
            "confidence": max(
                best_model["confidence"],
                0.95,
            ),
        }

        for item in model_results:

            if (
                item["name"]
                == best_model["name"]
            ):

                item["prediction"] = "Legit"

                item["confidence"] = (
                    best_model["confidence"]
                )

    if forced_prediction:
        best_model = {
            **best_model,
            "prediction": forced_prediction,
            "confidence": max(best_model["confidence"], 0.95),
        }
        for item in model_results:
            item["prediction"] = forced_prediction
            item["confidence"] = max(item["confidence"], 0.95)

    # -------------------------------------------------
    # STEP 6: Explainability
    # -------------------------------------------------

    reference_model = models[
        best_model["name"]
    ]

    if input_type.lower() == "phone":

        # IMPORTANT:
        # Phone requests use ONLY permutation.
        #
        # SHAP and LIME are deliberately skipped
        # to prevent Render timeout / memory issues.

        explanation_results, best_explainer = (
            compare_explainability_methods(
                reference_model,
                phone_x_train_for_explanation(
                    x_train
                ),
                x,
                cache=explainer_cache,
                model_name=best_model["name"],
                robustness_trials=0,
                methods=["Permutation"],
            )
        )

    elif os.getenv("RENDER", "").lower() == "true":

        # Render's small workers can be killed while SHAP/Numba compiles.
        # Permutation provides explanations without the LLVM memory spike.
        explanation_results, best_explainer = (
            compare_explainability_methods(
                reference_model,
                x_train,
                x,
                cache=explainer_cache,
                model_name=best_model["name"],
                robustness_trials=0,
                methods=["Permutation"],
            )
        )

    else:

        explanation_results, best_explainer = (
            compare_explainability_methods(
                reference_model,
                x_train,
                x,
                cache=explainer_cache,
                model_name=best_model["name"],
                robustness_trials=2,
            )
        )

    # -------------------------------------------------
    # STEP 7: Human-readable explanation
    # -------------------------------------------------

    explanation = (
        f"For this {input_type}, "
        f"{best_model['name']} is the best ML "
        f"algorithm (score: {best_model['score']}, "
        f"prediction: {best_model['prediction']}, "
        f"confidence: {best_model['confidence']}). "
        f"{best_explainer['name']} is the selected "
        f"explainability method "
        f"(score: {best_explainer['score']}, "
        f"robustness: "
        f"{best_explainer['robustness']}, "
        f"complexity: "
        f"{best_explainer['complexity']}s)."
        f"{trusted_note}"
        f"{heuristic_note}"
        f"{ensemble_note}"
    )

    # -------------------------------------------------
    # STEP 8: JSON response
    # -------------------------------------------------

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

        "best_explainer": (
            best_explainer["name"]
        ),

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

        "selected_top_features": (
            best_explainer["top_features"]
        ),

        "explanation": explanation,

        "trusted_override": trusted_override,
    }


def phone_x_train_for_explanation(
    x_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep the phone explanation background small.

    Permutation explanation does not actually need the
    complete training dataset, but it uses feature medians.
    A small sample is enough and reduces memory usage.
    """

    if len(x_train) <= 100:

        return x_train

    return x_train.sample(
        100,
        random_state=42,
    )


# -----------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------


def load_model_bundle(
    base_dir: str,
    prefix: str = "",
):
    """Load models and training data from /models."""
    models_name = (
        f"{prefix}saved_models.pkl"
        if prefix
        else "saved_models.pkl"
    )
    train_name = (
        f"{prefix}X_train.pkl"
        if prefix
        else "X_train.pkl"
    )
    models_path = os.path.join(base_dir, "models", models_name)
    x_train_path = os.path.join(base_dir, "models", train_name)
    models = joblib.load(models_path)
    x_train = joblib.load(x_train_path)
    cache = ExplainerCache(models, x_train)
    return models, x_train, cache


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# URL models
url_models, url_x_train, url_explainer_cache = (
    load_model_bundle(
        BASE_DIR
    )
)

# Phone models
phone_models, phone_x_train, phone_explainer_cache = (
    load_model_bundle(
        BASE_DIR,
        "phone_",
    )
)

# Email models are optional
try:

    email_models, email_x_train, email_explainer_cache = (
        load_model_bundle(
            BASE_DIR,
            "email_",
        )
    )

    email_models_loaded = True

except Exception as exc:

    print(
        f"Email ML models not loaded: {exc}"
    )

    email_models = None
    email_x_train = None
    email_explainer_cache = None
    email_models_loaded = False
