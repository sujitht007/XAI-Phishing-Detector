import time
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from src.explainability import AttributionMap, EXPLAINERS, ExplainerCache, available_explainers, generate_explanation


def _rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values))
    return ranks


def _spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 1.0
    if np.std(a) == 0 and np.std(b) == 0:
        return 1.0
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr = float(np.corrcoef(_rank_values(a), _rank_values(b))[0, 1])
    return 0.0 if np.isnan(corr) else corr


def attribution_similarity(
    base_attrs: AttributionMap,
    perturbed_attrs: AttributionMap,
    top_k: int = 5,
) -> float:
    features = sorted(set(base_attrs) | set(perturbed_attrs))
    base_vec = np.array([base_attrs.get(feature, 0.0) for feature in features])
    perturbed_vec = np.array([perturbed_attrs.get(feature, 0.0) for feature in features])

    rank_corr = max(0.0, _spearman_correlation(base_vec, perturbed_vec))

    top_base = set(
        feature
        for feature, _ in sorted(base_attrs.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    )
    top_perturbed = set(
        feature
        for feature, _ in sorted(perturbed_attrs.items(), key=lambda item: abs(item[1]), reverse=True)[:top_k]
    )
    union = top_base | top_perturbed
    jaccard = len(top_base & top_perturbed) / len(union) if union else 1.0

    return 0.5 * rank_corr + 0.5 * jaccard


def explanation_robustness_score(
    method_name: str,
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    trials: int = 5,
    noise_scale: float = 0.01,
) -> float:
    base_attrs = generate_explanation(method_name, model, x_train, x_sample)
    stable = 0.0

    for _ in range(trials):
        noise = np.random.normal(0, noise_scale, x_sample.values.shape)
        noisy_sample = pd.DataFrame(
            x_sample.values + noise,
            columns=x_sample.columns,
        )
        noisy_attrs = generate_explanation(method_name, model, x_train, noisy_sample)
        stable += attribution_similarity(base_attrs, noisy_attrs)

    return stable / trials


def explanation_complexity_score(
    method_name: str,
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
) -> float:
    start = time.time()
    generate_explanation(method_name, model, x_train, x_sample)
    return time.time() - start


def compute_explanation_score(robustness: float, complexity: float, max_complexity: float) -> float:
    """Select explanations that are robust and less complex."""
    normalized_complexity = complexity / max_complexity if max_complexity > 0 else 0.0
    return (0.7 * robustness) - (0.3 * normalized_complexity)


def compute_model_score(robustness: float, complexity: float, confidence: float, max_complexity: float) -> float:
    """Prioritize prediction quality; use speed only as a minor tiebreaker."""
    normalized_complexity = complexity / max_complexity if max_complexity > 0 else 0.0
    return (0.6 * confidence) + (0.38 * robustness) - (0.02 * normalized_complexity)


def model_robustness_score(model, x: pd.DataFrame, trials: int = 2, noise_scale: float = 0.01) -> float:
    base_pred = int(model.predict(x)[0])
    stable = 0

    for _ in range(trials):
        noise = np.random.normal(0, noise_scale, x.values.shape)
        noisy_sample = pd.DataFrame(x.values + noise, columns=x.columns)
        if int(model.predict(noisy_sample)[0]) == base_pred:
            stable += 1

    return stable / trials


def model_complexity_score(model, x: pd.DataFrame) -> float:
    start = time.time()
    model.predict(x)
    return time.time() - start


def compare_ml_models(models: dict, x: pd.DataFrame) -> Tuple[list, dict]:
    results = []

    for name, model in models.items():
        robustness = model_robustness_score(model, x)
        complexity = model_complexity_score(model, x)
        prediction = int(model.predict(x)[0])
        confidence = float(model.predict_proba(x)[0].max())

        results.append({
            "name": name,
            "robustness": round(float(robustness), 3),
            "complexity": round(float(complexity), 3),
            "confidence": round(confidence, 3),
            "prediction": "Phishing" if prediction == 1 else "Legit",
        })

    max_complexity = max(item["complexity"] for item in results) or 1.0
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

    best = max(results, key=lambda item: item["score"])
    return results, best


def select_best_explainer(results: list) -> dict:
    if not results:
        raise ValueError("No explainability results to compare.")
    return max(results, key=lambda item: item["score"])


def compare_explainability_methods(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: ExplainerCache | None = None,
    model_name: str | None = None,
    robustness_trials: int = 2,
) -> Tuple[list, dict]:
    results = []

    for method_name in available_explainers(model):
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

        stable = 0.0
        for _ in range(robustness_trials):
            noise = np.random.normal(0, 0.01, x_sample.values.shape)
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
            stable += attribution_similarity(attributions, noisy_attrs)
        robustness = stable / robustness_trials

        results.append({
            "name": method_name,
            "robustness": round(float(robustness), 3),
            "complexity": round(float(complexity), 3),
            "attributions": attributions,
        })

    max_complexity = max(item["complexity"] for item in results) or 1.0
    for item in results:
        item["score"] = round(
            compute_explanation_score(item["robustness"], item["complexity"], max_complexity),
            3,
        )
        ranked = sorted(
            item["attributions"].items(),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        item["top_features"] = [
            {"feature": feature, "importance": round(float(value), 4)}
            for feature, value in ranked[:5]
        ]

    best = select_best_explainer(results)
    return results, best
