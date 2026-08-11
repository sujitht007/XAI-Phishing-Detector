import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.getLogger("shap").setLevel(logging.WARNING)

AttributionMap = Dict[str, float]
ExplainFn = Callable[..., AttributionMap]


def _target_class(model, x_sample: pd.DataFrame) -> int:
    return int(model.predict(x_sample)[0])


def _to_attribution_map(feature_names: List[str], values: np.ndarray) -> AttributionMap:
    return {
        feature: float(value)
        for feature, value in zip(feature_names, values)
    }


class ExplainerCache:
    """Reuse expensive SHAP/LIME explainers across requests."""

    def __init__(self, models: dict, x_train: pd.DataFrame):
        self.x_train = x_train
        self.feature_names = list(x_train.columns)
        self.background = x_train.sample(min(30, len(x_train)), random_state=42)
        self._shap_tree: Dict[str, shap.TreeExplainer] = {}
        self._shap_general: Dict[str, shap.Explainer] = {}
        self._lime: Dict[str, LimeTabularExplainer] = {}

        for name, model in models.items():
            estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
            if isinstance(estimator, (RandomForestClassifier, XGBClassifier)):
                # Prefer Kernel/Permutation explainers for scaled pipelines; keep LIME+Permutation fast.
                pass

            self._lime[name] = LimeTabularExplainer(
                x_train.values,
                feature_names=self.feature_names,
                class_names=["Legit", "Phishing"],
                mode="classification",
                random_state=42,
            )

        self._predictors = {
            name: self._make_predictor(model)
            for name, model in models.items()
        }

    def _make_predictor(self, model):
        feature_names = self.feature_names

        def predict_proba_with_names(data):
            frame = pd.DataFrame(data, columns=feature_names)
            return model.predict_proba(frame)

        return predict_proba_with_names


def explain_with_shap(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:
    target = _target_class(model, x_sample)
    feature_names = list(x_sample.columns)

    if cache and model_name and model_name in cache._shap_tree:
        shap_values = cache._shap_tree[model_name](x_sample)
        values = shap_values.values[0]
        if values.ndim > 1:
            values = values[:, target]
    elif cache and model_name and model_name in cache._shap_general:
        shap_values = cache._shap_general[model_name](x_sample)
        values = shap_values.values[0, :, target]
    elif isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(x_sample)
        values = shap_values.values[0]
        if values.ndim > 1:
            values = values[:, target]
    else:
        background = x_train.sample(min(30, len(x_train)), random_state=42)
        explainer = shap.Explainer(model.predict_proba, background, algorithm="permutation")
        shap_values = explainer(x_sample)
        values = shap_values.values[0, :, target]

    return _to_attribution_map(feature_names, values)


def explain_with_lime(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:
    target = _target_class(model, x_sample)

    if cache and model_name:
        explainer = cache._lime[model_name]
        predict_fn = cache._predictors[model_name]
    else:
        feature_names = list(x_train.columns)

        def predict_fn(data):
            frame = pd.DataFrame(data, columns=feature_names)
            return model.predict_proba(frame)

        explainer = LimeTabularExplainer(
            x_train.values,
            feature_names=feature_names,
            class_names=["Legit", "Phishing"],
            mode="classification",
            random_state=42,
        )

    explanation = explainer.explain_instance(
        x_sample.values[0],
        predict_fn,
        num_features=len(x_sample.columns),
        top_labels=2,
        num_samples=100,
    )
    available_labels = explanation.available_labels()
    label = target if target in available_labels else available_labels[0]
    pairs = explanation.as_list(label=label)
    return {feature: float(weight) for feature, weight in pairs}


def explain_with_permutation(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:
    target = _target_class(model, x_sample)
    baseline = float(model.predict_proba(x_sample)[0][target])
    attributions = {}

    for feature in x_sample.columns:
        perturbed = x_sample.copy()
        perturbed[feature] = x_train[feature].median()
        perturbed_score = float(model.predict_proba(perturbed)[0][target])
        attributions[feature] = baseline - perturbed_score

    return attributions


EXPLAINERS: Dict[str, ExplainFn] = {
    "SHAP": explain_with_shap,
    "LIME": explain_with_lime,
    "Permutation": explain_with_permutation,
}


def _unwrap_estimator(model):
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def available_explainers(model) -> List[str]:
    """Tree models support fast SHAP; SVM / pipelines without trees use lighter explainers."""
    estimator = _unwrap_estimator(model)
    if isinstance(estimator, (RandomForestClassifier, XGBClassifier)):
        return list(EXPLAINERS.keys())
    return ["LIME", "Permutation"]


def generate_explanation(
    method_name: str,
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:
    if method_name not in EXPLAINERS:
        raise ValueError(f"Unknown explainability method: {method_name}")
    return EXPLAINERS[method_name](
        model,
        x_train,
        x_sample,
        cache=cache,
        model_name=model_name,
    )
