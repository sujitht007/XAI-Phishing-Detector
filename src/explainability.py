
import logging
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


logging.getLogger("shap").setLevel(
    logging.WARNING
)


AttributionMap = Dict[str, float]

ExplainFn = Callable[..., AttributionMap]


def _target_class(
    model,
    x_sample: pd.DataFrame,
) -> int:

    return int(
        model.predict(x_sample)[0]
    )


def _to_attribution_map(
    feature_names: List[str],
    values: np.ndarray,
) -> AttributionMap:

    values = np.asarray(values).reshape(-1)

    return {
        feature: float(value)
        for feature, value in zip(
            feature_names,
            values,
        )
    }


class ExplainerCache:
    """
    Lightweight lazy cache for explainers.

    Expensive LIME/SHAP objects are created only when
    actually needed instead of creating all explainers
    during Flask startup.
    """

    def __init__(
        self,
        models: dict,
        x_train: pd.DataFrame,
    ):

        self.x_train = x_train

        self.feature_names = list(
            x_train.columns
        )

        # Small background dataset.
        # This reduces memory usage.
        self.background = (
            x_train.sample(
                min(20, len(x_train)),
                random_state=42,
            )
        )

        self._shap_tree = {}

        self._shap_general = {}

        self._lime = {}

        self._predictors = {}

        self._models = models

    def _make_predictor(
        self,
        model,
    ):

        feature_names = self.feature_names

        def predict_proba_with_names(data):

            frame = pd.DataFrame(
                data,
                columns=feature_names,
            )

            return model.predict_proba(
                frame
            )

        return predict_proba_with_names

    def get_lime(
        self,
        model_name: str,
    ):

        if model_name not in self._lime:

            self._lime[model_name] = (
                LimeTabularExplainer(
                    self.x_train.values,
                    feature_names=self.feature_names,
                    class_names=[
                        "Legit",
                        "Phishing",
                    ],
                    mode="classification",
                    random_state=42,
                    discretize_continuous=True,
                )
            )

        return self._lime[model_name]

    def get_predictor(
        self,
        model_name: str,
        model,
    ):

        if model_name not in self._predictors:

            self._predictors[
                model_name
            ] = self._make_predictor(
                model
            )

        return self._predictors[
            model_name
        ]


def _unwrap_estimator(model):

    if (
        hasattr(model, "named_steps")
        and "model" in model.named_steps
    ):
        return model.named_steps["model"]

    return model


def explain_with_shap(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:

    target = _target_class(
        model,
        x_sample,
    )

    feature_names = list(
        x_sample.columns
    )

    estimator = _unwrap_estimator(
        model
    )

    # -------------------------------------------------
    # Tree models
    # -------------------------------------------------

    if isinstance(
        estimator,
        (
            RandomForestClassifier,
            XGBClassifier,
        ),
    ):

        if (
            cache is not None
            and model_name
            and model_name in cache._shap_tree
        ):

            explainer = cache._shap_tree[
                model_name
            ]

        else:

            # Small background for lower memory usage.
            background = (
                x_train.sample(
                    min(20, len(x_train)),
                    random_state=42,
                )
            )

            explainer = shap.TreeExplainer(
                estimator,
                data=background,
            )

            if (
                cache is not None
                and model_name
            ):

                cache._shap_tree[
                    model_name
                ] = explainer

        shap_values = explainer(
            x_sample
        )

        values = shap_values.values

        # SHAP can return:
        #
        # [samples, features]
        #
        # or
        #
        # [samples, features, classes]

        if values.ndim == 3:

            values = values[
                0,
                :,
                target,
            ]

        elif values.ndim == 2:

            values = values[
                0,
                :
            ]

        else:

            values = values.reshape(-1)

        return _to_attribution_map(
            feature_names,
            values,
        )

    # -------------------------------------------------
    # General models
    # -------------------------------------------------

    if (
        cache is not None
        and model_name
        and model_name in cache._shap_general
    ):

        explainer = cache._shap_general[
            model_name
        ]

    else:

        background = (
            x_train.sample(
                min(10, len(x_train)),
                random_state=42,
            )
        )

        explainer = shap.Explainer(
            model.predict_proba,
            background,
            algorithm="permutation",
        )

        if (
            cache is not None
            and model_name
        ):

            cache._shap_general[
                model_name
            ] = explainer

    shap_values = explainer(
        x_sample
    )

    values = shap_values.values

    if values.ndim == 3:

        values = values[
            0,
            :,
            target,
        ]

    elif values.ndim == 2:

        values = values[
            0,
            :
        ]

    else:

        values = values.reshape(-1)

    return _to_attribution_map(
        feature_names,
        values,
    )


def explain_with_lime(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:

    target = _target_class(
        model,
        x_sample,
    )

    feature_names = list(
        x_train.columns
    )

    # Reuse the predictor if available.
    if (
        cache is not None
        and model_name
    ):

        explainer = cache.get_lime(
            model_name
        )

        predict_fn = cache.get_predictor(
            model_name,
            model,
        )

    else:

        explainer = (
            LimeTabularExplainer(
                x_train.values,
                feature_names=feature_names,
                class_names=[
                    "Legit",
                    "Phishing",
                ],
                mode="classification",
                random_state=42,
                discretize_continuous=True,
            )
        )

        def predict_fn(data):

            frame = pd.DataFrame(
                data,
                columns=feature_names,
            )

            return model.predict_proba(
                frame
            )

    explanation = explainer.explain_instance(
        x_sample.values[0],
        predict_fn,
        num_features=len(
            x_sample.columns
        ),
        top_labels=2,
        num_samples=50,
    )

    available_labels = (
        explanation.available_labels()
    )

    if not available_labels:
        return {}

    label = (
        target
        if target in available_labels
        else available_labels[0]
    )

    pairs = explanation.as_list(
        label=label
    )

    return {
        feature: float(weight)
        for feature, weight in pairs
    }


def explain_with_permutation(
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:

    target = _target_class(
        model,
        x_sample,
    )

    baseline = float(
        model.predict_proba(
            x_sample
        )[0][target]
    )

    attributions = {}

    # Calculate medians once.
    medians = x_train.median(
        numeric_only=True
    )

    for feature in x_sample.columns:

        perturbed = x_sample.copy()

        if feature in medians.index:

            perturbed[feature] = (
                medians[feature]
            )

        perturbed_score = float(
            model.predict_proba(
                perturbed
            )[0][target]
        )

        attributions[feature] = (
            baseline - perturbed_score
        )

    return attributions


EXPLAINERS: Dict[
    str,
    ExplainFn
] = {
    "SHAP": explain_with_shap,
    "LIME": explain_with_lime,
    "Permutation": explain_with_permutation,
}


def available_explainers(
    model,
) -> List[str]:

    estimator = _unwrap_estimator(
        model
    )

    if isinstance(
        estimator,
        (
            RandomForestClassifier,
            XGBClassifier,
        ),
    ):

        return list(
            EXPLAINERS.keys()
        )

    return [
        "LIME",
        "Permutation",
    ]


def generate_explanation(
    method_name: str,
    model,
    x_train: pd.DataFrame,
    x_sample: pd.DataFrame,
    cache: Optional[ExplainerCache] = None,
    model_name: Optional[str] = None,
) -> AttributionMap:

    if method_name not in EXPLAINERS:

        raise ValueError(
            f"Unknown explainability method: "
            f"{method_name}"
        )

    return EXPLAINERS[
        method_name
    ](
        model,
        x_train,
        x_sample,
        cache=cache,
        model_name=model_name,
    )
