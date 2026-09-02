"""Model pipeline construction (Step 05).

One rule governs this module: **every transformation lives inside the fitted
pipeline object**. Nothing is imputed, encoded or scaled outside it. That is
what lets `joblib.load(...)` produce an artifact that is complete on its own,
and it removes the entire class of bugs where serving preprocesses differently
from training.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hrai.features.engineering import FeatureEngineer, feature_columns
from hrai.utils.logger import get_logger

log = get_logger(__name__)


def split_column_types(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    df: pd.DataFrame, columns: list[str], scale: bool = True
) -> ColumnTransformer:
    """Impute + encode + (optionally) scale.

    Scaling matters for Logistic Regression and is irrelevant for tree models,
    so it is a parameter rather than an unconditional step — an unnecessary
    scaler on a tree model is harmless but misleading to whoever reads the
    artifact later.
    """
    numeric, categorical = split_column_types(df, columns)

    numeric_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        # Unknown categories at serve time must not raise — an unseen
                        # job title should degrade the prediction, not 500 the API.
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False, min_frequency=5
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_pipeline(
    df: pd.DataFrame, estimator, *, scale: bool = True, engineer: bool = True
) -> Pipeline:
    """Assemble the full estimator: engineer -> preprocess -> classify."""
    engineered = FeatureEngineer(enabled=engineer).fit(df).transform(df)
    columns = feature_columns(engineered)
    numeric, categorical = split_column_types(engineered, columns)

    log.info(
        "pipeline built",
        extra={
            "estimator": type(estimator).__name__,
            "numeric_features": len(numeric),
            "categorical_features": len(categorical),
            "scaled": scale,
        },
    )
    return Pipeline(
        [
            ("engineer", FeatureEngineer(enabled=engineer)),
            ("prep", build_preprocessor(engineered, columns, scale=scale)),
            ("clf", estimator),
        ]
    )


def fitted_feature_names(pipeline: Pipeline) -> list[str]:
    """Post-encoding feature names, for SHAP and coefficient inspection."""
    return list(pipeline.named_steps["prep"].get_feature_names_out())
