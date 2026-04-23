"""
HMDA mortgage decision model — training pipeline aligned with `fairlytics.ipynb`.

Target: action_taken 1 or 2 → 1 (originated/approved), 3 → 0 (denied).
Leakage columns are removed before training.
"""

from __future__ import annotations

import warnings
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Mirrors the notebook
LEAKAGE_COLS: Tuple[str, ...] = (
    "action_taken",
    "action_taken_name",
    "denial_reason_name_1",
    "denial_reason_1",
    "denial_reason_2",
    "denial_reason_3",
    "rate_spread",
    "edit_status",
    "sequence_number",
    "application_date_indicator",
)


def add_target_from_action_taken(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target"] = out["action_taken"].map({1: 1, 2: 1, 3: 0})
    return out


def drop_high_null_columns(df: pd.DataFrame, null_fraction: float = 0.5) -> pd.DataFrame:
    null_pct = df.isnull().sum() / len(df)
    drop_cols = null_pct[null_pct > null_fraction].index.tolist()
    if not drop_cols:
        return df
    return df.drop(columns=drop_cols)


def drop_leakage_columns(df: pd.DataFrame, extra_drop: Iterable[str] | None = None) -> pd.DataFrame:
    to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if extra_drop:
        to_drop.extend(c for c in extra_drop if c in df.columns)
    if not to_drop:
        return df
    return df.drop(columns=to_drop)


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same cleaning as the notebook: target, drop leakage, drop sparse columns,
    drop rows with missing target.
    """
    df = add_target_from_action_taken(df)
    df = drop_leakage_columns(df)
    df = drop_high_null_columns(df)
    df = df.dropna(subset=["target"])
    return df


def feature_target_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df["target"].astype(int)
    X = df.drop(columns=["target"])
    return X, y


def _numeric_and_categorical_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessors(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_hmda_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build a sklearn Pipeline (preprocess + XGBClassifier) for a cleaned feature matrix X.
    """
    numeric_features, categorical_features = _numeric_and_categorical_columns(X)
    preprocessor = build_preprocessors(numeric_features, categorical_features)
    # Notebook used RandomForest in imports but XGBClassifier in clf; match fitted model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def train_hmda_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = build_hmda_pipeline(X)
    pipe.fit(X, y)
    return pipe


def synthetic_hmda_rows(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Random HMDA-like rows including action_taken (for training / bootstrap in Docker)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        action = int(rng.choice([1, 2, 3], p=[0.45, 0.45, 0.1]))
        rows.append(
            {
                "action_taken": action,
                "action_taken_name": "irrelevant",
                "loan_amount_000s": float(rng.integers(50, 800)),
                "applicant_income_000s": float(rng.integers(10, 250)),
                "loan_type": int(rng.integers(1, 5)),
                "property_type": int(rng.integers(1, 4)),
                "applicant_ethnicity": int(rng.integers(1, 4)),
                "applicant_race_1": int(rng.integers(1, 8)),
                "applicant_sex": int(rng.integers(1, 5)),
                "hoepa_status": int(rng.integers(1, 3)),
                "lien_status": int(rng.integers(1, 3)),
                "population": float(rng.integers(500, 10000)),
                "minority_population": float(rng.uniform(1, 99)),
                "state_name": rng.choice(["CA", "TX", "NY", "FL"]),
                "loan_type_name": rng.choice(["Conventional", "FHA", "VA"]),
                "applicant_ethnicity_name": rng.choice(["Hispanic", "Not Hispanic"]),
                "applicant_race_name_1": rng.choice(["White", "Asian", "Black"]),
                "denial_reason_1": np.nan,
                "denial_reason_name_1": np.nan,
                "rate_spread": np.nan,
                "edit_status": np.nan,
                "sequence_number": np.nan,
                "application_date_indicator": np.nan,
            }
        )
    return pd.DataFrame(rows)
