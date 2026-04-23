"""Tests for the HMDA fairness pipeline (see `fairlytics.ipynb` and `hmda_model.py`)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hmda_model import (
    LEAKAGE_COLS,
    add_target_from_action_taken,
    build_hmda_pipeline,
    drop_high_null_columns,
    drop_leakage_columns,
    feature_target_split,
    prepare_training_frame,
    synthetic_hmda_rows,
    train_hmda_pipeline,
)


def test_target_mapping():
    df = pd.DataFrame({"action_taken": [1, 2, 3, 1]})
    out = add_target_from_action_taken(df)
    assert list(out["target"]) == [1, 1, 0, 1]


def test_leakage_columns_removed():
    df = synthetic_hmda_rows(20)
    cleaned = drop_leakage_columns(df)
    for c in LEAKAGE_COLS:
        if c in df.columns:
            assert c not in cleaned.columns


def test_high_null_columns_dropped():
    df = pd.DataFrame(
        {
            "action_taken": [1, 1, 1],
            "keep_col": [1.0, 2.0, 3.0],
            "sparse_col": [1.0, np.nan, np.nan],
        }
    )
    df = add_target_from_action_taken(df)
    df = drop_leakage_columns(df)
    reduced = drop_high_null_columns(df, null_fraction=0.5)
    assert "sparse_col" not in reduced.columns
    assert "keep_col" in reduced.columns


def test_prepare_training_frame_end_to_end():
    df = synthetic_hmda_rows(300)
    ready = prepare_training_frame(df)
    assert "target" in ready.columns
    assert "action_taken" not in ready.columns
    assert ready["target"].notna().all()
    X, y = feature_target_split(ready)
    assert "target" not in X.columns
    assert set(y.unique()).issubset({0, 1})


def test_pipeline_fit_and_predict():
    df = synthetic_hmda_rows(500)
    ready = prepare_training_frame(df)
    X, y = feature_target_split(ready)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )
    model = train_hmda_pipeline(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)


def test_build_pipeline_without_fit():
    df = prepare_training_frame(synthetic_hmda_rows(50))
    X, _ = feature_target_split(df)
    pipe = build_hmda_pipeline(X)
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps
