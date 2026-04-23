#!/usr/bin/env python3
"""Train a default HMDA model from synthetic data (used in Docker when no model file is mounted)."""

from __future__ import annotations

import os
from pathlib import Path

import joblib

from hmda_model import feature_target_split, prepare_training_frame, synthetic_hmda_rows, train_hmda_pipeline


def main() -> None:
    out = Path(os.environ.get("MODEL_PATH", "model_hmda.pkl"))
    n = int(os.environ.get("BOOTSTRAP_ROWS", "3000"))
    df = synthetic_hmda_rows(n=n, seed=42)
    ready = prepare_training_frame(df)
    X, y = feature_target_split(ready)
    model = train_hmda_pipeline(X, y)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"Wrote {out} (trained on {len(X)} rows)")


if __name__ == "__main__":
    main()
