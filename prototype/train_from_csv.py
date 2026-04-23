#!/usr/bin/env python3
"""
Train the HMDA pipeline on a CSV (same schema as `hmda_2017_nationwide_all-records_labels.csv`)
and save a joblib model for deployment or local checks.

Example:
  python train_from_csv.py --csv /path/to/hmda_2017_nationwide_all-records_labels.csv \\
    --rows 100000 --out model_hmda.pkl
"""

from __future__ import annotations

import argparse
import joblib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from hmda_model import feature_target_split, prepare_training_frame, train_hmda_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Train HMDA XGB pipeline from CSV")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("model_hmda.pkl"))
    p.add_argument("--rows", type=int, default=None, help="Max rows to read (for subsampling)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.csv, nrows=args.rows, low_memory=False)
    ready = prepare_training_frame(df)
    X, y = feature_target_split(ready)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    model = train_hmda_pipeline(X_train, y_train)
    joblib.dump(model, args.out)
    print(f"Saved {args.out} (trained on {len(X_train)} rows)")


if __name__ == "__main__":
    main()
