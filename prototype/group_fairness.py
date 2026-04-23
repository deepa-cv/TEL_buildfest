"""
Group fairness metrics over sliding windows (HMDA-aligned attributes).

Maps dataset fields to monitoring groups:
- Region: US census region from `state_name` (abbreviation or full name).
- Sex: `applicant_sex` (HMDA codes 1/2 → Male/Female; other codes preserved).
- Race: simplified bucket from `applicant_race_name_1` (and code fallback).
- Income: low vs high from `applicant_income_000s` (proxy when individual age is absent in LAR).
- Tract minority: high vs low from `minority_population` (community context).
- Intersection: concatenation of sex × race × income (and optional region).

Metrics (Y = ground truth positive, Ŷ = predicted positive):
- Demographic parity: comparable positive prediction rates P(Ŷ=1|group).
- Disparate impact (80% rule style): ratio of group positive rates vs reference (min/max).
- Equal opportunity: TPR parity across groups (needs Y).
- Equalized odds: TPR and FPR parity (needs Y).
- Predictive parity: PPV parity P(Y=1|Ŷ=1) (needs Y).
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional

import pandas as pd

# US census regions by two-letter abbreviation
_ABBR_TO_REGION = {
    "CT": "Northeast",
    "ME": "Northeast",
    "MA": "Northeast",
    "NH": "Northeast",
    "RI": "Northeast",
    "VT": "Northeast",
    "NJ": "Northeast",
    "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",
    "IN": "Midwest",
    "MI": "Midwest",
    "OH": "Midwest",
    "WI": "Midwest",
    "IA": "Midwest",
    "KS": "Midwest",
    "MN": "Midwest",
    "MO": "Midwest",
    "NE": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "DE": "South",
    "FL": "South",
    "GA": "South",
    "MD": "South",
    "NC": "South",
    "SC": "South",
    "VA": "South",
    "WV": "South",
    "KY": "South",
    "TN": "South",
    "AL": "South",
    "MS": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "DC": "South",
    "MT": "West",
    "ID": "West",
    "WY": "West",
    "CO": "West",
    "NM": "West",
    "AZ": "West",
    "UT": "West",
    "NV": "West",
    "WA": "West",
    "OR": "West",
    "CA": "West",
    "AK": "West",
    "HI": "West",
}

# Common full state names → region (when API sends full name)
_NAME_TO_REGION = {
    "California": "West",
    "Texas": "South",
    "Florida": "South",
    "New York": "Northeast",
    "Washington": "West",
    "Oregon": "West",
    "Illinois": "Midwest",
    "Pennsylvania": "Northeast",
    "Ohio": "Midwest",
    "Georgia": "South",
    "North Carolina": "South",
    "Michigan": "Midwest",
    "New Jersey": "Northeast",
    "Virginia": "South",
    "Arizona": "West",
    "Massachusetts": "Northeast",
    "Tennessee": "South",
    "Indiana": "Midwest",
    "Missouri": "Midwest",
    "Maryland": "South",
    "Wisconsin": "Midwest",
    "Colorado": "West",
    "Minnesota": "Midwest",
    "South Carolina": "South",
    "Alabama": "South",
    "Louisiana": "South",
    "Kentucky": "South",
    "Oklahoma": "South",
    "Connecticut": "Northeast",
    "Utah": "West",
    "Iowa": "Midwest",
    "Nevada": "West",
    "Arkansas": "South",
    "Mississippi": "South",
    "Kansas": "Midwest",
    "New Mexico": "West",
    "Nebraska": "Midwest",
    "Idaho": "West",
    "West Virginia": "South",
    "Hawaii": "West",
    "New Hampshire": "Northeast",
    "Maine": "Northeast",
    "Montana": "West",
    "Rhode Island": "Northeast",
    "Delaware": "South",
    "South Dakota": "Midwest",
    "North Dakota": "Midwest",
    "Alaska": "West",
    "Vermont": "Northeast",
    "Wyoming": "West",
    "District of Columbia": "South",
}

INCOME_LOW_THRESHOLD_K = 80.0  # $000s — below = "low_income", at/above = "high_income"
MINORITY_HIGH_PCT = 30.0  # tract minority share threshold


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def region_from_state(state_name: Any) -> str:
    if state_name is None:
        return "Unknown"
    s = str(state_name).strip()
    if len(s) == 2:
        return _ABBR_TO_REGION.get(s.upper(), "Other")
    return _NAME_TO_REGION.get(s.title(), "Other")


def sex_bucket(applicant_sex: Any) -> str:
    try:
        code = int(float(applicant_sex))
    except (TypeError, ValueError):
        return "Unknown"
    if code == 1:
        return "Male"
    if code == 2:
        return "Female"
    return f"Sex_code_{code}"


def race_bucket(race_name: Any, race_code: Any) -> str:
    if race_name and str(race_name).strip() and str(race_name).lower() != "nan":
        n = str(race_name).strip().split()[0]
        return n[:32]
    try:
        rc = int(float(race_code))
        return f"race_{rc}"
    except (TypeError, ValueError):
        return "Unknown"


def income_bracket(income_k: Any) -> str:
    v = _safe_float(income_k)
    if v is None:
        return "Unknown"
    return "low_income" if v < INCOME_LOW_THRESHOLD_K else "high_income"


def minority_tract_bucket(minority_pct: Any) -> str:
    v = _safe_float(minority_pct)
    if v is None:
        return "Unknown"
    return "high_minority_tract" if v >= MINORITY_HIGH_PCT else "low_minority_tract"


def build_analysis_meta(features: Dict[str, Any]) -> Dict[str, Any]:
    """Derive stable group tags from HMDA-style feature dict (no model changes required)."""
    region = region_from_state(features.get("state_name"))
    sex = sex_bucket(features.get("applicant_sex"))
    race = race_bucket(features.get("applicant_race_name_1"), features.get("applicant_race_1"))
    inc = income_bracket(features.get("applicant_income_000s"))
    tract = minority_tract_bucket(features.get("minority_population"))
    intersection = f"{sex}|{race}|{inc}|{region}"
    return {
        "us_region": region,
        "applicant_sex": sex,
        "race_bucket": race,
        "income_bracket": inc,
        "tract_minority_bucket": tract,
        "intersection_key": intersection,
    }


def _group_series(df: pd.DataFrame, attribute: str) -> pd.Series:
    if attribute == "intersection":
        return df["intersection_key"]
    if attribute in df.columns:
        return df[attribute]
    meta = df["analysis_meta"]
    if hasattr(meta, "apply"):
        return meta.apply(lambda m: (m or {}).get(attribute, "Unknown"))
    return pd.Series(["Unknown"] * len(df))


def positive_rates(df: pd.DataFrame, group_col: str) -> pd.Series:
    g = df.groupby(group_col, dropna=False)["y_hat"]
    return g.mean()


def demographic_parity_gap(df: pd.DataFrame, group_col: str) -> float:
    """Max positive rate minus min across groups (0 = perfect parity on approval rates)."""
    rates = positive_rates(df, group_col)
    if len(rates) < 2:
        return 0.0
    return float(rates.max() - rates.min())


def disparate_impact_ratio(df: pd.DataFrame, group_col: str) -> Optional[float]:
    """
    Ratio min(selection rate) / max(selection rate) across groups.
    Values near 1.0 are balanced; common rule-of-thumb threshold 0.8 (80% rule).
    """
    rates = positive_rates(df, group_col)
    if len(rates) < 2:
        return None
    mx = float(rates.max())
    mn = float(rates.min())
    if mx <= 0:
        return None
    return mn / mx


def _confusion_by_group(df: pd.DataFrame, group_col: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for g, sub in df.groupby(group_col, dropna=False):
        y = sub["y"].astype(int)
        yh = sub["y_hat"].astype(int)
        tp = int(((y == 1) & (yh == 1)).sum())
        fp = int(((y == 0) & (yh == 1)).sum())
        fn = int(((y == 1) & (yh == 0)).sum())
        tn = int(((y == 0) & (yh == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        out[str(g)] = {"tpr": tpr, "fpr": fpr, "ppv": ppv, "n": len(sub)}
    return out


def equal_opportunity_gap(df: pd.DataFrame, group_col: str) -> Optional[float]:
    """Max TPR − min TPR across groups (needs labeled rows)."""
    if "y" not in df.columns or df["y"].notna().sum() < 2:
        return None
    d = df.dropna(subset=["y"])
    if len(d) < 2:
        return None
    cm = _confusion_by_group(d, group_col)
    tprs = [v["tpr"] for v in cm.values() if not math.isnan(v["tpr"])]
    if len(tprs) < 2:
        return None
    return float(max(tprs) - min(tprs))


def equalized_odds_gap(df: pd.DataFrame, group_col: str) -> Optional[Dict[str, float]]:
    """Report max TPR gap and max FPR gap (needs labels)."""
    if "y" not in df.columns or df["y"].notna().sum() < 2:
        return None
    d = df.dropna(subset=["y"])
    cm = _confusion_by_group(d, group_col)
    tprs = [v["tpr"] for v in cm.values() if not math.isnan(v["tpr"])]
    fprs = [v["fpr"] for v in cm.values() if not math.isnan(v["fpr"])]
    if len(tprs) < 2 and len(fprs) < 2:
        return None
    return {
        "tpr_gap": float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0,
        "fpr_gap": float(max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0,
    }


def predictive_parity_gap(df: pd.DataFrame, group_col: str) -> Optional[float]:
    """Max PPV − min PPV across groups (needs labels)."""
    if "y" not in df.columns or df["y"].notna().sum() < 2:
        return None
    d = df.dropna(subset=["y"])
    cm = _confusion_by_group(d, group_col)
    ppvs = [v["ppv"] for v in cm.values() if not math.isnan(v["ppv"])]
    if len(ppvs) < 2:
        return None
    return float(max(ppvs) - min(ppvs))


def compute_metrics_for_window(
    rows: List[Dict[str, Any]],
    split_attribute: str,
) -> Dict[str, Any]:
    """
    rows: each has prediction (int), analysis_meta (dict), optional actual_outcome (0/1).
    split_attribute: us_region | applicant_sex | race_bucket | income_bracket | tract_minority_bucket | intersection
    """
    if not rows:
        return {"error": "no_rows", "n": 0}

    flat: List[Dict[str, Any]] = []
    for r in rows:
        meta = r.get("analysis_meta") or {}
        if isinstance(meta, str):
            meta = json.loads(meta)
        y_hat = int(r["prediction"])
        rec = {
            "y_hat": y_hat,
            "us_region": meta.get("us_region", "Unknown"),
            "applicant_sex": meta.get("applicant_sex", "Unknown"),
            "race_bucket": meta.get("race_bucket", "Unknown"),
            "income_bracket": meta.get("income_bracket", "Unknown"),
            "tract_minority_bucket": meta.get("tract_minority_bucket", "Unknown"),
            "intersection_key": meta.get("intersection_key", "Unknown"),
            "analysis_meta": meta,
        }
        ao = r.get("actual_outcome")
        rec["y"] = int(ao) if ao is not None else None
        flat.append(rec)

    df = pd.DataFrame(flat)
    if split_attribute not in df.columns and split_attribute != "intersection":
        return {"error": "bad_attribute", "n": len(df)}

    gcol = "intersection_key" if split_attribute == "intersection" else split_attribute
    rates = positive_rates(df, gcol)
    labeled = df["y"].notna()
    n_labeled = int(labeled.sum())

    out: Dict[str, Any] = {
        "n": len(df),
        "n_labeled": n_labeled,
        "split_attribute": split_attribute,
        "selection_rates_by_group": rates.to_dict(),
        "demographic_parity_gap": demographic_parity_gap(df, gcol),
        "disparate_impact_ratio": disparate_impact_ratio(df, gcol),
        "equal_opportunity_tpr_gap": equal_opportunity_gap(df, gcol) if n_labeled >= 2 else None,
        "equalized_odds": equalized_odds_gap(df, gcol) if n_labeled >= 2 else None,
        "predictive_parity_ppv_gap": predictive_parity_gap(df, gcol) if n_labeled >= 2 else None,
    }
    return out


METRIC_DESCRIPTIONS = [
    {
        "key": "demographic_parity_gap",
        "title": "Demographic parity",
        "text": "Compares the rate of positive predictions across groups. The gap is max rate minus min rate; smaller is better.",
    },
    {
        "key": "disparate_impact_ratio",
        "title": "Disparate impact (selection-rate ratio)",
        "text": "Ratio of the smallest group positive rate to the largest (min/max). Values near 1.0 are balanced; 0.8 is a common rule-of-thumb floor.",
    },
    {
        "key": "equal_opportunity_tpr_gap",
        "title": "Equal opportunity (TPR parity)",
        "text": "True positive rate (recall among qualified applicants) should be similar across groups. Needs observed outcomes (actual_outcome) in the window.",
    },
    {
        "key": "equalized_odds",
        "title": "Equalized odds",
        "text": "Both TPR and FPR should align across groups. Reported as separate TPR and FPR gaps. Needs labels.",
    },
    {
        "key": "predictive_parity_ppv_gap",
        "title": "Predictive parity (PPV)",
        "text": "Among predicted approvals, the fraction that are truly positive (precision) should be similar across groups. Needs labels.",
    },
]

ATTRIBUTE_OPTIONS = [
    ("us_region", "US region (from state)"),
    ("applicant_sex", "Applicant sex"),
    ("race_bucket", "Race (simplified from applicant_race_name_1)"),
    ("income_bracket", "Income (low vs high, $000s)"),
    ("tract_minority_bucket", "Tract minority share"),
    ("intersection_key", "Intersection (sex × race × income × region)"),
]
