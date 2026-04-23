import sys
from pathlib import Path

# Ensure prototype root on path for `import group_fairness`
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from group_fairness import build_analysis_meta, compute_metrics_for_window


def test_build_analysis_meta_regions():
    m = build_analysis_meta(
        {
            "state_name": "CA",
            "applicant_sex": 2,
            "applicant_race_name_1": "Asian",
            "applicant_race_1": 2,
            "applicant_income_000s": 120.0,
            "minority_population": 45.0,
        }
    )
    assert m["us_region"] == "West"
    assert m["applicant_sex"] == "Female"
    assert m["income_bracket"] == "high_income"
    assert m["tract_minority_bucket"] == "high_minority_tract"


def test_sliding_window_dp_di():
    rows = []
    for i in range(30):
        st = "CA" if i < 15 else "NY"
        feat = {
            "state_name": st,
            "applicant_sex": 1,
            "applicant_race_name_1": "White",
            "applicant_race_1": 5,
            "applicant_income_000s": 90.0,
            "minority_population": 12.0,
        }
        rows.append(
            {
                "prediction": 1 if i % 3 != 0 else 0,
                "analysis_meta": build_analysis_meta(feat),
                "actual_outcome": None,
            }
        )
    out = compute_metrics_for_window(rows, "us_region")
    assert out["n"] == 30
    assert "demographic_parity_gap" in out
    assert out["disparate_impact_ratio"] is not None
    assert out["equal_opportunity_tpr_gap"] is None


def test_metrics_with_labels_for_eo():
    rows = []
    for i in range(24):
        feat = {
            "state_name": "TX",
            "applicant_sex": 1 if i < 12 else 2,
            "applicant_race_name_1": "White",
            "applicant_race_1": 5,
            "applicant_income_000s": 70.0,
            "minority_population": 20.0,
        }
        rows.append(
            {
                "prediction": 1 if i % 2 == 0 else 0,
                "analysis_meta": build_analysis_meta(feat),
                "actual_outcome": 1 if i % 2 == 0 else 0,
            }
        )
    out = compute_metrics_for_window(rows, "applicant_sex")
    assert out["n_labeled"] == 24
    assert out["equal_opportunity_tpr_gap"] is not None
    assert out["equalized_odds"] is not None
