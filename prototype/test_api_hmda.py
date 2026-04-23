#!/usr/bin/env python3
"""Smoke-test the HMDA stack (run inside compose or against localhost:5051)."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE", "http://127.0.0.1:5051").rstrip("/")

SAMPLE_FEATURES = {
    "loan_amount_000s": 320.0,
    "applicant_income_000s": 85.0,
    "loan_type": 1,
    "property_type": 1,
    "applicant_ethnicity": 2,
    "applicant_race_1": 5,
    "applicant_sex": 1,
    "hoepa_status": 2,
    "lien_status": 1,
    "population": 4000.0,
    "minority_population": 12.5,
    "state_name": "CA",
    "loan_type_name": "Conventional",
    "applicant_ethnicity_name": "Not Hispanic",
    "applicant_race_name_1": "White",
}


def _request(method: str, path: str, data: dict | None = None) -> tuple[int, dict]:
    url = f"{BASE}{path}"
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            return resp.status, payload
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(err_body) if err_body else {}
        except json.JSONDecodeError:
            payload = {"_raw": err_body}
        print(f"HTTP {e.code}: {payload}", file=sys.stderr)
        return e.code, payload
    except urllib.error.URLError as e:
        print(f"Connection failed ({url}): {e.reason}", file=sys.stderr)
        return 0, {"error": str(e.reason)}


def main() -> int:
    failed: list[str] = []

    code, h = _request("GET", "/health")
    print("health:", code, h)
    if code != 200 or h.get("status") != "ok":
        failed.append("GET /health")

    code, p = _request("POST", "/predict", {"features": SAMPLE_FEATURES, "group_label": "A"})
    print("predict:", code, {k: p.get(k) for k in ("prediction", "request_id") if k in p})
    if code != 200 or "prediction" not in p:
        failed.append("POST /predict")

    code, e = _request("POST", "/explain", {"features": SAMPLE_FEATURES})
    print("explain:", code, "prediction" in e)
    if code != 200 or "prediction" not in e:
        failed.append("POST /explain")

    code, c = _request("POST", "/counterfactual_test", {"features": SAMPLE_FEATURES})
    print("counterfactual:", code, {k: c.get(k) for k in ("changed_due_to_attribute",) if k in c})
    if code != 200 or "original_prediction" not in c:
        failed.append("POST /counterfactual_test")

    if failed:
        print("FAILED:", ", ".join(failed), file=sys.stderr)
        return 1
    print("All HMDA API checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
