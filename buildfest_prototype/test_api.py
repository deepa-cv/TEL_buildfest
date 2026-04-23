#!/usr/bin/env python3
"""
Smoke-test the fairness Flask API (health + predict + counterfactual + explain).

Usage:
  export API_BASE=http://127.0.0.1:5050   # optional
  python test_api.py

Requires `app/model.pkl` and a running stack: `docker compose up --build`.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE", "http://127.0.0.1:5050").rstrip("/")

SAMPLE_FEATURES = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


def _request(method: str, path: str, data: dict | None = None) -> tuple[int, dict]:
    url = f"{BASE}{path}"
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
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


def test_health() -> bool:
    code, body = _request("GET", "/health")
    ok = code == 200 and body.get("status") == "ok"
    print("health:", code, body)
    return ok


def test_predict() -> bool:
    code, body = _request("POST", "/predict", {"features": SAMPLE_FEATURES, "group_label": "A"})
    ok = code == 200 and "prediction" in body
    print("predict:", code, {k: body.get(k) for k in ("request_id", "prediction") if k in body})
    return ok


def test_explain() -> bool:
    code, body = _request("POST", "/explain", {"features": SAMPLE_FEATURES})
    ok = code == 200 and "prediction" in body
    print("explain:", code, "prediction" in body)
    return ok


def test_counterfactual() -> bool:
    code, body = _request("POST", "/counterfactual_test", {"features": SAMPLE_FEATURES})
    ok = code == 200 and "original_prediction" in body
    print("counterfactual:", code, {k: body.get(k) for k in ("changed_due_to_attribute",) if k in body})
    return ok


def main() -> int:
    checks = [
        ("GET /health", test_health),
        ("POST /predict", test_predict),
        ("POST /explain", test_explain),
        ("POST /counterfactual_test", test_counterfactual),
    ]
    failed = []
    for name, fn in checks:
        try:
            if not fn():
                failed.append(name)
        except Exception as e:
            print(f"{name} raised: {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed), file=sys.stderr)
        return 1
    print("All API checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
