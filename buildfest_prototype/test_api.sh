#!/usr/bin/env bash
# Curl-based smoke tests for the Flask fairness API (see README Testing section).
set -euo pipefail

API_BASE="${API_BASE:-http://127.0.0.1:5050}"
API_BASE="${API_BASE%/}"

echo "== GET /health"
curl -sf "${API_BASE}/health" | python3 -m json.tool

echo "== POST /predict"
curl -sf -X POST "${API_BASE}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
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
      "native-country": "United-States"
    },
    "group_label": "A"
  }' | python3 -m json.tool

echo "== POST /counterfactual_test"
curl -sf -X POST "${API_BASE}/counterfactual_test" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
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
      "native-country": "United-States"
    }
  }' | python3 -m json.tool

echo "All curl checks passed."
