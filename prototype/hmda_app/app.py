"""
HMDA mortgage API + customer / internal web UI.
"""

from __future__ import annotations

import json
import os
import uuid
from functools import wraps
import urllib.error
import urllib.request

import joblib
import numpy as np
import pandas as pd
import psycopg2
import shap
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from psycopg2.extras import RealDictCursor
from sklearn.pipeline import Pipeline

from group_fairness import (
    ATTRIBUTE_OPTIONS,
    METRIC_DESCRIPTIONS,
    build_analysis_meta,
    compute_metrics_for_window,
)
from hmda_api_features import (
    COUNTERFACTUAL_ATTRIBUTE,
    CUSTOMER_FORM_FIELDS,
    FIELD_SELECT_OPTIONS,
    FLOAT_FIELDS,
    INTEGER_FIELDS,
    REQUIRED_FIELDS,
)

# One valid HMDA row for SHAP PermutationExplainer background (must match pipeline dtypes).
SHAP_BACKGROUND_FEATURES: dict = {
    "loan_amount_000s": 300.0,
    "applicant_income_000s": 80.0,
    "loan_type": 1,
    "property_type": 1,
    "applicant_ethnicity": 2,
    "applicant_race_1": 5,
    "applicant_sex": 1,
    "hoepa_status": 2,
    "lien_status": 1,
    "population": 4000.0,
    "minority_population": 15.0,
    "state_name": "CA",
    "loan_type_name": "Conventional",
    "applicant_ethnicity_name": "Not Hispanic or Latino",
    "applicant_race_name_1": "White",
}


def _strip_text(s: str) -> str:
    """Strip whitespace and common invisible Unicode (ZWSP, BOM) from form/JSON text."""
    t = s.strip()
    for ch in ("\u200b", "\u200c", "\u200d", "\ufeff"):
        t = t.replace(ch, "")
    return t


def normalize_features(feat: dict) -> dict:
    """
    Strip string fields and coerce numeric columns to int/float.
    JSON clients often send numbers as strings; sklearn's scaler would otherwise raise
    TypeError (unsupported operand type(s) for -: 'str' and 'float').
    """
    out = dict(feat)
    for k, v in list(out.items()):
        if isinstance(v, str):
            out[k] = _strip_text(v)
    for k in INTEGER_FIELDS:
        if k not in out:
            continue
        v = out[k]
        if v == "" or v is None:
            raise ValueError(f"Invalid empty value for {k}")
        try:
            out[k] = int(float(v))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {k} (expected integer code): {feat.get(k)!r}") from e
    for k in FLOAT_FIELDS:
        if k not in out:
            continue
        v = out[k]
        if v == "" or v is None:
            raise ValueError(f"Invalid empty value for {k}")
        try:
            out[k] = float(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid value for {k} (expected number): {feat.get(k)!r}") from e
    return out


def build_feature_dataframe(feat_dict: dict) -> pd.DataFrame:
    """
    One row in strict REQUIRED_FIELDS order with explicit numeric dtypes.
    A plain pd.DataFrame([dict]) often uses object dtype for mixed columns; sklearn's
    StandardScaler then raises TypeError on str - float.
    """
    row = normalize_features(dict(feat_dict))
    df = pd.DataFrame([{c: row[c] for c in REQUIRED_FIELDS}], columns=list(REQUIRED_FIELDS))
    for c in INTEGER_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    for c in FLOAT_FIELDS:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    return df


app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-hmda-change-me-in-production")

INTERNAL_USERNAME = os.environ.get("INTERNAL_USERNAME", "admin")
INTERNAL_PASSWORD = os.environ.get("INTERNAL_PASSWORD", "changeme")
GRAFANA_PUBLIC_URL = os.environ.get("GRAFANA_PUBLIC_URL", "http://localhost:3001")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "hmda-v1")
FAIRNESS_DP_GAP_MAX = float(os.environ.get("FAIRNESS_DP_GAP_MAX", "0.15"))
FAIRNESS_DI_MIN = float(os.environ.get("FAIRNESS_DI_MIN", "0.80"))
FAIRNESS_SLACK_WEBHOOK = os.environ.get("FAIRNESS_SLACK_WEBHOOK", "").strip()

MODEL_PATH = os.getenv("MODEL_PATH", "model_hmda.pkl")
model = joblib.load(MODEL_PATH)


def _permutation_explainer_fallback():
    """
    PermutationExplainer(data=DataFrame) uses a Tabular masker that calls np.isclose on
    background vs row — that raises TypeError on string categoricals. Independent masker
    does not, and matches sklearn pipelines with object/string columns.
    """
    bg = build_feature_dataframe(SHAP_BACKGROUND_FEATURES)
    return shap.PermutationExplainer(model.predict, shap.maskers.Independent(bg))


if isinstance(model, Pipeline):
    underlying_model = model.steps[-1][1]
    print(f"✔ Pipeline, underlying estimator: {type(underlying_model).__name__}")
    try:
        explainer = shap.TreeExplainer(underlying_model)
        print("✔ TreeExplainer")
    except Exception as e:
        print(f"⚠ TreeExplainer failed: {e}, using PermutationExplainer + Independent masker")
        explainer = _permutation_explainer_fallback()
else:
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = _permutation_explainer_fallback()

print("✔ Model + SHAP ready (numeric row coercion + SHAP background row active)", flush=True)


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        dbname=os.getenv("DB_NAME", "fairness"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres"),
    )


def run_prediction(feat_dict: dict) -> int:
    df = build_feature_dataframe(feat_dict)
    return int(model.predict(df)[0])


def map_to_original_features(transformed_feature_names, original_feature_names):
    feature_mapping = {}
    for tf_name in transformed_feature_names:
        original_feat = None
        for orig_feat in original_feature_names:
            if tf_name.startswith(orig_feat + "_") or tf_name == orig_feat:
                original_feat = orig_feat
                break
        if original_feat is None and tf_name in original_feature_names:
            original_feat = tf_name
        if original_feat is None:
            parts = tf_name.split("__")
            for part in reversed(parts):
                if part in original_feature_names:
                    original_feat = part
                    break
                for orig in original_feature_names:
                    if part.startswith(orig + "_"):
                        original_feat = orig
                        break
                if original_feat:
                    break
        feature_mapping[tf_name] = original_feat if original_feat else tf_name
    return feature_mapping


def aggregate_shap_by_original_features(shap_vals, transformed_features, original_features):
    feature_mapping = map_to_original_features(transformed_features, original_features)
    aggregated = {}
    for i, tf_feat in enumerate(transformed_features):
        orig_feat = feature_mapping.get(tf_feat, tf_feat)
        if orig_feat not in aggregated:
            aggregated[orig_feat] = []
        aggregated[orig_feat].append(shap_vals[i])
    aggregated_shap = {}
    for orig_feat, vals in aggregated.items():
        if len(vals) == 1:
            aggregated_shap[orig_feat] = vals[0]
        else:
            aggregated_shap[orig_feat] = sum(abs(v) for v in vals)
    return aggregated_shap, feature_mapping


def compute_shap(feat_dict):
    df = build_feature_dataframe(feat_dict)
    original_features = df.columns.tolist()

    if isinstance(explainer, shap.TreeExplainer) and isinstance(model, Pipeline):
        transformed_data = df.copy()
        for _, step_transformer in model.steps[:-1]:
            transformed = step_transformer.transform(transformed_data)
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            if isinstance(transformed, (list, tuple)):
                transformed = np.array(transformed)
            if hasattr(step_transformer, "get_feature_names_out"):
                feature_names = step_transformer.get_feature_names_out(transformed_data.columns)
            elif hasattr(step_transformer, "get_feature_names"):
                feature_names = step_transformer.get_feature_names(transformed_data.columns)
            else:
                if transformed.ndim == 2:
                    feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
                else:
                    feature_names = [f"feature_{i}" for i in range(len(transformed))]
            transformed_data = pd.DataFrame(transformed, columns=feature_names)

        shap_result = explainer.shap_values(transformed_data)
        if isinstance(shap_result, list):
            shap_vals = shap_result[0]
        else:
            shap_vals = shap_result[0] if shap_result.ndim > 1 else shap_result
        if isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim > 1:
                shap_vals = shap_vals.flatten()
            shap_vals = shap_vals.tolist()
        transformed_features = transformed_data.columns.tolist()
        aggregated_shap, _ = aggregate_shap_by_original_features(
            shap_vals, transformed_features, original_features
        )
        features = list(aggregated_shap.keys())
        shap_vals = list(aggregated_shap.values())
    else:
        try:
            shap_result = explainer.shap_values(df)
        except (TypeError, ValueError) as err:
            # e.g. older SHAP + Tabular masker + string columns
            app.logger.warning("SHAP permutation path failed (%s); using zero placeholder", err)
            features = df.columns.tolist()
            shap_vals = [0.0] * len(features)
        else:
            if isinstance(shap_result, list):
                shap_vals = shap_result[0]
            else:
                shap_vals = shap_result[0] if shap_result.ndim > 1 else shap_result
            if isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim > 1:
                    shap_vals = shap_vals.flatten()
                shap_vals = shap_vals.tolist()
            features = df.columns.tolist()

    if len(features) != len(shap_vals):
        features = [f"feature_{i}" for i in range(len(shap_vals))]

    abs_vals = [abs(v) for v in shap_vals]
    top_idx = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i], reverse=True)
    top_features = [{"feature": features[i], "shap_value": float(shap_vals[i])} for i in top_idx]
    return shap_vals, top_features


def flip_applicant_sex(v):
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return v
    if iv == 1:
        return 2
    if iv == 2:
        return 1
    return 3 if iv != 3 else 4


def parse_customer_form(form) -> dict:
    raw = {}
    for name in REQUIRED_FIELDS:
        v = form.get(name)
        if v is None:
            raise ValueError(f"Missing or empty field: {name}")
        if isinstance(v, str) and not _strip_text(v):
            raise ValueError(f"Missing or empty field: {name}")
        raw[name] = v
    return normalize_features(raw)


def customer_decision_label(prediction: int, fairness_passed: bool) -> str:
    if not fairness_passed:
        return "pending"
    return "approved" if prediction == 1 else "rejected"


def insert_decision_log(
    cur,
    request_id: str,
    features_for_storage: dict,
    prediction: int,
    shap_vals,
    top_features,
    fairness_passed: bool | None,
    ui_outcome: str | None,
    analysis_meta: dict | None = None,
    actual_outcome: int | None = None,
    model_version: str | None = None,
):
    mv = model_version or MODEL_VERSION
    cur.execute(
        """
        INSERT INTO decision_logs (
            request_id, model_version, features, prediction, shap_values, top_features,
            fairness_passed, ui_outcome, analysis_meta, actual_outcome
        )
        VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s, %s, %s::jsonb, %s)
        """,
        (
            request_id,
            mv,
            json.dumps(features_for_storage),
            prediction,
            json.dumps(shap_vals),
            json.dumps(top_features),
            fairness_passed,
            ui_outcome,
            json.dumps(analysis_meta) if analysis_meta is not None else None,
            actual_outcome,
        ),
    )


def _maybe_fairness_alerts(metrics: dict) -> None:
    if metrics.get("error") or not metrics.get("n"):
        return
    dp = metrics.get("demographic_parity_gap")
    di = metrics.get("disparate_impact_ratio")
    alerts = []
    if dp is not None and dp > FAIRNESS_DP_GAP_MAX:
        alerts.append(
            ("warning", "demographic_parity_gap", f"DP gap {dp:.4f} exceeds {FAIRNESS_DP_GAP_MAX}", {"value": dp})
        )
    if di is not None and di < FAIRNESS_DI_MIN:
        alerts.append(
            ("warning", "disparate_impact_ratio", f"DI ratio {di:.4f} below {FAIRNESS_DI_MIN}", {"value": di})
        )
    if not alerts:
        return
    conn = get_conn()
    cur = conn.cursor()
    for sev, name, msg, details in alerts:
        cur.execute(
            """
            INSERT INTO fairness_alerts (severity, metric_name, message, details)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (sev, name, msg, json.dumps(details)),
        )
    conn.commit()
    cur.close()
    conn.close()
    if FAIRNESS_SLACK_WEBHOOK:
        for _sev, _name, msg, _d in alerts:
            try:
                body = json.dumps({"text": f"[HMDA Fairness] {msg}"}).encode("utf-8")
                req = urllib.request.Request(
                    FAIRNESS_SLACK_WEBHOOK, data=body, method="POST", headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=5)
            except (urllib.error.URLError, OSError):
                pass


def _load_window_rows(model_version: str, window_hours: float, row_limit: int) -> list:
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    if window_hours and window_hours > 0:
        cur.execute(
            """
            SELECT prediction, analysis_meta, actual_outcome, model_version, timestamp
            FROM decision_logs
            WHERE analysis_meta IS NOT NULL AND model_version = %s
              AND timestamp >= NOW() - %s::interval
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (model_version, f"{float(window_hours)} hours", int(row_limit)),
        )
    else:
        cur.execute(
            """
            SELECT prediction, analysis_meta, actual_outcome, model_version, timestamp
            FROM decision_logs
            WHERE analysis_meta IS NOT NULL AND model_version = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """,
            (model_version, int(row_limit)),
        )
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def _persist_snapshot(model_version: str, window_hours: float, row_limit: int, attr: str, metrics: dict) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO fairness_snapshots (model_version, window_hours, row_limit, split_attribute, metrics_json)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        """,
        (model_version, window_hours, row_limit, attr, json.dumps(metrics)),
    )
    conn.commit()
    cur.close()
    conn.close()


def internal_login_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("internal_ok"):
            return redirect(url_for("internal_login", next=request.path))
        return f(*args, **kwargs)

    return wrapped


# --- Customer UI ---


@app.route("/")
def home():
    return redirect(url_for("customer_apply"))


@app.route("/apply", methods=["GET", "POST"])
def customer_apply():
    sections: dict[str, list] = {}
    for name, _typ, label, sec in CUSTOMER_FORM_FIELDS:
        sections.setdefault(sec, []).append((name, _typ, label))

    if request.method == "GET":
        return render_template(
            "customer_apply.html", sections=sections, select_options=FIELD_SELECT_OPTIONS
        )

    try:
        features = parse_customer_form(request.form)
    except ValueError as e:
        return (
            render_template(
                "customer_apply.html",
                sections=sections,
                select_options=FIELD_SELECT_OPTIONS,
                error=str(e),
            ),
            400,
        )

    request_id = str(uuid.uuid4())
    attr = COUNTERFACTUAL_ATTRIBUTE

    try:
        prediction = run_prediction(features)
        cf_features = features.copy()
        cf_features[attr] = flip_applicant_sex(features[attr])
        cf_pred = run_prediction(cf_features)
        changed = prediction != cf_pred
        fairness_passed = not changed
        shap_vals, top_features = compute_shap(features)
        ui_outcome = customer_decision_label(prediction, fairness_passed)

        storage_features = {
            **features,
            "_fairness_passed": fairness_passed,
            "_counterfactual_prediction": cf_pred,
            "_changed_due_to_sex": changed,
        }

        conn = get_conn()
        cur = conn.cursor()
        insert_decision_log(
            cur,
            request_id,
            storage_features,
            prediction,
            shap_vals,
            top_features,
            fairness_passed,
            ui_outcome,
            analysis_meta=build_analysis_meta(features),
            actual_outcome=None,
            model_version=MODEL_VERSION,
        )
        cur.execute(
            """
            INSERT INTO counterfactual_logs (
                request_id, original_sex, counterfactual_sex,
                original_prediction, counterfactual_prediction, changed
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                request_id,
                str(features.get(attr)),
                str(cf_features.get(attr)),
                prediction,
                cf_pred,
                changed,
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
    except ValueError as ex:
        return (
            render_template(
                "customer_apply.html",
                sections=sections,
                select_options=FIELD_SELECT_OPTIONS,
                error=str(ex),
            ),
            400,
        )
    except Exception as ex:
        app.logger.exception("customer_apply failed after validation")
        return (
            render_template(
                "customer_apply.html",
                sections=sections,
                select_options=FIELD_SELECT_OPTIONS,
                error=f"Unable to complete review: {ex}",
            ),
            500,
        )

    return render_template(
        "customer_result.html",
        request_id=request_id,
        prediction=prediction,
        fairness_passed=fairness_passed,
        changed_due_to_attribute=changed,
        ui_outcome=ui_outcome,
        counterfactual_prediction=cf_pred,
    )


# --- Internal UI ---


def _safe_internal_redirect(target: str | None) -> str:
    if target and target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("internal_dashboard")


@app.route("/internal/login", methods=["GET", "POST"])
def internal_login():
    nxt = _safe_internal_redirect(request.args.get("next"))
    if request.method == "POST":
        nxt = _safe_internal_redirect(request.form.get("next"))
        u = request.form.get("username", "")
        p = request.form.get("password", "")
        if u == INTERNAL_USERNAME and p == INTERNAL_PASSWORD:
            session["internal_ok"] = True
            return redirect(nxt)
        return render_template("internal_login.html", error="Invalid credentials", next_url=nxt)
    return render_template("internal_login.html", next_url=nxt)


@app.route("/internal/logout", methods=["POST"])
def internal_logout():
    session.pop("internal_ok", None)
    return redirect(url_for("customer_apply"))


@app.route("/internal")
@internal_login_required
def internal_dashboard():
    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT timestamp, request_id, prediction, fairness_passed, ui_outcome
        FROM decision_logs
        ORDER BY timestamp DESC
        LIMIT 40
        """
    )
    decisions = cur.fetchall()
    cur.execute(
        """
        SELECT timestamp, request_id, original_sex, counterfactual_sex,
               original_prediction, counterfactual_prediction, changed
        FROM counterfactual_logs
        ORDER BY timestamp DESC
        LIMIT 40
        """
    )
    counterfactuals = cur.fetchall()
    cur.close()
    conn.close()

    failed_cf = sum(1 for r in counterfactuals if r.get("changed"))
    return render_template(
        "internal_dashboard.html",
        decisions=decisions,
        counterfactuals=counterfactuals,
        failed_cf=failed_cf,
        grafana_url=GRAFANA_PUBLIC_URL,
    )


@app.route("/internal/grafana")
@internal_login_required
def internal_grafana():
    return render_template("internal_grafana.html", grafana_url=GRAFANA_PUBLIC_URL)


@app.route("/internal/fairness", methods=["GET"])
@internal_login_required
def internal_fairness():
    """Sliding-window group metrics + optional two-model comparison (filter by model_version in logs)."""
    window_hours = float(request.args.get("hours", "24") or 0)
    row_limit = int(request.args.get("limit", "500") or 500)
    attr = request.args.get("attr", "us_region") or "us_region"
    mv1 = request.args.get("mv1", MODEL_VERSION) or MODEL_VERSION
    mv2 = request.args.get("mv2", "") or ""

    rows1 = _load_window_rows(mv1, window_hours, row_limit)
    m1 = compute_metrics_for_window(rows1, attr)
    if request.args.get("check_alerts") == "1":
        _maybe_fairness_alerts(m1)
    if request.args.get("snapshot") == "1":
        _persist_snapshot(mv1, window_hours, row_limit, attr, m1)

    m2 = None
    rows2 = []
    if mv2 and mv2 != mv1:
        rows2 = _load_window_rows(mv2, window_hours, row_limit)
        m2 = compute_metrics_for_window(rows2, attr)
        if request.args.get("check_alerts") == "1":
            _maybe_fairness_alerts(m2)
        if request.args.get("snapshot") == "1":
            _persist_snapshot(mv2, window_hours, row_limit, attr, m2)

    conn = get_conn()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(
        """
        SELECT created_at, severity, metric_name, message
        FROM fairness_alerts
        ORDER BY created_at DESC
        LIMIT 25
        """
    )
    recent_alerts = cur.fetchall()
    cur.execute(
        """
        SELECT calc_time, model_version, window_hours, row_limit, split_attribute, metrics_json
        FROM fairness_snapshots
        ORDER BY calc_time DESC
        LIMIT 15
        """
    )
    snapshots = cur.fetchall()
    cur.close()
    conn.close()

    return render_template(
        "internal_fairness.html",
        metric_descriptions=METRIC_DESCRIPTIONS,
        attribute_options=ATTRIBUTE_OPTIONS,
        window_hours=window_hours,
        row_limit=row_limit,
        attr=attr,
        mv1=mv1,
        mv2=mv2,
        metrics_a=m1,
        metrics_b=m2,
        recent_alerts=recent_alerts,
        snapshots=snapshots,
        dp_max=FAIRNESS_DP_GAP_MAX,
        di_min=FAIRNESS_DI_MIN,
    )


# --- JSON API (unchanged contract + richer logging when columns exist) ---


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "hmda"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    features = data.get("features", {})

    if not features:
        return jsonify({"error": "Missing 'features'"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        features = normalize_features(dict(features))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prediction = run_prediction(features)
    request_id = str(uuid.uuid4())
    shap_vals, top_features = compute_shap(features)

    ao = data.get("actual_outcome")
    if ao is not None:
        ao = int(ao)
    mv_req = (data.get("model_version") or "").strip() or MODEL_VERSION

    conn = get_conn()
    cur = conn.cursor()
    insert_decision_log(
        cur,
        request_id,
        features,
        prediction,
        shap_vals,
        top_features,
        None,
        None,
        analysis_meta=build_analysis_meta(features),
        actual_outcome=ao,
        model_version=mv_req,
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"request_id": request_id, "prediction": prediction, "top_features": top_features}), 200


@app.route("/counterfactual_test", methods=["POST"])
def counterfactual_test():
    data = request.json or {}
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features'"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        features = normalize_features(dict(features))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    attr = COUNTERFACTUAL_ATTRIBUTE
    original_pred = run_prediction(features)
    cf_features = features.copy()
    cf_features[attr] = flip_applicant_sex(features[attr])
    cf_pred = run_prediction(cf_features)
    changed = original_pred != cf_pred
    request_id = str(uuid.uuid4())

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO counterfactual_logs (
            request_id, original_sex, counterfactual_sex,
            original_prediction, counterfactual_prediction, changed
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            request_id,
            str(features.get(attr)),
            str(cf_features.get(attr)),
            original_pred,
            cf_pred,
            changed,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify(
        {
            "request_id": request_id,
            "attribute_tested": attr,
            "original_prediction": original_pred,
            "counterfactual_prediction": cf_pred,
            "changed_due_to_attribute": changed,
        }
    ), 200


@app.route("/explain", methods=["POST"])
def explain():
    data = request.json or {}
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features'"}), 400

    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        features = normalize_features(dict(features))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prediction = run_prediction(features)
    request_id = str(uuid.uuid4())
    shap_vals, top_features = compute_shap(features)

    ao = data.get("actual_outcome")
    if ao is not None:
        ao = int(ao)
    mv_req = (data.get("model_version") or "").strip() or MODEL_VERSION

    conn = get_conn()
    cur = conn.cursor()
    insert_decision_log(
        cur,
        request_id,
        features,
        prediction,
        shap_vals,
        top_features,
        None,
        None,
        analysis_meta=build_analysis_meta(features),
        actual_outcome=ao,
        model_version=mv_req,
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify(
        {
            "request_id": request_id,
            "prediction": prediction,
            "shap_values": shap_vals,
            "top_features": top_features,
        }
    ), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
