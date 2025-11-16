from flask import Flask, request, jsonify, render_template
import psycopg2
import os
import uuid
import pandas as pd
import numpy as np
import joblib
import json
import shap

app = Flask(__name__)

# ------------------------------------------------------------
# REQUIRED FIELDS (Adult dataset)
# ------------------------------------------------------------
REQUIRED_FIELDS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
]

# ------------------------------------------------------------
# LOAD MODEL + SHAP EXPLAINER
# ------------------------------------------------------------
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# Handle Pipeline models - extract the underlying model for TreeExplainer
from sklearn.pipeline import Pipeline
if isinstance(model, Pipeline):
    # Get the final estimator from the pipeline
    underlying_model = model.steps[-1][1]
    print(f"✔ Model is a Pipeline, using underlying model: {type(underlying_model).__name__}")
    try:
        explainer = shap.TreeExplainer(underlying_model)
        print("✔ Using TreeExplainer")
    except Exception as e:
        print(f"⚠ TreeExplainer failed: {e}, falling back to PermutationExplainer")
        # Create a sample dataframe for background data
        sample_data = pd.DataFrame([{f: 0 for f in REQUIRED_FIELDS}])
        explainer = shap.PermutationExplainer(model.predict, sample_data)
else:
    try:
        explainer = shap.TreeExplainer(model)
        print("✔ Using TreeExplainer")
    except Exception as e:
        print(f"⚠ TreeExplainer failed: {e}, falling back to PermutationExplainer")
        # Create a sample dataframe for background data
        sample_data = pd.DataFrame([{f: 0 for f in REQUIRED_FIELDS}])
        explainer = shap.PermutationExplainer(model.predict, sample_data)

print("✔ Loaded ML model + SHAP explainer")


# ------------------------------------------------------------
# DB CONNECTION HELPER
# ------------------------------------------------------------
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        dbname=os.getenv("DB_NAME", "fairness"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres")
    )


# ------------------------------------------------------------
# SHARED HELPERS
# ------------------------------------------------------------
def normalize_features(feat_dict):
    if "sex" in feat_dict and isinstance(feat_dict["sex"], str):
        feat_dict["sex"] = feat_dict["sex"].strip().title()
    if "race" in feat_dict and isinstance(feat_dict["race"], str):
        feat_dict["race"] = feat_dict["race"].strip().title()
    return feat_dict


def run_prediction(feat_dict):
    df = pd.DataFrame([feat_dict])
    return int(model.predict(df)[0])


def map_to_original_features(transformed_feature_names, original_feature_names):
    """
    Map transformed feature names back to original feature names.
    Handles one-hot encoding and other transformations.
    """
    feature_mapping = {}
    
    for tf_name in transformed_feature_names:
        # Try to extract original feature name from transformed name
        # Common patterns: "feature_original", "original_category", etc.
        original_feat = None
        
        # Check if it's a one-hot encoded feature (e.g., "workclass_State-gov")
        for orig_feat in original_feature_names:
            if tf_name.startswith(orig_feat + "_") or tf_name == orig_feat:
                original_feat = orig_feat
                break
        
        # If not found, check if it matches an original feature exactly
        if original_feat is None and tf_name in original_feature_names:
            original_feat = tf_name
        
        # If still not found, try to parse common transformer patterns
        if original_feat is None:
            # Some transformers create names like "pipeline__columntransformer__onehotencoder__workclass_State-gov"
            parts = tf_name.split("__")
            for part in reversed(parts):
                if part in original_feature_names:
                    original_feat = part
                    break
                # Check if it starts with an original feature name
                for orig in original_feature_names:
                    if part.startswith(orig + "_"):
                        original_feat = orig
                        break
                if original_feat:
                    break
        
        feature_mapping[tf_name] = original_feat if original_feat else tf_name
    
    return feature_mapping


def aggregate_shap_by_original_features(shap_vals, transformed_features, original_features):
    """
    Aggregate SHAP values for transformed features back to original features.
    For one-hot encoded features, sum the absolute values.
    """
    feature_mapping = map_to_original_features(transformed_features, original_features)
    
    # Aggregate SHAP values by original feature
    aggregated = {}
    for i, tf_feat in enumerate(transformed_features):
        orig_feat = feature_mapping.get(tf_feat, tf_feat)
        if orig_feat not in aggregated:
            aggregated[orig_feat] = []
        aggregated[orig_feat].append(shap_vals[i])
    
    # For each original feature, use the sum of absolute values for ranking
    # but return the actual SHAP values for display
    aggregated_shap = {}
    for orig_feat, vals in aggregated.items():
        if len(vals) == 1:
            aggregated_shap[orig_feat] = vals[0]
        else:
            # For one-hot encoded features, sum the absolute values for importance
            aggregated_shap[orig_feat] = sum(abs(v) for v in vals)
    
    return aggregated_shap, feature_mapping


def compute_shap(feat_dict):
    df = pd.DataFrame([feat_dict])
    original_features = df.columns.tolist()
    
    # Handle different explainer types
    if isinstance(explainer, shap.TreeExplainer):
        # For TreeExplainer, we need to transform the data through the pipeline first
        if isinstance(model, Pipeline):
            # Transform through all steps except the last (the estimator)
            transformed_data = df.copy()
            for step_name, step_transformer in model.steps[:-1]:
                transformed = step_transformer.transform(transformed_data)
                # Handle different transformer output types
                if hasattr(transformed, 'toarray'):  # Sparse matrix
                    transformed = transformed.toarray()
                if isinstance(transformed, (list, tuple)):
                    transformed = np.array(transformed)
                # Get feature names if available
                if hasattr(step_transformer, 'get_feature_names_out'):
                    feature_names = step_transformer.get_feature_names_out(transformed_data.columns)
                elif hasattr(step_transformer, 'get_feature_names'):
                    feature_names = step_transformer.get_feature_names(transformed_data.columns)
                else:
                    # Fallback: use number of features
                    if transformed.ndim == 2:
                        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]
                    else:
                        feature_names = [f"feature_{i}" for i in range(len(transformed))]
                transformed_data = pd.DataFrame(transformed, columns=feature_names)
            
            shap_result = explainer.shap_values(transformed_data)
            # Handle binary classification - shap_values returns list of arrays
            if isinstance(shap_result, list):
                shap_vals = shap_result[0]  # For binary classification, use first class
            else:
                shap_vals = shap_result[0] if shap_result.ndim > 1 else shap_result
            
            # Convert to list
            if isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim > 1:
                    shap_vals = shap_vals.flatten()
                shap_vals = shap_vals.tolist()
            
            # Map back to original features
            transformed_features = transformed_data.columns.tolist()
            aggregated_shap, feature_mapping = aggregate_shap_by_original_features(
                shap_vals, transformed_features, original_features
            )
            
            # Use aggregated values
            features = list(aggregated_shap.keys())
            shap_vals = list(aggregated_shap.values())
            
        else:
            shap_result = explainer.shap_values(df)
            if isinstance(shap_result, list):
                shap_vals = shap_result[0]
            else:
                shap_vals = shap_result[0] if shap_result.ndim > 1 else shap_result
            features = df.columns.tolist()
            
            # Convert to list if needed
            if isinstance(shap_vals, np.ndarray):
                if shap_vals.ndim > 1:
                    shap_vals = shap_vals.flatten()
                shap_vals = shap_vals.tolist()
    else:
        # For PermutationExplainer or other explainers
        shap_result = explainer.shap_values(df)
        if isinstance(shap_result, list):
            shap_vals = shap_result[0]
        else:
            shap_vals = shap_result[0] if shap_result.ndim > 1 else shap_result
        features = df.columns.tolist()
        
        # Convert to list if needed
        if isinstance(shap_vals, np.ndarray):
            if shap_vals.ndim > 1:
                shap_vals = shap_vals.flatten()
            shap_vals = shap_vals.tolist()
    
    # Final conversion to ensure it's a list
    if not isinstance(shap_vals, list):
        if isinstance(shap_vals, np.ndarray):
            shap_vals = shap_vals.tolist()
        elif hasattr(shap_vals, 'tolist'):
            shap_vals = shap_vals.tolist()
        else:
            shap_vals = list(shap_vals)

    # Ensure features and shap_vals have the same length
    if len(features) != len(shap_vals):
        # If lengths don't match, use indices
        features = [f"feature_{i}" for i in range(len(shap_vals))]

    # top 5 drivers
    abs_vals = [abs(v) for v in shap_vals]
    top_idx = sorted(range(len(abs_vals)), key=lambda i: abs_vals[i], reverse=True)#[:5]

    top_features = [
        {"feature": features[i], "shap_value": float(shap_vals[i])}
        for i in top_idx
    ]
    return shap_vals, top_features


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# @app.route("/predict", methods=["GET"])
# def predict_page():
#     return render_template("predict.html")


@app.route("/explain", methods=["GET"])
def explain_page():
    return render_template("explain.html")


@app.route("/counterfactual", methods=["GET"])
def counterfactual_page():
    return render_template("counterfactual.html")


@app.route("/grafana", methods=["GET"])
def grafana_page():
    return render_template("grafana.html")


@app.route("/bias", methods=["GET"])
def bias_page():
    return render_template("bias.html")


@app.route("/bias_reports_page", methods=["GET"])
def bias_reports_page():
    """Render the bias reports page."""
    return render_template("bias_reports.html")


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ------------------------------------------------------------
# MAIN PREDICTION ENDPOINT (with DB logging + SHAP logs)
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features", {})
    group_label = data.get("group_label", "A")

    if not features:
        return jsonify({"error": "Missing 'features'"}), 400

    features = normalize_features(features)

    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # ---------------- MODEL PREDICTION ----------------
    prediction = run_prediction(features)
    request_id = str(uuid.uuid4())

    # ---------------- SHAP EXPLANATION ----------------
    shap_vals, top_features = compute_shap(features)

    # ---------------- DB LOGGING ----------------
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO decision_logs (
            request_id, 
            model_version, 
            features, 
            prediction,
            shap_values, 
            top_features
        )
        VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
    """, (
        request_id,
        "v1",
        json.dumps(features),
        prediction,
        json.dumps(shap_vals),
        json.dumps(top_features)
    ))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "request_id": request_id,
        "prediction": prediction,
        "top_features": top_features
    }), 200


# ------------------------------------------------------------
# COUNTERFACTUAL FAIRNESS CHECK (Fixed attribute: sex)
# ------------------------------------------------------------
@app.route("/counterfactual_test", methods=["POST"])
def counterfactual_test():
    data = request.json
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features'"}), 400

    features = normalize_features(features)

    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # flip function
    def flip_sex(v):
        return "Female" if v.lower() == "male" else "Male"

    original_pred = run_prediction(features)

    cf_features = features.copy()
    cf_features["sex"] = flip_sex(features["sex"])
    cf_pred = run_prediction(cf_features)

    changed = (original_pred != cf_pred)
    request_id = str(uuid.uuid4())

    # DB logging
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO counterfactual_logs (
            request_id,
            original_sex,
            counterfactual_sex,
            original_prediction,
            counterfactual_prediction,
            changed
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        request_id,
        features["sex"],
        cf_features["sex"],
        original_pred,
        cf_pred,
        changed
    ))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "request_id": request_id,
        "attribute_tested": "sex",
        "original_prediction": original_pred,
        "counterfactual_prediction": cf_pred,
        "changed_due_to_attribute": changed
    }), 200


# ------------------------------------------------------------
# SHAP EXPLANATION ENDPOINT (with DB logging)
# ------------------------------------------------------------
@app.route("/explain", methods=["POST"])
def explain():
    data = request.json
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features'"}), 400

    features = normalize_features(features)

    # Check required fields
    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Get prediction (needed for DB logging)
    prediction = run_prediction(features)
    request_id = str(uuid.uuid4())

    # Compute SHAP explanation
    shap_vals, top_features = compute_shap(features)

    # ---------------- DB LOGGING ----------------
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO decision_logs (
            request_id, 
            model_version, 
            features, 
            prediction,
            shap_values, 
            top_features
        )
        VALUES (%s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb)
    """, (
        request_id,
        "v1",
        json.dumps(features),
        prediction,
        json.dumps(shap_vals),
        json.dumps(top_features)
    ))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "request_id": request_id,
        "prediction": prediction,
        "shap_values": shap_vals,
        "top_features": top_features
    }), 200


# ------------------------------------------------------------
# BIAS REPORTING ENDPOINTS
# ------------------------------------------------------------
@app.route("/bias_report", methods=["POST"])
def submit_bias_report():
    """Submit a new bias incident report."""
    data = request.json
    
    # Required fields
    reporter_name = data.get("reporter_name")
    reporter_role = data.get("reporter_role")
    reporter_team = data.get("reporter_team")
    description = data.get("description")
    
    if not all([reporter_name, reporter_role, reporter_team, description]):
        return jsonify({
            "error": "Missing required fields: reporter_name, reporter_role, reporter_team, description"
        }), 400
    
    # Optional fields
    related_request_id = data.get("related_request_id")
    suspected_feature = data.get("suspected_feature")
    priority = data.get("priority", "medium")
    additional_context = data.get("additional_context", {})
    
    # Validate priority
    if priority not in ["low", "medium", "high"]:
        priority = "medium"
    
    # Generate report ID
    report_id = str(uuid.uuid4())
    
    # Insert into database
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO bias_reports (
                report_id, reporter_name, reporter_role, reporter_team,
                description, related_request_id, suspected_feature,
                status, priority, additional_context
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """, (
            report_id, reporter_name, reporter_role, reporter_team,
            description, related_request_id, suspected_feature,
            "open", priority, json.dumps(additional_context)
        ))
        conn.commit()
        
        # Get timestamp
        cur.execute("SELECT timestamp FROM bias_reports WHERE report_id = %s", (report_id,))
        timestamp = cur.fetchone()[0]
        
        return jsonify({
            "report_id": report_id,
            "timestamp": timestamp.isoformat() if timestamp else None,
            "status": "open",
            "message": "Bias report submitted successfully"
        }), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/bias_reports", methods=["GET"])
def list_bias_reports():
    """List all bias reports with optional filters."""
    # Check if this is an AJAX/API request (has query params or wants JSON)
    wants_json = (
        request.args.get("limit") or 
        request.args.get("status") or 
        request.args.get("priority") or
        request.args.get("reporter_team") or
        "application/json" in request.headers.get("Accept", "")
    )
    
    # If no query params and wants HTML, return the template
    if not wants_json:
        return render_template("bias_reports.html")
    
    # Otherwise return JSON data
    # Get query parameters
    status_filter = request.args.get("status")
    priority_filter = request.args.get("priority")
    reporter_team_filter = request.args.get("reporter_team")
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    
    # Validate limit
    if limit > 1000:
        limit = 1000
    if limit < 1:
        limit = 1
    
    # Build query
    query = "SELECT report_id, timestamp, reporter_name, reporter_role, reporter_team, description, related_request_id, suspected_feature, status, priority, reviewer_notes, reviewed_by, reviewed_at, resolved_at, additional_context FROM bias_reports WHERE 1=1"
    params = []
    
    if status_filter:
        query += " AND status = %s"
        params.append(status_filter)
    
    if priority_filter:
        query += " AND priority = %s"
        params.append(priority_filter)
    
    if reporter_team_filter:
        query += " AND reporter_team = %s"
        params.append(reporter_team_filter)
    
    query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    # Count total
    count_query = "SELECT COUNT(*) FROM bias_reports WHERE 1=1"
    count_params = []
    if status_filter:
        count_query += " AND status = %s"
        count_params.append(status_filter)
    if priority_filter:
        count_query += " AND priority = %s"
        count_params.append(priority_filter)
    if reporter_team_filter:
        count_query += " AND reporter_team = %s"
        count_params.append(reporter_team_filter)
    
    conn = get_conn()
    cur = conn.cursor()
    try:
        # Get total count
        cur.execute(count_query, count_params)
        total = cur.fetchone()[0]
        
        # Get reports
        cur.execute(query, params)
        rows = cur.fetchall()
        
        reports = []
        for row in rows:
            reports.append({
                "report_id": row[0],
                "timestamp": row[1].isoformat() if row[1] else None,
                "reporter_name": row[2],
                "reporter_role": row[3],
                "reporter_team": row[4],
                "description": row[5],
                "related_request_id": row[6],
                "suspected_feature": row[7],
                "status": row[8],
                "priority": row[9],
                "reviewer_notes": row[10],
                "reviewed_by": row[11],
                "reviewed_at": row[12].isoformat() if row[12] else None,
                "resolved_at": row[13].isoformat() if row[13] else None,
                "additional_context": row[14] if row[14] else {}
            })
        
        return jsonify({
            "reports": reports,
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/bias_report/<report_id>", methods=["GET"])
def get_bias_report(report_id):
    """Get full details of a specific bias report."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT report_id, timestamp, reporter_name, reporter_role, reporter_team,
                   description, related_request_id, suspected_feature, status, priority,
                   reviewer_notes, reviewed_by, reviewed_at, resolved_at, additional_context
            FROM bias_reports
            WHERE report_id = %s
        """, (report_id,))
        
        row = cur.fetchone()
        if not row:
            return jsonify({"error": "Report not found"}), 404
        
        return jsonify({
            "report_id": row[0],
            "timestamp": row[1].isoformat() if row[1] else None,
            "reporter_name": row[2],
            "reporter_role": row[3],
            "reporter_team": row[4],
            "description": row[5],
            "related_request_id": row[6],
            "suspected_feature": row[7],
            "status": row[8],
            "priority": row[9],
            "reviewer_notes": row[10],
            "reviewed_by": row[11],
            "reviewed_at": row[12].isoformat() if row[12] else None,
            "resolved_at": row[13].isoformat() if row[13] else None,
            "additional_context": row[14] if row[14] else {}
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/bias_report/<report_id>", methods=["PATCH"])
def update_bias_report(report_id):
    """Update a bias report status, add reviewer notes, or change priority."""
    data = request.json
    
    # Check if report exists
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT status FROM bias_reports WHERE report_id = %s", (report_id,))
        existing = cur.fetchone()
        if not existing:
            return jsonify({"error": "Report not found"}), 404
        
        # Build update query dynamically
        updates = []
        params = []
        
        if "status" in data:
            new_status = data["status"]
            if new_status not in ["open", "reviewed", "in_progress", "resolved", "closed"]:
                return jsonify({"error": "Invalid status. Must be: open, reviewed, in_progress, resolved, or closed"}), 400
            updates.append("status = %s")
            params.append(new_status)
            
            # Auto-set resolved_at if status is resolved or closed
            if new_status in ["resolved", "closed"]:
                updates.append("resolved_at = NOW()")
            elif existing[0] in ["resolved", "closed"] and new_status not in ["resolved", "closed"]:
                updates.append("resolved_at = NULL")
        
        if "priority" in data:
            new_priority = data["priority"]
            if new_priority not in ["low", "medium", "high"]:
                return jsonify({"error": "Invalid priority. Must be: low, medium, or high"}), 400
            updates.append("priority = %s")
            params.append(new_priority)
        
        if "reviewer_notes" in data:
            updates.append("reviewer_notes = %s")
            params.append(data["reviewer_notes"])
        
        if "reviewed_by" in data:
            updates.append("reviewed_by = %s")
            params.append(data["reviewed_by"])
            # Set reviewed_at if not already set
            updates.append("reviewed_at = COALESCE(reviewed_at, NOW())")
        
        if "additional_context" in data:
            updates.append("additional_context = %s::jsonb")
            params.append(json.dumps(data["additional_context"]))
        
        if not updates:
            return jsonify({"error": "No fields to update"}), 400
        
        # Add report_id to params
        params.append(report_id)
        
        # Execute update
        update_query = f"UPDATE bias_reports SET {', '.join(updates)} WHERE report_id = %s"
        cur.execute(update_query, params)
        conn.commit()
        
        return jsonify({
            "report_id": report_id,
            "status": data.get("status", existing[0]),
            "message": "Report updated successfully"
        }), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


# ------------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
