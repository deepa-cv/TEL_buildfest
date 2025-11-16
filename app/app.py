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
MODEL_PATH = "/app/model.pkl"
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
# RUN SERVER
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
