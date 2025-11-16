from flask import Flask, request, jsonify
import psycopg2
import os
import uuid
import pandas as pd
import joblib
import json


app = Flask(__name__)

# ------------------------------------------------------------
# LOAD MODEL ONCE (GLOBAL)
# ------------------------------------------------------------
MODEL_PATH = "/app/model.pkl"   # Ensure model.pkl is inside /app folder
model = joblib.load(MODEL_PATH)
print("✔ Loaded ML model from", MODEL_PATH)


# ------------------------------------------------------------
# REQUIRED FIELDS FOR THE MODEL
# ------------------------------------------------------------
REQUIRED_FIELDS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country"
]


# ------------------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------------------
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        dbname=os.getenv("DB_NAME", "fairness"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres")
    )


# ------------------------------------------------------------
# Shared prediction function
# ------------------------------------------------------------
def run_prediction(feat_dict):
    df = pd.DataFrame([feat_dict])
    return int(model.predict(df)[0])


# ------------------------------------------------------------
# NORMALIZE INPUT FEATURE(S)
# ------------------------------------------------------------
def normalize_features(feat_dict):
    # normalize sex
    if "sex" in feat_dict and isinstance(feat_dict["sex"], str):
        feat_dict["sex"] = feat_dict["sex"].strip().title()

    # normalize race (just to avoid mismatches)
    if "race" in feat_dict and isinstance(feat_dict["race"], str):
        feat_dict["race"] = feat_dict["race"].strip().title()

    return feat_dict


# ------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return """
    <html>
        <head>
            <title>Fairness API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                .endpoint { margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #4CAF50; }
                .method { display: inline-block; padding: 4px 8px; border-radius: 4px; font-weight: bold; margin-right: 10px; }
                .get { background-color: #4CAF50; color: white; }
                .post { background-color: #2196F3; color: white; }
                code { background-color: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Fairness API</h1>
                <p>Welcome to the Fairness API. Available endpoints:</p>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <code>/health</code>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/predict</code>
                </div>

                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/counterfactual_test</code>
                </div>
            </div>
        </body>
    </html>
    """, 200


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# ------------------------------------------------------------
# PREDICT ENDPOINT (USING model.pkl)
# ------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features", {})
    group_label = data.get("group_label", "A")

    if not features:
        return jsonify({"error": "Missing 'features' in request"}), 400

    # Normalize
    features = normalize_features(features)

    # Required-field check
    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Model prediction
    prediction = run_prediction(features)

    request_id = str(uuid.uuid4())

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO decisions (request_id, model_version, group_label, prediction)
        VALUES (%s, %s, %s, %s)
        """,
        (request_id, "v1", group_label, prediction)
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "request_id": request_id,
        "prediction": prediction
    }), 200



# ------------------------------------------------------------
# COUNTERFACTUAL FAIRNESS TEST (FIXED ATTRIBUTE: sex)
# ------------------------------------------------------------
@app.route("/counterfactual_test", methods=["POST"])
def counterfactual_test():
    data = request.json
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features'"}), 400

    # Normalize
    features = normalize_features(features)

    # Required-field check
    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Flip the sex attribute
    def flip_sex(value):
        v = value.strip().title()
        return "Female" if v == "Male" else "Male"

    if "sex" not in features:
        return jsonify({"error": "Feature 'sex' missing"}), 400

    # Original prediction
    original_pred = run_prediction(features)

    # Counterfactual input
    cf_features = features.copy()
    cf_features["sex"] = flip_sex(cf_features["sex"])

    # Counterfactual prediction
    cf_pred = run_prediction(cf_features)

    changed = (original_pred != cf_pred)

        # --- Store in DB ---
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
        str(uuid.uuid4()),
        features["sex"],
        cf_features["sex"],
        original_pred,
        cf_pred,
        changed
    ))

    conn.commit()
    cur.close()
    conn.close()

    # --- API response ---
    return jsonify({
        "attribute_tested": "sex",
        "original_value": features["sex"],
        "counterfactual_value": cf_features["sex"],
        "original_prediction": original_pred,
        "counterfactual_prediction": cf_pred,
        "changed_due_to_attribute": changed,
        "explanation": 
            "⚠️ Model decision changed when sex was flipped — possible bias."
            if changed else
            "✔ Model decision is stable against the sex flip — no counterfactual bias detected."
    }), 200




# ------------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
