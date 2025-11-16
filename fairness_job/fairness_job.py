from flask import Flask, request, jsonify
import psycopg2
import os
import uuid

app = Flask(__name__)

# Database connection
def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "db"),
        dbname=os.getenv("DB_NAME", "fairness"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "postgres")
    )

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = data.get("features", {})
    group_label = data.get("group_label", "A")

    # simple dummy model: approve if income > 50000
    prediction = 1 if features.get("income", 0) > 50000 else 0

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
