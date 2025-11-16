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
                    <p>Health check endpoint. Returns API status.</p>
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <code>/predict</code>
                    <p>Prediction endpoint. Accepts JSON with <code>features</code> and <code>group_label</code>.</p>
                    <p><strong>Example:</strong></p>
                    <pre>{
  "features": {"income": 60000},
  "group_label": "A"
}</pre>
                </div>
            </div>
        </body>
    </html>
    """, 200

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

