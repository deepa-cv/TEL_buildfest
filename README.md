# Fairness Monitoring Dashboard

A ML fairness monitoring system with SHAP explanations, counterfactual testing, and real-time analytics visualization.

## Features

- 🔍 **SHAP Explanations**: Feature importance analysis
- ⚖️ **Counterfactual Testing**: Fairness evaluation by flipping protected attributes
- 📊 **Grafana Dashboards**: Visualize predictions and SHAP values
- 🚨 **Alerting**: Alerts for extreme SHAP values (2σ threshold)
- 💾 **Database Logging**: All predictions logged to PostgreSQL

## Quick Start

### Prerequisites
- Docker and Docker Compose
- ML model file (`model.pkl`) in `app/` directory

### Setup

1. **Add your model**:
   ```bash
   cp /path/to/your/model.pkl app/model.pkl
   ```

2. **Start services**:
   ```bash
   docker-compose up --build
   ```

3. **Access**:
   - Web UI: http://localhost:5050
   - Grafana: http://localhost:3000 (admin/admin)
   - API: http://localhost:5050

## Architecture

- **Flask API** (`flask_api`): Main server with ML model and SHAP explainer
- **PostgreSQL** (`db`): Stores predictions, SHAP values, and metrics
- **Fairness Job** (`fairness_job`): Background fairness metrics computation
- **Grafana** (`grafana`): Visualization and alerting

## API Endpoints

### SHAP Explanation
```bash
POST /explain
Content-Type: application/json

{
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
}
```

### Counterfactual Test
```bash
POST /counterfactual_test
Content-Type: application/json

{
  "features": { ... }
}
```

### Health Check
```bash
GET /health
```

## Database Schema

### `decision_logs`
- `request_id`, `timestamp`, `model_version`
- `features` (JSONB), `prediction`
- `shap_values` (JSONB), `top_features` (JSONB)

### `counterfactual_logs`
- `request_id`, `original_sex`, `counterfactual_sex`
- `original_prediction`, `counterfactual_prediction`, `changed`

See `db/init.sql` for complete schema.

## Project Structure

```
TEL_buildfest/
├── app/              # Flask API
│   ├── app.py        # Main application
│   ├── model.pkl     # Your ML model
│   ├── templates/    # HTML templates
│   └── static/       # CSS/JS assets
├── db/
│   └── init.sql      # Database schema
├── fairness_job/     # Fairness metrics service
├── docker-compose.yml
└── README.md
```

## Configuration

- **Database**: PostgreSQL on port 5432, database `fairness`, user/pass: `postgres`
- **Flask API**: Port 5050
- **Grafana**: Port 3000, credentials `admin/admin`

### Model Requirements

Your `model.pkl` should:
- Be saved with `joblib`
- Accept pandas DataFrame with Adult dataset features
- Support SHAP TreeExplainer (tree models) or PermutationExplainer (fallback)

## Grafana Setup

1. Log in at http://localhost:3000
2. Add PostgreSQL data source: `db:5432`, database `fairness`
3. Use queries from `GRAFANA_QUERIES.md` for dashboards
4. Configure alerts for extreme SHAP values

## Testing

```bash
# Test API
./test_api.sh
# or
python test_api.py
```

## Documentation

- `UI_README.md`: Web interface guide
- `GRAFANA_QUERIES.md`: SQL queries and alert setup
- `API_TESTING.md`: API testing examples

## Troubleshooting

```bash
# View logs
docker-compose logs flask_api
docker-compose logs db

# Restart services
docker-compose restart flask_api

# Rebuild
docker-compose up --build
```

## Key Features

- **SHAP**: Handles Pipeline models, maps features, aggregates one-hot encoded values
- **Counterfactual**: Flips "sex" attribute to test for bias
- **Alerting**: Dynamic threshold (mean + 2×std) per prediction

---

For detailed information, see the documentation files in the repository.
