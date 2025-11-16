# Flask UI for Fairness Monitoring Dashboard

A simple web-based UI for interacting with the Fairness Monitoring API and viewing Grafana dashboards.

## Features

1. **Predict Page** (`/predict`)
   - Form to input all required features
   - Submit to get model predictions
   - View prediction results and top contributing features

2. **SHAP Explanation Page** (`/explain`)
   - Get detailed feature importance explanations
   - Visualize SHAP values in a table with color-coded bars
   - See which features contribute most to the prediction

3. **Counterfactual Test Page** (`/counterfactual`)
   - Test fairness by flipping the "sex" attribute
   - Compare original vs counterfactual predictions
   - See if the model's prediction changes based on gender

4. **Grafana Dashboards** (`/grafana`)
   - Embedded Grafana interface
   - View analytics and visualizations
   - Access to all Grafana features

## Usage

1. **Start the application:**
   ```bash
   docker-compose up
   ```

2. **Access the UI:**
   - Open your browser and navigate to: `http://localhost:5050`
   - You'll see the home page with links to all features

3. **Using the Forms:**
   - Click "Fill Sample Data" to quickly populate the form with example values
   - Fill in all required fields
   - Click the submit button to get results

4. **Viewing Grafana:**
   - Click "Grafana Dashboards" in the navigation
   - The Grafana interface will be embedded in the page
   - **Default Grafana credentials:** `admin` / `admin`
   - If you can't login, the password has been reset to `admin`

## API Endpoints

The UI uses the following API endpoints:

- `POST /predict` - Make a prediction
- `POST /explain` - Get SHAP explanations
- `POST /counterfactual_test` - Run counterfactual fairness test
- `GET /health` - Health check

## File Structure

```
app/
├── templates/
│   ├── base.html          # Base template with navigation
│   ├── index.html         # Home page
│   ├── predict.html      # Prediction form
│   ├── explain.html      # SHAP explanation form
│   ├── counterfactual.html  # Counterfactual test form
│   └── grafana.html       # Grafana dashboard page
├── static/
│   ├── css/
│   │   └── style.css     # Main stylesheet
│   └── js/
│       └── main.js        # Common JavaScript utilities
└── app.py                 # Flask application with routes
```

## Styling

The UI uses a clean, modern design with:
- Responsive grid layouts
- Color-coded SHAP values (green for positive, red for negative)
- Visual indicators for fairness test results
- Mobile-friendly responsive design

## Notes

- All forms include a "Fill Sample Data" button for quick testing
- Results are displayed below the form after submission
- Error messages are shown in red if API calls fail
- The Grafana iframe requires Grafana to be running on `localhost:3000`

