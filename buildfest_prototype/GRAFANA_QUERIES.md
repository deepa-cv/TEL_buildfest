## Interpretability:

WITH feature_data AS (
  SELECT
    timestamp AS time,
    jsonb_array_elements(top_features)->>'feature' AS feature,
    (jsonb_array_elements(top_features)->>'shap_value')::float AS shap_value
  FROM decision_logs
  WHERE $__timeFilter(timestamp)
)
SELECT
  time,
  feature,
  shap_value
FROM feature_data
ORDER BY shap_value desc, feature;



## counterfactual tests:
WITH feature_data AS (
  SELECT
    timestamp AS time,
    jsonb_array_elements(top_features)->>'feature' AS feature,
    (jsonb_array_elements(top_features)->>'shap_value')::float AS shap_value
  FROM decision_logs
  WHERE $__timeFilter(timestamp)
)
SELECT
  time,
  feature,
  shap_value
FROM feature_data
ORDER BY shap_value desc, feature;

## Interpretability

WITH shap_expanded AS (
  SELECT
    timestamp,
    jsonb_array_elements(top_features)->>'feature' AS feature,
    ABS((jsonb_array_elements(top_features)->>'shap_value')::float) AS abs_shap_value,
    (jsonb_array_elements(top_features)->>'shap_value')::float AS shap_value
  FROM decision_logs
  WHERE top_features IS NOT NULL
    AND $__timeFilter(timestamp)
),
shap_statistics_per_prediction AS (
  SELECT
    timestamp AS time,
    AVG(abs_shap_value) AS mean_abs_shap,
    STDDEV(abs_shap_value) AS stddev_abs_shap,
    MAX(abs_shap_value) AS max_abs_shap
  FROM shap_expanded
  GROUP BY timestamp
),
max_feature_per_prediction AS (
  SELECT DISTINCT ON (e.timestamp)
    e.timestamp AS time,
    e.feature,
    e.abs_shap_value AS max_shap_value,
    e.shap_value AS original_shap_value,
    s.mean_abs_shap,
    COALESCE(s.stddev_abs_shap, 0) AS stddev_abs_shap,
    (s.mean_abs_shap + 2 * COALESCE(s.stddev_abs_shap, 0)) AS threshold_2std
  FROM shap_expanded e
  JOIN shap_statistics_per_prediction s ON e.timestamp = s.time
  WHERE e.abs_shap_value = s.max_abs_shap
  ORDER BY e.timestamp, e.abs_shap_value DESC
)
SELECT
  time,
  feature AS exceeding_feature,
  max_shap_value,
  original_shap_value,
  mean_abs_shap,
  stddev_abs_shap,
  threshold_2std,
  CASE 
    WHEN max_shap_value > threshold_2std THEN 1 
    ELSE 0 
  END AS alert_triggered
FROM max_feature_per_prediction
ORDER BY time DESC;