CREATE TABLE IF NOT EXISTS decisions (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(64),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(32),
    group_label VARCHAR(32),
    prediction INTEGER
);

CREATE TABLE IF NOT EXISTS fairness_metrics (
    id SERIAL PRIMARY KEY,
    calc_time TIMESTAMPTZ DEFAULT NOW(),
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    group_a_selection_rate DOUBLE PRECISION,
    group_b_selection_rate DOUBLE PRECISION,
    disparate_impact DOUBLE PRECISION,
    status VARCHAR(16)
);

CREATE TABLE IF NOT EXISTS system_status (
    id BOOLEAN PRIMARY KEY DEFAULT TRUE,
    status VARCHAR(16),
    safe_mode BOOLEAN
);

INSERT INTO system_status (id,status,safe_mode)
VALUES (TRUE,'FAIR',FALSE)
ON CONFLICT (id) DO NOTHING;


CREATE TABLE IF NOT EXISTS counterfactual_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    request_id VARCHAR(64),

    -- original input
    original_sex VARCHAR(20),
    counterfactual_sex VARCHAR(20),

    -- predictions
    original_prediction INT,
    counterfactual_prediction INT,

    -- changed?
    changed BOOLEAN
);


CREATE TABLE IF NOT EXISTS decision_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    request_id VARCHAR(64),
    model_version VARCHAR(20),
    features JSONB,
    prediction INT,
    shap_values JSONB,
    top_features JSONB
);
