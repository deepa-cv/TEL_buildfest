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
    top_features JSONB,
    fairness_passed BOOLEAN,
    ui_outcome VARCHAR(32),
    analysis_meta JSONB,
    actual_outcome INT
);

CREATE INDEX IF NOT EXISTS idx_decision_logs_ts ON decision_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decision_logs_mv ON decision_logs (model_version);

CREATE TABLE IF NOT EXISTS fairness_alerts (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    severity VARCHAR(16),
    metric_name VARCHAR(64),
    message TEXT,
    details JSONB
);

CREATE TABLE IF NOT EXISTS fairness_snapshots (
    id SERIAL PRIMARY KEY,
    calc_time TIMESTAMPTZ DEFAULT NOW(),
    model_version VARCHAR(64),
    window_hours DOUBLE PRECISION,
    row_limit INT,
    split_attribute VARCHAR(64),
    metrics_json JSONB
);

CREATE TABLE IF NOT EXISTS bias_reports (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(64) UNIQUE NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    reporter_name VARCHAR(255) NOT NULL,
    reporter_role VARCHAR(255) NOT NULL,
    reporter_team VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    related_request_id VARCHAR(64),
    suspected_feature VARCHAR(255),
    status VARCHAR(32) DEFAULT 'open',
    priority VARCHAR(32) DEFAULT 'medium',
    reviewer_notes TEXT,
    reviewed_by VARCHAR(255),
    reviewed_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    additional_context JSONB
);

CREATE INDEX IF NOT EXISTS idx_bias_reports_report_id ON bias_reports(report_id);
CREATE INDEX IF NOT EXISTS idx_bias_reports_status ON bias_reports(status);
CREATE INDEX IF NOT EXISTS idx_bias_reports_priority ON bias_reports(priority);
