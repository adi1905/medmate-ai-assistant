-- MedMate Database Schema
-- SQLite. Foreign keys must be enabled per-connection: PRAGMA foreign_keys = ON;

PRAGMA foreign_keys = ON;

-- ============================================================
-- PATIENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS patients (
    patient_id      TEXT PRIMARY KEY,          -- e.g. 'PT-00001'
    first_name      TEXT NOT NULL,
    last_name       TEXT NOT NULL,
    dob             TEXT NOT NULL,              -- ISO date YYYY-MM-DD
    sex             TEXT NOT NULL CHECK (sex IN ('M', 'F')),
    phone           TEXT NOT NULL,
    email           TEXT,
    address         TEXT,
    insurance_provider TEXT,
    created_at      TEXT NOT NULL,              -- ISO datetime
    updated_at      TEXT NOT NULL
);

-- ============================================================
-- PROVIDERS (clinic staff who see patients)
-- ============================================================
CREATE TABLE IF NOT EXISTS providers (
    provider_id     TEXT PRIMARY KEY,           -- e.g. 'PR-001'
    name            TEXT NOT NULL,
    specialty       TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- ============================================================
-- APPOINTMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id  TEXT PRIMARY KEY,           -- e.g. 'AP-000001'
    patient_id      TEXT NOT NULL,
    provider_id     TEXT NOT NULL,
    scheduled_time  TEXT NOT NULL,              -- ISO datetime
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('scheduled', 'completed', 'cancelled', 'no_show')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id)
);

CREATE INDEX IF NOT EXISTS idx_appointments_patient ON appointments(patient_id);
CREATE INDEX IF NOT EXISTS idx_appointments_status ON appointments(status);
CREATE INDEX IF NOT EXISTS idx_appointments_time ON appointments(scheduled_time);

-- ============================================================
-- MEDICATIONS REFERENCE (curated static data — drives interaction checks)
-- ============================================================
CREATE TABLE IF NOT EXISTS medications_reference (
    medication_name TEXT PRIMARY KEY,           -- lowercase, canonical
    drug_class      TEXT NOT NULL,
    typical_dosage  TEXT,
    notes           TEXT
);

-- ============================================================
-- PRESCRIPTIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS prescriptions (
    prescription_id TEXT PRIMARY KEY,           -- e.g. 'RX-000001'
    patient_id      TEXT NOT NULL,
    provider_id     TEXT NOT NULL,
    medication_name TEXT NOT NULL,              -- FK to medications_reference
    dosage          TEXT NOT NULL,               -- e.g. '10mg once daily'
    start_date      TEXT NOT NULL,
    end_date        TEXT,                        -- NULL = ongoing
    status          TEXT NOT NULL CHECK (status IN ('active', 'completed', 'discontinued')),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id),
    FOREIGN KEY (medication_name) REFERENCES medications_reference(medication_name)
);

CREATE INDEX IF NOT EXISTS idx_prescriptions_patient ON prescriptions(patient_id);
CREATE INDEX IF NOT EXISTS idx_prescriptions_status ON prescriptions(status);

-- ============================================================
-- LAB RESULTS
-- ============================================================
CREATE TABLE IF NOT EXISTS lab_results (
    lab_result_id   TEXT PRIMARY KEY,           -- e.g. 'LB-000001'
    patient_id      TEXT NOT NULL,
    test_name       TEXT NOT NULL,
    value           REAL NOT NULL,
    unit            TEXT NOT NULL,
    reference_low   REAL NOT NULL,
    reference_high  REAL NOT NULL,
    flagged         INTEGER NOT NULL DEFAULT 0 CHECK (flagged IN (0, 1)),
    recorded_at     TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_lab_results_patient ON lab_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_lab_results_flagged ON lab_results(flagged);

-- ============================================================
-- LONG-TERM MEMORY (agent-facing durable facts, cross-session)
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_memory (
    memory_id       TEXT PRIMARY KEY,           -- e.g. 'MEM-000001'
    patient_id      TEXT NOT NULL,
    memory_type     TEXT NOT NULL CHECK (memory_type IN ('preference', 'recurring_topic', 'flag')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_patient ON agent_memory(patient_id);


CREATE TABLE IF NOT EXISTS users (
    user_id         TEXT PRIMARY KEY,           -- e.g. 'USR-000001'
    email           TEXT NOT NULL UNIQUE,
    hashed_password TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('patient', 'staff', 'admin')),
    patient_id      TEXT,                        -- required iff role='patient'
    provider_id     TEXT,                        -- optional, staff/admin only
    display_name    TEXT NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id),
    CHECK (
        (role = 'patient' AND patient_id IS NOT NULL)
        OR (role IN ('staff', 'admin'))
    )
);
 
-- One login per patient. Partial index (SQLite supports WHERE on unique
-- indexes) so it doesn't collide with the many NULL patient_ids from
-- staff/admin rows.
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_patient_id
    ON users(patient_id) WHERE patient_id IS NOT NULL;
 
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
 
-- ============================================================
-- REFRESH TOKENS (enables real logout / revocation, matches the
-- deny-list note left as a TODO in routers/auth.py::logout)
-- ============================================================
CREATE TABLE IF NOT EXISTS refresh_tokens (
    jti             TEXT PRIMARY KEY,            -- unique id embedded in the refresh JWT
    user_id         TEXT NOT NULL,
    issued_at       TEXT NOT NULL,
    expires_at      TEXT NOT NULL,
    revoked         INTEGER NOT NULL DEFAULT 0 CHECK (revoked IN (0, 1)),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
 
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);
 
-- ============================================================
-- CHAT THREADS (thread ownership -- needed by routers/chat.py's
-- list_threads / _assert_owns_thread, which currently have TODOs)
-- ============================================================
-- Message content itself doesn't need to be duplicated here -- your
-- LangGraph checkpointer already persists it per thread_id. This table
-- exists only so the API can answer "which threads belong to this user"
-- and "does this user own this thread_id" without deserializing
-- checkpoint state just to check ownership.
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id       TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL,
    title           TEXT,
    created_at      TEXT NOT NULL,
    last_message_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
 
CREATE INDEX IF NOT EXISTS idx_chat_threads_user ON chat_threads(user_id);


CREATE TABLE IF NOT EXISTS chat_messages (
    message_id      TEXT PRIMARY KEY,
    thread_id       TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES chat_threads(thread_id)
);
 
CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id);
 
-- ============================================================
-- REQUEST/TURN LOGS -- observability only. PII-REDACTED before insert.
-- This is what backs /admin/metrics and /admin/traces: latency, tool
-- call counts, HITL confirm/abandon outcomes, per-tool errors. Devs and
-- admins reading this table should never see raw clinical content.
-- ============================================================
CREATE TABLE IF NOT EXISTS request_logs (
    log_id           TEXT PRIMARY KEY,
    thread_id        TEXT NOT NULL,
    user_id          TEXT NOT NULL,
    role             TEXT NOT NULL,
    event_type       TEXT NOT NULL CHECK (event_type IN
                          ('message', 'hitl_prompt', 'hitl_resolved', 'error')),
    trace_id         TEXT,               -- correlates to a LangSmith run, if you wire that in
    node_path        TEXT,               -- JSON list, e.g. '["router","patient_records","synthesizer"]'
    tool_call_count  INTEGER NOT NULL DEFAULT 0,
    latency_ms       REAL,
    hitl_outcome     TEXT CHECK (hitl_outcome IN ('approved', 'rejected', 'edited') OR hitl_outcome IS NULL),
    redacted_summary TEXT,                -- PII-scrubbed one-liner via redact_pii(), NOT raw content
    created_at       TEXT NOT NULL,
    FOREIGN KEY (thread_id) REFERENCES chat_threads(thread_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
 
CREATE INDEX IF NOT EXISTS idx_request_logs_thread ON request_logs(thread_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_event ON request_logs(event_type);
 
-- last_message_at on chat_threads (from 002) gets touched on every turn --
-- no schema change needed there, just an UPDATE in the route.

CREATE TABLE IF NOT EXISTS pending_actions (
    action_id   TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    tool_name   TEXT NOT NULL,
    tool_args   TEXT NOT NULL,   -- JSON-encoded dict
    summary     TEXT,
    status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'expired')),
    created_at  TEXT NOT NULL,
    resolved_at TEXT,
    FOREIGN KEY (thread_id) REFERENCES chat_threads(thread_id)
);
 
CREATE INDEX IF NOT EXISTS idx_pending_actions_thread ON pending_actions(thread_id);
CREATE INDEX IF NOT EXISTS idx_pending_actions_status ON pending_actions(status);
