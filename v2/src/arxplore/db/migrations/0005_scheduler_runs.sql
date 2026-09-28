CREATE TABLE scheduler_runs (
    id BIGSERIAL PRIMARY KEY,
    job TEXT NOT NULL,
    trigger TEXT NOT NULL,
    scheduled_for TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running',
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 1,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    CONSTRAINT ck_scheduler_runs_trigger CHECK (trigger IN ('scheduled', 'manual')),
    CONSTRAINT ck_scheduler_runs_status CHECK (status IN ('running', 'success', 'failed', 'abandoned')),
    CONSTRAINT ck_scheduler_runs_scheduled_for CHECK (trigger = 'manual' OR scheduled_for IS NOT NULL)
);

CREATE UNIQUE INDEX uq_scheduler_runs_job_scheduled_for
ON scheduler_runs (job, scheduled_for)
WHERE trigger = 'scheduled';

CREATE INDEX idx_scheduler_runs_job_started_at
ON scheduler_runs (job, started_at DESC);
