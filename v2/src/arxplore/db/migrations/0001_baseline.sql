CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    authors JSONB NOT NULL DEFAULT '[]'::jsonb,
    abstract TEXT NOT NULL DEFAULT '',
    primary_category TEXT,
    categories JSONB NOT NULL DEFAULT '[]'::jsonb,
    pdf_url TEXT,
    published_at TIMESTAMPTZ NULL,
    updated_at TIMESTAMPTZ NULL,
    upvotes INTEGER NOT NULL DEFAULT 0,
    github_url TEXT,
    github_stars INTEGER NULL,
    citation_count INTEGER NULL,
    source TEXT NOT NULL DEFAULT 'hf_daily_papers',
    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_fulltexts (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
    source TEXT NOT NULL DEFAULT 'pdf',
    quality_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    artifacts JSONB NOT NULL DEFAULT '{}'::jsonb,
    parser_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE paper_fulltexts
ADD COLUMN IF NOT EXISTS quality_metrics JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE paper_fulltexts
ADD COLUMN IF NOT EXISTS artifacts JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE paper_fulltexts
ADD COLUMN IF NOT EXISTS parser_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE paper_fulltexts ADD COLUMN IF NOT EXISTS content_hash TEXT NULL;

CREATE TABLE IF NOT EXISTS paper_chunks (
    id BIGSERIAL PRIMARY KEY,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    section_title TEXT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(arxiv_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS paper_embeddings (
    chunk_id BIGINT PRIMARY KEY REFERENCES paper_chunks(id) ON DELETE CASCADE,
    embedding VECTOR(1536) NOT NULL,
    model_name TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS topics (
    topic_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS topic_papers (
    topic_id INTEGER NOT NULL,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    PRIMARY KEY (topic_id, arxiv_id)
);

CREATE TABLE IF NOT EXISTS topic_documents (
    topic_id INTEGER PRIMARY KEY,
    document JSONB NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE papers
    ADD COLUMN IF NOT EXISTS title_abstract_vector tsvector
    GENERATED ALWAYS AS (setweight(to_tsvector('english', coalesce(title, '')), 'A') || setweight(to_tsvector('english', coalesce(abstract, '')), 'B')) STORED;

ALTER TABLE paper_chunks
    ADD COLUMN IF NOT EXISTS chunk_vector tsvector
    GENERATED ALWAYS AS (setweight(to_tsvector('english', coalesce(chunk_text, '')), 'C')) STORED;

CREATE INDEX IF NOT EXISTS idx_papers_title_abstract_vector ON papers USING GIN (title_abstract_vector);

CREATE INDEX IF NOT EXISTS idx_paper_chunks_chunk_vector ON paper_chunks USING GIN (chunk_vector);

DROP INDEX IF EXISTS idx_paper_chunks_fts;

DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS paper_embeddings_embedding_hnsw ON paper_embeddings USING hnsw (embedding vector_cosine_ops);
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'paper_embeddings HNSW 인덱스를 만들지 못했습니다(pgvector 0.5.0 이상 필요). 벡터 검색은 순차 스캔으로 동작합니다: %', SQLERRM;
END
$$;

CREATE TABLE IF NOT EXISTS paper_ai_overviews (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    overview TEXT,
    key_findings JSONB NOT NULL DEFAULT '[]'::jsonb,
    model TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_ai_detailed_summaries (
    id BIGSERIAL PRIMARY KEY,
    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    summary TEXT NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (arxiv_id, model)
);

CREATE TABLE IF NOT EXISTS raw_daily_papers (
    source TEXT NOT NULL,
    date DATE NOT NULL,
    payload JSONB NOT NULL,
    payload_hash TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    fetched_count INTEGER,
    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, date)
);

CREATE TABLE IF NOT EXISTS pipeline_state (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS prepare_jobs (
    id BIGSERIAL PRIMARY KEY,
    mode TEXT NOT NULL,
    target_date TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'collect',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    worker_id TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    claimed_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_prepare_jobs_mode_target_date UNIQUE (mode, target_date)
);

ALTER TABLE prepare_jobs
    ADD COLUMN IF NOT EXISTS claim_generation INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS raw_revision INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS pending_refresh BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_prepare_jobs_mode_status_target_date
ON prepare_jobs (mode, status, target_date);

CREATE INDEX IF NOT EXISTS idx_prepare_jobs_status_updated_at
ON prepare_jobs (status, updated_at DESC);
