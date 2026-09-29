ALTER TABLE papers
    ADD COLUMN arxiv_lookup_at TIMESTAMPTZ,
    ADD COLUMN arxiv_lookup_misses INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN arxiv_lookup_retry_at TIMESTAMPTZ;
