ALTER TABLE paper_ai_overviews
    ADD COLUMN IF NOT EXISTS prompt_version INTEGER NOT NULL DEFAULT 1
        CONSTRAINT ck_paper_ai_overviews_prompt_version CHECK (prompt_version > 0);

ALTER TABLE paper_ai_detailed_summaries
    ADD COLUMN IF NOT EXISTS prompt_version INTEGER NOT NULL DEFAULT 1
        CONSTRAINT ck_paper_ai_detailed_summaries_prompt_version CHECK (prompt_version > 0);
