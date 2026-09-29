DO $$
DECLARE
    invalid_target_date TEXT;
BEGIN
    SELECT target_date INTO invalid_target_date
    FROM prepare_jobs
    WHERE target_date !~ '^\d{4}-\d{2}-\d{2}$'
    LIMIT 1;
    IF FOUND THEN
        RAISE EXCEPTION 'prepare_jobs.target_date is not an ISO date (YYYY-MM-DD): %', invalid_target_date;
    END IF;
END
$$;

ALTER TABLE prepare_jobs
    ALTER COLUMN target_date TYPE DATE USING target_date::date;

ALTER TABLE prepare_jobs
    ADD CONSTRAINT ck_prepare_jobs_status CHECK (status IN ('pending', 'processing', 'done', 'failed'));
