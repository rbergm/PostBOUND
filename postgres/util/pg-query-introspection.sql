CREATE OR REPLACE FUNCTION running_queries()
RETURNS TABLE (datname NAME, pid INTEGER, state TEXT, query TEXT, age INTERVAL)
AS $$
    SELECT datname, pid, state, query, age(clock_timestamp(), query_start) AS age
    FROM pg_stat_activity
    WHERE state <> 'idle'
        AND pid <> pg_backend_pid()
    ORDER BY age;
$$ LANGUAGE SQL;
