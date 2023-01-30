-- Disables parallel workers for all queries in a workload
SET max_parallel_workers = 0;
SET max_parallel_workers_per_gather = 0;
