CREATE OR REPLACE FUNCTION stats_ndistinct(tabname TEXT,
										   colname TEXT DEFAULT NULL,
										   schemname TEXT DEFAULT NULL)
RETURNS FLOAT AS
$$
DECLARE
	n_distinct FLOAT;
BEGIN
	IF schemname IS NULL THEN
		schemname := 'public';
	END IF;
	IF colname IS NULL THEN
		tabname := SPLIT_PART(tabname, '.', 1);
		colname := SPLIT_PART(tabname, '.', 2);
	END IF;

	SELECT CASE WHEN sts.n_distinct < 0 THEN -1 * cls.reltuples * sts.n_distinct ELSE sts.n_distinct END AS n_distinct
	INTO n_distinct
	FROM pg_stats sts
		JOIN pg_class cls ON cls.oid = sts.tablename::regclass
	WHERE sts.schemaname = schemname
		AND sts.tablename = tabname
		AND sts.attname = colname;

	RETURN n_distinct;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION stats_mcv(tabname TEXT,
									 colname TEXT DEFAULT NULL,
									 schemname TEXT DEFAULT NULL)
RETURNS TABLE (val TEXT, freq REAL) AS
$$
DECLARE
	total_rows FLOAT4;
BEGIN
	IF schemname IS NULL THEN
		schemname := 'public';
	END IF;
	IF colname IS NULL THEN
		tabname := SPLIT_PART(tabname, '.', 1);
		colname := SPLIT_PART(tabname, '.', 2);
	END IF;

	SELECT cls.reltuples INTO total_rows
	FROM pg_class cls WHERE cls.relname = tabname AND cls.relnamespace = schemname::regnamespace;

	RETURN QUERY SELECT UNNEST(most_common_vals::text::text[]) AS val,
						UNNEST(most_common_freqs) * total_rows AS freq
	FROM pg_stats sts
		JOIN pg_class cls ON cls.oid = sts.tablename::regclass
	WHERE sts.schemaname = schemname
		AND sts.tablename = tabname
		AND sts.attname = colname
	ORDER BY freq DESC, val ASC;
END;
$$ LANGUAGE plpgsql;
