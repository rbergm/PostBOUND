-- provide the cardinality estimate for the result set of a query
CREATE OR REPLACE FUNCTION explain_cardest(q TEXT)
RETURNS INTEGER
AS
$$
DECLARE
	rec RECORD;
	n_rows INTEGER;
BEGIN
	FOR rec IN EXECUTE 'EXPLAIN ' || q LOOP
		n_rows := SUBSTRING(rec."QUERY PLAN" FROM ' rows=([[:digit:]]+)');
		EXIT WHEN n_rows IS NOT NULL;
	END LOOP;

	RETURN n_rows;
END
$$ LANGUAGE plpgsql;

-- provide the final cost estimate of a query
CREATE OR REPLACE FUNCTION explain_costest(q TEXT)
RETURNS NUMERIC
AS
$$
DECLARE
	rec RECORD;
	n_rows NUMERIC;
BEGIN
	FOR rec IN EXECUTE 'EXPLAIN ' || q LOOP
		n_rows := SUBSTRING(rec."QUERY PLAN" FROM 'cost=[[:digit:]]+\.[[:digit:]]{2}\.\.([[:digit:]]+\.[[:digit:]]{2})');
		EXIT WHEN n_rows IS NOT NULL;
	END LOOP;

	RETURN n_rows;
END
$$ LANGUAGE plpgsql;
