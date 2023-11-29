-- provide an overview of foreign key constraint defined in a specific namespace.
-- the table that references its partner is described by referencing_relname, the table which has to contain matching columns
-- is contained in target_relname. The foreign key name itself is keyname.
CREATE OR REPLACE FUNCTION pg_fkeys(nspname NAME DEFAULT 'public')
RETURNS TABLE (nspname NAME, referencing_relname NAME, target_relname NAME, keyname NAME)
AS $$
	SELECT nsp.nspname AS nspname,
		cls.relname AS referencing_relname,
		target_cls.relname AS target_relname,
		cons.conname AS keyname
	FROM pg_constraint cons
		JOIN pg_class cls ON cons.conrelid = cls.oid
		JOIN pg_namespace nsp ON cls.relnamespace = nsp.oid
		JOIN pg_class target_cls ON cons.confrelid = target_cls.oid
	WHERE nsp.nspname = nspname
		AND cons.contype = 'f';
$$ LANGUAGE SQL;
