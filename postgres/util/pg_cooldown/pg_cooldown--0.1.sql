/* contrib/pg_cooldown/pg_cooldown--0.1.sql */

-- complain if script is sourced in psql, rather than via CREATE EXTENSION
\echo Use "CREATE EXTENSION pg_cooldown" to load this file. \quit

-- Register the function.
CREATE FUNCTION pg_cooldown(regclass)
RETURNS int8
AS 'MODULE_PATHNAME', 'pg_cooldown'
LANGUAGE C PARALLEL SAFE;
