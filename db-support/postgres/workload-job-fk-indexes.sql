--
-- Creates all foreign keys for the IMDB schema, as well as all indexes on the foreign key columns.
--
-- Since PostgreSQL does not automatically create indexes for foreign keys, we add them separately once the constraints
-- have been added.
-- Furthermore, notice that the IMDB dump is not completly sound from a logical perspective, i.e. there are some foreign key
-- tuples which do not have a matching primary key partner. We delete these "dangling" foreign key tuples first.
-- The actual indexes are based on the IMDB schema as depicted by V. Leis et al. in their paper "Query Optimization Through
-- the Looking Glass, and What We Found Running the Join Order Benchmark" (VLDB Journal 27, 2018.
-- https://doi.org/10.1007/s00778-017-0480-7). See Figure 2 for the schema.
--

BEGIN TRANSACTION;
SET CONSTRAINTS ALL DEFERRED;

--
-- Phase 1: delete all FK tuples that have no partner in the PK table
--
-- We use a slightly more complicated selection criteria instead of a NOT IN predicate.
-- This is because the Postgres optimizer often fully materiliazes the result of a NOT IN subquery, which is highly
-- inefficient for our use-case.
-- On the other hand, the dependent subquery combined with a NOT EXISTS predicate guides the optimizer towards the usage
-- of proper anti joins (Postgres does not have an actual ANTI JOIN keyword as of v16).
--
-- The order in which the DELETE statements take place matters because we need to resolve some chains of foreign key
-- references. More specifically, we need to start with the title and aka_name tables, since they contain own foreign key
-- columns and act as foreign keys for other tables as well. The order of the other DELETE statements is arbitrary.
--
--

DELETE FROM title t
WHERE NOT EXISTS (SELECT 1 FROM kind_type kt WHERE kt.id = t.kind_id)
    AND t.kind_id IS NOT NULL;

DELETE FROM aka_name an
WHERE NOT EXISTS (SELECT 1 FROM name n WHERE n.id = an.person_id)
    AND an.person_id IS NOT NULL;

DELETE FROM aka_title at
WHERE NOT EXISTS (SELECT 1 FROM title t WHERE t.id = at.movie_id)
    AND at.movie_id IS NOT NULL;

DELETE FROM cast_info ci
WHERE (NOT EXISTS (SELECT 1 FROM title t WHERE t.id = ci.movie_id) AND ci.movie_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM name n WHERE n.id = ci.person_id) AND ci.person_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM char_name chn WHERE chn.id = ci.person_role_id) AND ci.person_role_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM role_type rt WHERE rt.id = ci.role_id) AND ci.role_id IS NOT NULL);

DELETE FROM complete_cast cc
WHERE (NOT EXISTS (SELECT 1 FROM comp_cast_type cct WHERE cct.id = cc.subject_id) AND cc.subject_id IS NOT NULL)
    OR (NOT EXISTS (SELECT 1 FROM comp_cast_type cct WHERE cct.id = cc.status_id) AND cc.status_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM title t WHERE t.id = cc.movie_id) AND cc.movie_id IS NOT NULL);

DELETE FROM movie_companies mc
WHERE (NOT EXISTS (SELECT 1 FROM company_name cn WHERE cn.id = mc.company_id) AND mc.company_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM title t WHERE t.id = mc.movie_id) AND mc.movie_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM company_type ct WHERE ct.id = mc.company_type_id) AND mc.company_type_id IS NOT NULL);

DELETE FROM movie_info mi
WHERE (NOT EXISTS (SELECT 1 FROM title t WHERE t.id = mi.movie_id) AND mi.movie_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM info_type it WHERE it.id = mi.info_type_id) AND mi.info_type_id IS NOT NULL);

DELETE FROM movie_info_idx mi_idx
WHERE (NOT EXISTS (SELECT 1 FROM title t WHERE t.id = mi_idx.movie_id) AND mi_idx.movie_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM info_type it WHERE it.id = mi_idx.info_type_id) AND mi_idx.info_type_id IS NOT NULL);

DELETE FROM movie_keyword mk
WHERE (NOT EXISTS (SELECT 1 FROM title t WHERE t.id = mk.movie_id) AND mk.movie_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM keyword k WHERE k.id = mk.keyword_id) AND mk.keyword_id IS NOT NULL);

DELETE FROM movie_link ml
WHERE (NOT EXISTS (SELECT 1 FROM title t WHERE t.id = ml.movie_id OR t.id = ml.linked_movie_id) AND ml.linked_movie_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM link_type lt WHERE lt.id = ml.link_type_id) AND ml.link_type_id IS NOT NULL);

DELETE FROM person_info pi
WHERE (NOT EXISTS(SELECT 1 FROM name n WHERE n.id = pi.person_id) AND pi.person_id IS NOT NULL)
	OR (NOT EXISTS (SELECT 1 FROM info_type it WHERE it.id = pi.info_type_id) AND pi.info_type_id IS NOT NULL);

--
-- Phase 2: create all foreign keys
--
-- All foreign keys share a common naming pattern: fk_{table short form}_{column name}
-- The table short form is based on the JOB queries, e.g. title is referred to as t and movie_info is referred to as mi.
-- However, the underscore _ does not act as a perfect separator: movie_info_idx is abbreviated as mi_idx.
--

ALTER TABLE aka_name
ADD CONSTRAINT fk_an_person_id
FOREIGN KEY (person_id)
REFERENCES name(id)
ON DELETE CASCADE;

ALTER TABLE aka_title
ADD CONSTRAINT fk_at_movie_id
FOREIGN KEY (movie_id)
REFERENCES title(id)
ON DELETE CASCADE;

ALTER TABLE cast_info
ADD CONSTRAINT fk_ci_movie_id
    FOREIGN KEY (movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_ci_person_id
    FOREIGN KEY (person_id)
    REFERENCES name(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_ci_person_role_id
    FOREIGN KEY (person_role_id)
    REFERENCES char_name(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_ci_role_id
    FOREIGN KEY (role_id)
    REFERENCES role_type(id)
    ON DELETE CASCADE;

ALTER TABLE complete_cast
ADD CONSTRAINT fk_cc_subject_id
    FOREIGN KEY (subject_id)
    REFERENCES comp_cast_type(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_cc_status_id
    FOREIGN KEY (status_id)
    REFERENCES comp_cast_type(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_cc_movie_id
    FOREIGN KEY (movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE;

ALTER TABLE movie_companies
ADD CONSTRAINT fk_mc_company_id
    FOREIGN KEY (company_id)
    REFERENCES company_name(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_mc_movie_id
    FOREIGN KEY (movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_mc_company_type_id
    FOREIGN KEY (company_type_id)
    REFERENCES company_type(id)
    ON DELETE CASCADE;

ALTER TABLE movie_info
ADD CONSTRAINT fk_mi_movie_id
    FOREIGN KEY (movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_mi_info_type_id
    FOREIGN KEY (info_type_id)
    REFERENCES info_type(id)
    ON DELETE CASCADE;

ALTER TABLE movie_info_idx
ADD CONSTRAINT fk_mi_idx_movie_id
    FOREIGN KEY (movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_mi_idx_info_type_id
    FOREIGN KEY (info_type_id)
    REFERENCES info_type(id)
    ON DELETE CASCADE;

ALTER TABLE movie_keyword
ADD CONSTRAINT fk_mk_movie_id
    FOREIGN KEY (movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_mk_keyword_id
    FOREIGN KEY (keyword_id)
    REFERENCES keyword(id)
    ON DELETE CASCADE;

ALTER TABLE movie_link
ADD CONSTRAINT fk_ml_movie_id
    FOREIGN KEY (movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_ml_linked_movie_id
    FOREIGN KEY (linked_movie_id)
    REFERENCES title(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_ml_link_type_id
    FOREIGN KEY (link_type_id)
    REFERENCES link_type(id)
    ON DELETE CASCADE;

ALTER TABLE person_info
ADD CONSTRAINT fk_pi_info_type_id
    FOREIGN KEY (info_type_id)
    REFERENCES info_type(id)
    ON DELETE CASCADE,
ADD CONSTRAINT fk_pi_person_id
    FOREIGN KEY (person_id)
    REFERENCES name(id)
    ON DELETE CASCADE;

ALTER TABLE title
ADD CONSTRAINT fk_t_kind_id
FOREIGN KEY (kind_id)
REFERENCES kind_type(id)
ON DELETE CASCADE;

--
-- Phase 3: create the actual indexes
--
-- All indexes share a common naming pattern: {table short form}_{column name} indicating which column is being indexed.
-- The table short form is based on the JOB queries, e.g. title is referred to as t and movie_info is referred to as mi.
-- However, the underscore _ does not act as a perfect separator: movie_info_idx is abbreviated as mi_idx.
--

CREATE INDEX IF NOT EXISTS an_person_id
ON aka_name(person_id);

CREATE INDEX IF NOT EXISTS at_movie_id
ON aka_title(movie_id);

CREATE INDEX IF NOT EXISTS ci_movie_id
ON cast_info(movie_id);
CREATE INDEX IF NOT EXISTS ci_person_id
ON cast_info(person_id);
CREATE INDEX IF NOT EXISTS ci_person_role_id
ON cast_info(person_role_id);
CREATE INDEX IF NOT EXISTS ci_role_id
ON cast_info(role_id);

CREATE INDEX IF NOT EXISTS cc_subject_id
ON complete_cast(subject_id);
CREATE INDEX IF NOT EXISTS cc_status_id
ON complete_cast(status_id);
CREATE INDEX IF NOT EXISTS cc_movie_id
ON complete_cast(movie_id);

CREATE INDEX IF NOT EXISTS mc_company_id
ON movie_companies(company_id);
CREATE INDEX IF NOT EXISTS mc_movie_id
ON movie_companies(movie_id);
CREATE INDEX IF NOT EXISTS mc_company_type_id
ON movie_companies(company_type_id);

CREATE INDEX IF NOT EXISTS mi_movie_id
ON movie_info(movie_id);
CREATE INDEX IF NOT EXISTS mi_info_type_id
ON movie_info(info_type_id);

CREATE INDEX IF NOT EXISTS mi_idx_movie_id
ON movie_info_idx(movie_id);
CREATE INDEX IF NOT EXISTS mi_idx_info_type_id
ON movie_info_idx(info_type_id);

CREATE INDEX IF NOT EXISTS mk_movie_id
ON movie_keyword(movie_id);
CREATE INDEX IF NOT EXISTS mk_keyword_id
ON movie_keyword(keyword_id);

CREATE INDEX IF NOT EXISTS ml_movie_id
ON movie_link(movie_id);
CREATE INDEX IF NOT EXISTS ml_linked_movie_id
ON movie_link(linked_movie_id);
CREATE INDEX IF NOT EXISTS ml_link_type_id
ON movie_link(link_type_id);

CREATE INDEX IF NOT EXISTS pi_info_type_id
ON person_info(info_type_id);
CREATE INDEX IF NOT EXISTS pi_person_id
ON person_info(person_id);

CREATE INDEX IF NOT EXISTS t_kind_id
ON title(kind_id);


COMMIT;
