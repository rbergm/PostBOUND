SELECT COUNT(*)
FROM title AS t,
  cast_info AS ci,
  movie_keyword AS mk,
  movie_info_idx AS mi_idx
WHERE t.id = mk.movie_id
  AND t.id = ci.movie_id
  AND t.id = mi_idx.movie_id
  AND t.production_year > 2005
  AND t.kind_id = 1
  AND mi_idx.info_type_id = 101;
