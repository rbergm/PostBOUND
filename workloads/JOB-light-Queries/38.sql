SELECT COUNT(*)
FROM title AS t,
  movie_info AS mi,
  movie_info_idx AS mi_idx,
  movie_keyword AS mk
WHERE t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mi_idx.movie_id
  AND t.production_year > 2010
  AND t.kind_id = 1
  AND mi.info_type_id = 8
  AND mi_idx.info_type_id = 101;
