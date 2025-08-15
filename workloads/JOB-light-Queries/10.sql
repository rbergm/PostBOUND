SELECT COUNT(*)
FROM title AS t, movie_info_idx AS mi_idx, movie_keyword AS mk
WHERE t.id = mi_idx.movie_id
  AND t.id = mk.movie_id
  AND t.production_year > 2010
  AND mi_idx.info_type_id = 101;
