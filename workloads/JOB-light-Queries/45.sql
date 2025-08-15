SELECT COUNT(*)
FROM title AS t,
  movie_info AS mi,
  movie_info_idx AS mi_idx,
  cast_info AS ci
WHERE t.id = mi.movie_id
  AND t.id = mi_idx.movie_id
  AND t.id = ci.movie_id
  AND mi.info_type_id = 105
  AND mi_idx.info_type_id = 100;
