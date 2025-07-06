SELECT COUNT(*)
FROM movie_companies AS mc, title AS t, movie_info_idx AS mi_idx
WHERE t.id = mc.movie_id
  AND t.id = mi_idx.movie_id
  AND mi_idx.info_type_id = 112
  AND mc.company_type_id = 2;
