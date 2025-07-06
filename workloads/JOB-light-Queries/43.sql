SELECT COUNT(*)
FROM title AS t,
  movie_info AS mi,
  movie_companies AS mc,
  movie_keyword AS mk
WHERE t.id = mi.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mi.info_type_id = 16
  AND t.production_year > 1990;
