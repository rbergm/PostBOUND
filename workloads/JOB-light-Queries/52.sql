SELECT COUNT(*)
FROM title AS t, cast_info AS ci, movie_keyword AS mk
WHERE t.id = mk.movie_id
  AND t.id = ci.movie_id
  AND t.production_year > 2000
  AND t.kind_id = 1;
