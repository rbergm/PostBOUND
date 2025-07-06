SELECT COUNT(*)
FROM title AS t, cast_info AS ci, movie_keyword AS mk
WHERE t.id = mk.movie_id
  AND t.id = ci.movie_id
  AND t.production_year > 1950
  AND t.kind_id = 1;
