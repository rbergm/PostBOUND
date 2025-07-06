SELECT COUNT(*)
FROM movie_keyword AS mk, title AS t, cast_info AS ci
WHERE t.id = mk.movie_id
  AND t.id = ci.movie_id
  AND t.production_year > 2010
  AND mk.keyword_id = 8200;
