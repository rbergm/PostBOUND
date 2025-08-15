SELECT COUNT(*)
FROM cast_info AS ci, title AS t
WHERE t.id = ci.movie_id
  AND t.production_year > 1980
  AND t.production_year < 1995;
