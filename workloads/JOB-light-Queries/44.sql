SELECT COUNT(*)
FROM cast_info AS ci,
  title AS t,
  movie_keyword AS mk,
  movie_companies AS mc
WHERE t.id = ci.movie_id
  AND t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.keyword_id = 117;
