SELECT COUNT(*)
FROM movie_companies AS mc, title AS t, movie_keyword AS mk
WHERE t.id = mc.movie_id
  AND t.id = mk.movie_id
  AND mk.keyword_id = 117;
