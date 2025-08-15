SELECT COUNT(*)
FROM cast_info AS ci, title AS t, movie_companies AS mc
WHERE t.id = ci.movie_id
  AND t.id = mc.movie_id
  AND ci.role_id = 7;
