SELECT COUNT(*)
FROM title AS t, movie_keyword AS mk, movie_companies AS mc
WHERE t.id = mk.movie_id
  AND t.id = mc.movie_id
  AND mk.keyword_id = 398
  AND mc.company_type_id = 2;
