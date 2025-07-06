SELECT COUNT(*)
FROM title AS t, movie_info AS mi, movie_companies AS mc
WHERE t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND t.production_year > 2005
  AND mc.company_type_id = 2;
