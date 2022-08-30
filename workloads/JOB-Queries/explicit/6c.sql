select count(*) from 
movie_keyword AS mk 
join keyword AS k on (k.keyword = 'marvel-cinematic-universe' AND k.id = mk.keyword_id)
join title AS t on (t.production_year > 2014 AND t.id = mk.movie_id)
join
 (select person_id, movie_id from cast_info AS ci 
join name AS n on (n.name LIKE '%Downey%Robert%' AND n.id = ci.person_id)) as t_ci 
on(t_ci.movie_id = mk.movie_id);