select count(*) from 
movie_keyword AS mk 
join keyword AS k on (k.keyword ='character-name-in-title' AND mk.keyword_id = k.id)
join title AS t on (t.id = mk.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = mk.movie_id)
join company_name AS cn on (mc.company_id = cn.id)
join
 (select person_id, movie_id from cast_info AS ci 
join name AS n on (n.name LIKE 'X%' AND n.id = ci.person_id)) as t_ci 
on(t_ci.movie_id = mc.movie_id);