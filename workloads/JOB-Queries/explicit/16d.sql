select count(*) from 
movie_keyword AS mk 
join keyword AS k on (k.keyword ='character-name-in-title' AND mk.keyword_id = k.id)
join title AS t on (t.episode_nr >= 5 AND t.episode_nr < 100 AND t.id = mk.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = mk.movie_id)
join company_name AS cn on (cn.country_code ='[us]' AND mc.company_id = cn.id)
JOIN cast_info AS ci  
on(ci.movie_id = mc.movie_id)
JOIN aka_name AS an  
on(an.person_id = ci.person_id)
join name AS n on (an.person_id = n.id);