select count(*) from 
movie_keyword AS mk 
join keyword AS k on (k.keyword ='character-name-in-title' AND mk.keyword_id = k.id)
join title AS t on (t.id = mk.movie_id)
join
 (select movie_id from movie_companies AS mc 
join company_name AS cn on (cn.country_code ='[sm]' AND cn.id = mc.company_id)) as t_mc 
on(t_mc.movie_id = mk.movie_id);