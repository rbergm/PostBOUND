select count(*) from 
movie_keyword AS mk 
join keyword AS k on (k.keyword ='character-name-in-title' AND mk.keyword_id = k.id)
join title AS t on (t.id = mk.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = mk.movie_id)
join company_name AS cn on (cn.country_code ='[de]' AND cn.id = mc.company_id);