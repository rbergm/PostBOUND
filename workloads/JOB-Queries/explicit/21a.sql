select count(*) from 
movie_link AS ml 
join link_type AS lt on (lt.link LIKE '%follow%' AND lt.id = ml.link_type_id)
join title AS t on (t.production_year BETWEEN 1950 AND 2000 AND ml.movie_id = t.id)
JOIN movie_companies AS mc  
on(mc.movie_id = ml.movie_id and mc.note IS NULL)
join company_type AS ct on (ct.kind ='production companies' AND mc.company_type_id = ct.id)
join company_name AS cn on (cn.country_code !='[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.company_id = cn.id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword ='sequel' AND mk.keyword_id = k.id)) as t_mk 
on(t_mk.movie_id = mc.movie_id)
JOIN movie_info AS mi  
on(mi.movie_id = t_mk.movie_id and mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish','Denish', 'Norwegian', 'German'));