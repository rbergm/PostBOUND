select count(*) from 
movie_link AS ml 
join link_type AS lt on (lt.link LIKE '%follow%' AND lt.id = ml.link_type_id)
join title AS t on (t.production_year BETWEEN 1950 AND 2000 AND ml.movie_id = t.id)
join
 (select movie_id from complete_cast AS cc 
join comp_cast_type AS cct2 on (cct2.kind = 'complete' AND cct2.id = cc.status_id)
join comp_cast_type AS cct1 on (cct1.kind IN ('cast', 'crew') AND cct1.id = cc.subject_id)) as t_cc 
on(t_cc.movie_id = ml.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = t_cc.movie_id and mc.note IS NULL)
join company_type AS ct on (ct.kind ='production companies' AND mc.company_type_id = ct.id)
join company_name AS cn on (cn.country_code !='[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.company_id = cn.id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword ='sequel' AND mk.keyword_id = k.id)) as t_mk 
on(t_mk.movie_id = mc.movie_id)
JOIN movie_info AS mi  
on(mi.movie_id = t_mk.movie_id and mi.info IN ('Sweden', 'Germany', 'Swedish', 'German'));