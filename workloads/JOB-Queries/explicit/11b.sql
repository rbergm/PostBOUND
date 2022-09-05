select count(*) from 
movie_companies AS mc 
join company_type AS ct on (ct.kind ='production companies' AND mc.company_type_id = ct.id and mc.note IS NULL)
join title AS t on (t.production_year = 1998 AND t.title LIKE '%Money%' AND t.id = mc.movie_id)
join company_name AS cn on (cn.country_code !='[pl]' AND (cn.name LIKE '%Film%' OR cn.name LIKE '%Warner%') AND mc.company_id = cn.id)
join
 (select linked_movie_id, movie_id from movie_link AS ml 
join link_type AS lt on (lt.link LIKE '%follows%' AND lt.id = ml.link_type_id)) as t_ml 
on(t_ml.movie_id = mc.movie_id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword ='sequel' AND mk.keyword_id = k.id)) as t_mk 
on(t_mk.movie_id = t_ml.movie_id);