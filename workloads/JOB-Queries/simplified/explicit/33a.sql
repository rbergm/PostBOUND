select count(*) from 
movie_link AS ml 
join link_type AS lt on (lt.link IN ('sequel', 'follows', 'followed by') AND lt.id = ml.link_type_id)
join title AS t2 on (t2.production_year BETWEEN 2005 AND 2008 AND t2.id = ml.linked_movie_id)
join title AS t1 on (t1.id = ml.movie_id)
join kind_type AS kt1 on (kt1.kind IN ('tv series') AND kt1.id = t1.kind_id)
join kind_type AS kt2 on (kt2.kind IN ('tv series') AND kt2.id = t2.kind_id)
join
 (select movie_id from movie_info_idx AS mi_idx1 
join info_type AS it1 on (it1.info = 'rating' AND it1.id = mi_idx1.info_type_id)) as t_mi_idx1 
on(t_mi_idx1.movie_id = ml.movie_id)
join
 (select movie_id from movie_info_idx AS mi_idx2 
join info_type AS it2 on (it2.info = 'rating' AND it2.id = mi_idx2.info_type_id and mi_idx2.info < '3.0')) as t_mi_idx2 
on(t_mi_idx2.movie_id = ml.linked_movie_id)
JOIN movie_companies AS mc2  
on(mc2.movie_id = ml.linked_movie_id)
join company_name AS cn2 on (cn2.id = mc2.company_id)
JOIN movie_companies AS mc1  
on(mc1.movie_id = ml.movie_id)
join company_name AS cn1 on (cn1.country_code = '[us]' AND cn1.id = mc1.company_id);