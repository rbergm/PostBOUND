select count(*) from 
movie_info_idx AS mi_idx 
join info_type AS it2 on (it2.info ='bottom 10 rank' AND mi_idx.info_type_id = it2.id)
join title AS t on (t.production_year >2000 AND (t.title LIKE 'Birdemic%' OR t.title LIKE '%Movie%') AND t.id = mi_idx.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = mi_idx.movie_id)
join company_type AS ct on (ct.kind IS NOT NULL AND (ct.kind ='production companies' OR ct.kind = 'distributors') AND ct.id = mc.company_type_id)
join company_name AS cn on (cn.country_code ='[us]' AND cn.id = mc.company_id)
join
 (select movie_id from movie_info AS mi 
join info_type AS it1 on (it1.info ='budget' AND mi.info_type_id = it1.id)) as t_mi 
on(t_mi.movie_id = mc.movie_id);