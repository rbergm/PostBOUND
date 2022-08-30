select count(*) from 
movie_companies AS mc 
join company_type AS ct on (ct.kind = 'production companies' AND ct.id = mc.company_type_id and (mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%' AND mc.note LIKE '%(co-production)%'))
join title AS t on (t.production_year >2010 AND t.id = mc.movie_id)
join
 (select movie_id from movie_info_idx AS mi_idx 
join info_type AS it on (it.info = 'top 250 rank' AND it.id = mi_idx.info_type_id)) as t_mi_idx 
on(t_mi_idx.movie_id = mc.movie_id);