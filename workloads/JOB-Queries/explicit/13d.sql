select count(*) from 
movie_info_idx AS miidx 
join info_type AS it on (it.info ='rating' AND it.id = miidx.info_type_id)
join title AS t on (miidx.movie_id = t.id)
join kind_type AS kt on (kt.kind ='movie' AND kt.id = t.kind_id)
join
 (select movie_id from movie_companies AS mc 
join company_type AS ct on (ct.kind ='production companies' AND ct.id = mc.company_type_id)
join company_name AS cn on (cn.country_code ='[us]' AND cn.id = mc.company_id)) as t_mc 
on(t_mc.movie_id = miidx.movie_id)
join
 (select movie_id from movie_info AS mi 
join info_type AS it2 on (it2.info ='release dates' AND it2.id = mi.info_type_id)) as t_mi 
on(t_mi.movie_id = t_mc.movie_id);