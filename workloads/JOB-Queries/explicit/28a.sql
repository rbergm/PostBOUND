select count(*) from 
complete_cast AS cc 
join comp_cast_type AS cct1 on (cct1.kind = 'crew' AND cct1.id = cc.subject_id)
join comp_cast_type AS cct2 on (cct2.kind != 'complete+verified' AND cct2.id = cc.status_id)
join title AS t on (t.production_year > 2000 AND t.id = cc.movie_id)
join kind_type AS kt on (kt.kind IN ('movie',  'episode') AND kt.id = t.kind_id)
join
 (select movie_id from movie_info_idx AS mi_idx 
join info_type AS it2 on (it2.info = 'rating' AND it2.id = mi_idx.info_type_id and mi_idx.info < '8.5')) as t_mi_idx 
on(t_mi_idx.movie_id = cc.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = t_mi_idx.movie_id and mc.note NOT LIKE '%(USA)%' AND mc.note LIKE '%(200%)%')
join company_type AS ct on (ct.id = mc.company_type_id)
join company_name AS cn on (cn.country_code != '[us]' AND cn.id = mc.company_id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('murder', 'murder-in-title', 'blood', 'violence') AND k.id = mk.keyword_id)) as t_mk 
on(t_mk.movie_id = mc.movie_id)
JOIN movie_info AS mi  
on(mi.movie_id = t_mk.movie_id and mi.info IN ('Sweden', 'Norway', 'Germany', 'Denmark', 'Swedish','Danish', 'Norwegian', 'German', 'USA', 'American'))
join info_type AS it1 on (it1.info = 'countries' AND it1.id = mi.info_type_id);