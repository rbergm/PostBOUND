select count(*) from 
movie_companies AS mc 
join company_name AS cn on (cn.name LIKE 'Lionsgate%' AND cn.id = mc.company_id and mc.note LIKE '%(Blu-ray)%')
join title AS t on (t.production_year > 2000 AND (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%') AND t.id = mc.movie_id)
join
 (select movie_id from movie_info_idx AS mi_idx 
join info_type AS it2 on (it2.info = 'votes' AND it2.id = mi_idx.info_type_id)) as t_mi_idx 
on(t_mi_idx.movie_id = mc.movie_id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('murder',  'violence', 'blood', 'gore', 'death', 'female-nudity',  'hospital') AND k.id = mk.keyword_id)) as t_mk 
on(t_mk.movie_id = t_mi_idx.movie_id)
JOIN cast_info AS ci  
on(ci.movie_id = t_mk.movie_id and ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)'))
join name AS n on (n.gender = 'm' AND n.id = ci.person_id)
JOIN movie_info AS mi  
on(mi.movie_id = ci.movie_id and mi.info IN ('Horror', 'Thriller'))
join info_type AS it1 on (it1.info = 'genres' AND it1.id = mi.info_type_id);