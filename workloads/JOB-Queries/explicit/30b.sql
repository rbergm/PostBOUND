select count(*) from 
complete_cast AS cc 
join comp_cast_type AS cct2 on (cct2.kind ='complete+verified' AND cct2.id = cc.status_id)
join comp_cast_type AS cct1 on (cct1.kind IN ('cast', 'crew') AND cct1.id = cc.subject_id)
join title AS t on (t.production_year > 2000 AND (t.title LIKE '%Freddy%' OR t.title LIKE '%Jason%' OR t.title LIKE 'Saw%') AND t.id = cc.movie_id)
join
 (select movie_id from movie_info_idx AS mi_idx 
join info_type AS it2 on (it2.info = 'votes' AND it2.id = mi_idx.info_type_id)) as t_mi_idx 
on(t_mi_idx.movie_id = cc.movie_id)
JOIN movie_info AS mi  
on(mi.movie_id = t_mi_idx.movie_id and mi.info IN ('Horror', 'Thriller'))
join info_type AS it1 on (it1.info = 'genres' AND it1.id = mi.info_type_id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('murder', 'violence', 'blood', 'gore', 'death', 'female-nudity', 'hospital') AND k.id = mk.keyword_id)) as t_mk 
on(t_mk.movie_id = mi.movie_id)
JOIN cast_info AS ci  
on(ci.movie_id = t_mk.movie_id and ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)'))
join name AS n on (n.gender = 'm' AND n.id = ci.person_id);