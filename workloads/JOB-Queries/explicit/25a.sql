select count(*) from 
movie_info AS mi 
join info_type AS it1 on (it1.info = 'genres' AND it1.id = mi.info_type_id and mi.info = 'Horror')
join title AS t on (t.id = mi.movie_id)
join
 (select movie_id from movie_info_idx AS mi_idx 
join info_type AS it2 on (it2.info = 'votes' AND it2.id = mi_idx.info_type_id)) as t_mi_idx 
on(t_mi_idx.movie_id = mi.movie_id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('murder', 'blood',  'gore',  'death', 'female-nudity') AND k.id = mk.keyword_id)) as t_mk 
on(t_mk.movie_id = t_mi_idx.movie_id)
JOIN cast_info AS ci  
on(ci.movie_id = t_mk.movie_id and ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)'))
join name AS n on (n.gender = 'm' AND n.id = ci.person_id);