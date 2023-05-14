select count(*) from 
movie_info AS mi 
join info_type AS it1 on (it1.info = 'genres' AND it1.id = mi.info_type_id and mi.info IN ('Horror', 'Thriller') AND mi.note IS NULL)
join title AS t on (t.production_year BETWEEN 2008 AND 2014 AND t.id = mi.movie_id)
JOIN movie_info_idx AS mi_idx  
on(mi_idx.movie_id = mi.movie_id and mi_idx.info > '8.0')
join info_type AS it2 on (it2.info = 'rating' AND it2.id = mi_idx.info_type_id)
JOIN cast_info AS ci  
on(ci.movie_id = mi_idx.movie_id and ci.note IN ('(writer)',  '(head writer)',  '(written by)', '(story)', '(story editor)'))
join name AS n on (n.gender IS NOT NULL AND n.gender = 'f' AND n.id = ci.person_id);