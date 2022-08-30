select count(*) from 
movie_info_idx AS mi_idx 
join info_type AS it2 on (it2.info = 'rating' AND it2.id = mi_idx.info_type_id and mi_idx.info > '6.0')
join title AS t on (t.production_year > 2010 AND (t.title LIKE '%murder%' OR t.title LIKE '%Murder%' OR t.title LIKE '%Mord%') AND t.id = mi_idx.movie_id)
join kind_type AS kt on (kt.kind = 'movie' AND kt.id = t.kind_id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('murder', 'murder-in-title') AND k.id = mk.keyword_id)) as t_mk 
on(t_mk.movie_id = mi_idx.movie_id)
JOIN movie_info AS mi  
on(mi.movie_id = t_mk.movie_id and mi.info IN ('Sweden', 'Norway', 'Germany',  'Denmark',  'Swedish',  'Denish', 'Norwegian',  'German', 'USA',  'American'))
join info_type AS it1 on (it1.info = 'countries' AND it1.id = mi.info_type_id);