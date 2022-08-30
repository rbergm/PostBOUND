select count(*) from 
complete_cast AS cc 
join comp_cast_type AS cct1 on (cct1.kind = 'cast' AND cct1.id = cc.subject_id)
join comp_cast_type AS cct2 on (cct2.kind LIKE '%complete%' AND cct2.id = cc.status_id)
join title AS t on (t.production_year > 2005 AND t.id = cc.movie_id)
join kind_type AS kt on (kt.kind = 'movie' AND kt.id = t.kind_id)
JOIN movie_info_idx AS mi_idx  
on(mi_idx.movie_id = cc.movie_id and mi_idx.info > '8.0')
join info_type AS it2 on (it2.info = 'rating' AND it2.id = mi_idx.info_type_id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('superhero', 'marvel-comics', 'based-on-comic', 'fight') AND k.id = mk.keyword_id)) as t_mk 
on(t_mk.movie_id = mi_idx.movie_id)
JOIN cast_info AS ci  
on(ci.movie_id = t_mk.movie_id)
join char_name AS chn on (chn.name IS NOT NULL AND (chn.name LIKE '%man%' OR chn.name LIKE '%Man%') AND chn.id = ci.person_role_id)
join name AS n on (n.id = ci.person_id);