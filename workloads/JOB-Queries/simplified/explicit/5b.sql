select count(*) from 
movie_companies AS mc 
join company_type AS ct on (ct.kind = 'production companies' AND ct.id = mc.company_type_id and mc.note LIKE '%(VHS)%' AND mc.note LIKE '%(USA)%' AND mc.note LIKE '%(1994)%')
join title AS t on (t.production_year > 2010 AND t.id = mc.movie_id)
JOIN movie_info AS mi  
on(mi.movie_id = mc.movie_id and mi.info IN ('USA','America'))
join info_type AS it on (it.id = mi.info_type_id);