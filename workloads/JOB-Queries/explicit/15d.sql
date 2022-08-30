select count(*) from 
movie_info AS mi 
join info_type AS it1 on (it1.info = 'release dates' AND it1.id = mi.info_type_id and mi.note LIKE '%internet%')
join title AS t on (t.production_year > 1990 AND t.id = mi.movie_id)
JOIN aka_title AS at  
on(at.movie_id = mi.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = at.movie_id)
join company_type AS ct on (ct.id = mc.company_type_id)
join company_name AS cn on (cn.country_code = '[us]' AND cn.id = mc.company_id)
JOIN movie_keyword AS mk  
on(mk.movie_id = mc.movie_id)
join keyword AS k on (k.id = mk.keyword_id);