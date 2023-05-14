select count(*) from 
movie_companies AS mc 
join title AS t on (t.production_year BETWEEN 2006 AND 2007 AND (t.title LIKE 'One Piece%' OR t.title LIKE 'Dragon Ball Z%') AND t.id = mc.movie_id and mc.note LIKE '%(Japan)%' AND mc.note NOT LIKE '%(USA)%' AND (mc.note LIKE '%(2006)%' OR mc.note LIKE '%(2007)%'))
join company_name AS cn on (cn.country_code ='[jp]' AND mc.company_id = cn.id)
JOIN cast_info AS ci  
on(ci.movie_id = mc.movie_id and ci.note ='(voice: English version)')
join role_type AS rt on (rt.role ='actress' AND ci.role_id = rt.id)
JOIN aka_name AS an  
on(an.person_id = ci.person_id)
join name AS n on (n.name LIKE '%Yo%' AND n.name NOT LIKE '%Yu%' AND an.person_id = n.id);