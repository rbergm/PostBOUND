select count(*) from 
movie_companies AS mc 
join company_name AS cn on (cn.country_code ='[jp]' AND mc.company_id = cn.id and mc.note LIKE '%(Japan)%' AND mc.note NOT LIKE '%(USA)%')
join title AS t on (t.id = mc.movie_id)
JOIN cast_info AS ci  
on(ci.movie_id = mc.movie_id and ci.note ='(voice: English version)')
join role_type AS rt on (rt.role ='actress' AND ci.role_id = rt.id)
JOIN aka_name AS an1  
on(an1.person_id = ci.person_id)
join name AS n1 on (n1.name LIKE '%Yo%' AND n1.name NOT LIKE '%Yu%' AND an1.person_id = n1.id);