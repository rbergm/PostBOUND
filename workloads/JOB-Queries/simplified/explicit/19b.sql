select count(*) from 
movie_info AS mi 
join info_type AS it on (it.info = 'release dates' AND it.id = mi.info_type_id and mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%2007%' OR mi.info LIKE 'USA:%2008%'))
join title AS t on (t.production_year BETWEEN 2007 AND 2008 AND t.title LIKE '%Kung%Fu%Panda%' AND t.id = mi.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = mi.movie_id and mc.note LIKE '%(200%)%' AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%'))
join company_name AS cn on (cn.country_code ='[us]' AND cn.id = mc.company_id)
JOIN cast_info AS ci  
on(ci.movie_id = mc.movie_id and ci.note = '(voice)')
join role_type AS rt on (rt.role ='actress' AND rt.id = ci.role_id)
join char_name AS chn on (chn.id = ci.person_role_id)
join
 (select person_id from aka_name AS an 
join name AS n on (n.gender ='f' AND n.name LIKE '%Angel%' AND n.id = an.person_id)) as t_an 
on(t_an.person_id = ci.person_id);