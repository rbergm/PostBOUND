select count(*) from 
aka_name AS an 
join name AS n on (n.gender ='f' AND n.id = an.person_id)
JOIN cast_info AS ci  
on(ci.person_id = an.person_id and ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)'))
join role_type AS rt on (rt.role ='actress' AND rt.id = ci.role_id)
join title AS t on (t.production_year > 2000 AND t.id = ci.movie_id)
join char_name AS chn on (chn.id = ci.person_role_id)
JOIN movie_companies AS mc  
on(mc.movie_id = ci.movie_id)
join company_name AS cn on (cn.country_code ='[us]' AND cn.id = mc.company_id)
join
 (select movie_id from movie_info AS mi 
join info_type AS it on (it.info = 'release dates' AND it.id = mi.info_type_id)) as t_mi 
on(t_mi.movie_id = mc.movie_id);