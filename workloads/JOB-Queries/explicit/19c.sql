select count(*) from 
movie_info AS mi 
join info_type AS it on (it.info = 'release dates' AND it.id = mi.info_type_id and mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%200%' OR mi.info LIKE 'USA:%200%'))
join title AS t on (t.production_year > 2000 AND t.id = mi.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = mi.movie_id)
join company_name AS cn on (cn.country_code ='[us]' AND cn.id = mc.company_id)
JOIN cast_info AS ci  
on(ci.movie_id = mc.movie_id and ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)'))
join role_type AS rt on (rt.role ='actress' AND rt.id = ci.role_id)
join char_name AS chn on (chn.id = ci.person_role_id)
JOIN aka_name AS an  
on(an.person_id = ci.person_id)
join name AS n on (n.gender ='f' AND n.name LIKE '%An%' AND n.id = an.person_id);