select count(*) from 
aka_name AS an 
join name AS n on (n.gender ='f' AND n.name LIKE '%Ang%' AND an.person_id = n.id)
JOIN cast_info AS ci  
on(ci.person_id = an.person_id and ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)'))
join role_type AS rt on (rt.role ='actress' AND ci.role_id = rt.id)
join title AS t on (t.production_year BETWEEN 2005 AND 2015 AND ci.movie_id = t.id)
join char_name AS chn on (chn.id = ci.person_role_id)
JOIN movie_companies AS mc  
on(mc.movie_id = ci.movie_id and mc.note IS NOT NULL AND (mc.note LIKE '%(USA)%' OR mc.note LIKE '%(worldwide)%'))
join company_name AS cn on (cn.country_code ='[us]' AND mc.company_id = cn.id);