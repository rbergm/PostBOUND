select count(*) from 
aka_name AS an 
join name AS n on (n.gender ='f' AND n.name LIKE '%An%' AND an.person_id = n.id)
JOIN cast_info AS ci  
on(ci.person_id = an.person_id and ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)'))
join role_type AS rt on (rt.role ='actress' AND ci.role_id = rt.id)
join char_name AS chn on (chn.id = ci.person_role_id)
join title AS t on (ci.movie_id = t.id)
JOIN movie_companies AS mc  
on(mc.movie_id = ci.movie_id)
join company_name AS cn on (cn.country_code ='[us]' AND mc.company_id = cn.id);