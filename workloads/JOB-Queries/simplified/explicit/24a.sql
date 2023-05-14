select count(*) from 
movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('hero', 'martial-arts', 'hand-to-hand-combat') AND k.id = mk.keyword_id)
join title AS t on (t.production_year > 2010 AND t.id = mk.movie_id)
JOIN movie_companies AS mc  
on(mc.movie_id = mk.movie_id)
join company_name AS cn on (cn.country_code ='[us]' AND cn.id = mc.company_id)
JOIN cast_info AS ci  
on(ci.movie_id = mc.movie_id and ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)'))
join role_type AS rt on (rt.role ='actress' AND rt.id = ci.role_id)
join char_name AS chn on (chn.id = ci.person_role_id)
JOIN aka_name AS an  
on(an.person_id = ci.person_id)
join name AS n on (n.gender ='f' AND n.name LIKE '%An%' AND n.id = an.person_id)
JOIN movie_info AS mi  
on(mi.movie_id = ci.movie_id and mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%'))
join info_type AS it on (it.info = 'release dates' AND it.id = mi.info_type_id);