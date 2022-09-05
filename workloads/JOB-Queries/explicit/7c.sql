select count(*) from 
movie_link AS ml 
join link_type AS lt on (lt.link IN ('references', 'referenced in', 'features', 'featured in') AND lt.id = ml.link_type_id)
join title AS t on (t.production_year BETWEEN 1980 AND 2010 AND ml.linked_movie_id = t.id)
JOIN cast_info AS ci  
on(ci.movie_id = ml.linked_movie_id)
JOIN aka_name AS an  
on(an.person_id = ci.person_id and an.name IS NOT NULL AND (an.name LIKE '%a%' OR an.name LIKE 'A%'))
JOIN person_info AS pi  
on(pi.person_id = an.person_id and pi.note IS NOT NULL)
join info_type AS it on (it.info ='mini biography' AND it.id = pi.info_type_id)
join name AS n on (n.name_pcode_cf BETWEEN 'A' AND 'F' AND (n.gender='m' OR (n.gender = 'f' AND n.name LIKE 'A%')) AND n.id = pi.person_id);