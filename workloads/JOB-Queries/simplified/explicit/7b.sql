select count(*) from 
person_info AS pi 
join info_type AS it on (it.info ='mini biography' AND it.id = pi.info_type_id and pi.note ='Volker Boehm')
join name AS n on (n.name_pcode_cf LIKE 'D%' AND n.gender='m' AND n.id = pi.person_id)
JOIN aka_name AS an  
on(an.person_id = pi.person_id and an.name LIKE '%a%')
JOIN cast_info AS ci  
on(ci.person_id = an.person_id)
join title AS t on (t.production_year BETWEEN 1980 AND 1984 AND t.id = ci.movie_id)
join
 (select linked_movie_id, movie_id from movie_link AS ml 
join link_type AS lt on (lt.link ='features' AND lt.id = ml.link_type_id)) as t_ml 
on(t_ml.linked_movie_id = ci.movie_id);