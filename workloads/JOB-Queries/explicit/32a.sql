select count(*) from 
movie_link AS ml 
join link_type AS lt on (lt.id = ml.link_type_id)
join title AS t1 on (ml.movie_id = t1.id)
join title AS t2 on (ml.linked_movie_id = t2.id)
join
 (select movie_id from movie_keyword AS mk 
join keyword AS k on (k.keyword ='10,000-mile-club' AND mk.keyword_id = k.id)) as t_mk 
on(t_mk.movie_id = ml.movie_id);