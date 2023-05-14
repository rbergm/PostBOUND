select count(*) from 
movie_keyword AS mk 
join keyword AS k on (k.keyword IN ('superhero', 'sequel', 'second-part', 'marvel-comics', 'based-on-comic', 'tv-special', 'fight','violence') AND k.id = mk.keyword_id)
join title AS t on (t.production_year > 2000 AND t.id = mk.movie_id)
JOIN cast_info AS ci  
on(ci.movie_id = mk.movie_id)
join name AS n on (n.id = ci.person_id);