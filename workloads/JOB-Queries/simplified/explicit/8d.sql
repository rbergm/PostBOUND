select count(*) from 
aka_name AS an1 
join name AS n1 on (an1.person_id = n1.id)
join
 (select person_id, movie_id from cast_info AS ci 
join role_type AS rt on (rt.role ='costume designer' AND ci.role_id = rt.id)
join title AS t on (ci.movie_id = t.id)) as t_ci 
on(t_ci.person_id = an1.person_id)
JOIN movie_companies AS mc  
on(mc.movie_id = t_ci.movie_id)
join company_name AS cn on (cn.country_code ='[us]' AND mc.company_id = cn.id);