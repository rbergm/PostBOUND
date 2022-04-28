-- JOB 18a
select count(*)
from movie_info_idx AS mi_idx
    join info_type AS it2 on (it2.info = 'votes' AND it2.id = mi_idx.info_type_id)
    join title AS t on (t.id = mi_idx.movie_id)
    join
        (
            select person_id, movie_id
            from cast_info AS ci
                join name AS n on
                    (
                        n.gender = 'm' AND n.name LIKE '%Tim%' AND n.id = ci.person_id AND ci.note IN ('(producer)', '(executive producer)')
                    )
        )
        as t_ci on(t_ci.movie_id = mi_idx.movie_id)
    join (select movie_id from movie_info AS mi join info_type AS it1 on (it1.info = 'budget' AND it1.id = mi.info_type_id)) as t_mi on(t_mi.movie_id = t_ci.movie_id);
