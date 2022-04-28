-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 14b

SELECT COUNT(*)
FROM movie_info_idx AS mi_idx
JOIN info_type AS it2
    ON it2.info = 'rating'
    AND it2.id = mi_idx.info_type_id
    AND mi_idx.info > '6.0'
JOIN title AS t
    ON t.production_year > 2010
    AND (t.title LIKE '%murder%' OR t.title LIKE '%murder%' OR t.title LIKE '%mord%')
    AND t.id = mi_idx.movie_id
JOIN kind_type AS kt
    ON kt.kind = 'movie'
    AND kt.id = t.kind_id
JOIN movie_keyword AS mk
    ON mk.movie_id = mi_idx.movie_id
JOIN keyword AS k
    ON k.keyword IN ('murder', 'murder-in-title')
    AND k.id = mk.keyword_id
JOIN movie_info AS mi
    ON mi.movie_id = mk.movie_id
    AND mi.info IN ('sweden', 'norway', 'germany', 'denmark', 'swedish', 'denish', 'norwegian', 'german', 'usa', 'american')
JOIN info_type AS it1
    ON it1.info = 'countries'
    AND it1.id = mi.info_type_id




-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 23c

SELECT COUNT(*)
FROM movie_info AS mi
JOIN info_type AS it1
    ON it1.info = 'release dates'
    AND it1.id = mi.info_type_id
    AND mi.note LIKE '%internet%'
    AND mi.info IS NOT NULL
    AND (mi.info LIKE 'usa:% 199%' OR mi.info LIKE 'usa:% 200%')
JOIN title AS t
    ON t.production_year > 1990
    AND t.id = mi.movie_id
JOIN kind_type AS kt
    ON kt.kind IN ('movie', 'tv movie', 'video movie', 'video game')
    AND kt.id = t.kind_id
JOIN complete_cast AS cc
    ON cc.movie_id = mi.movie_id
JOIN comp_cast_type AS cct1
    ON cct1.kind = 'complete+verified'
    AND cct1.id = cc.status_id
JOIN movie_companies AS mc
    ON mc.movie_id = cc.movie_id
JOIN company_type AS ct
    ON ct.id = mc.company_type_id
JOIN company_name AS cn
    ON cn.country_code = '[us]'
    AND cn.id = mc.company_id
JOIN movie_keyword AS mk
    ON mk.movie_id = mc.movie_id
JOIN keyword AS k
    ON k.id = mk.keyword_id



-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 9d

SELECT COUNT(*)
FROM aka_name AS an
JOIN name AS n
    ON n.gender = 'f'
    AND an.person_id = n.id
JOIN cast_info AS ci
    ON ci.person_id = an.person_id
    AND ci.note IN ('(voice)', '(voice: japanese version)', '(voice) (uncredited)', '(voice: english version)')
JOIN role_type AS rt
    ON rt.role = 'actress'
    AND ci.role_id = rt.id
JOIN char_name AS chn
    ON chn.id = ci.person_role_id
JOIN title AS t
    ON ci.movie_id = t.id
JOIN movie_companies AS mc
    ON mc.movie_id = ci.movie_id
JOIN company_name AS cn
    ON cn.country_code = '[us]'
    AND mc.company_id = cn.id

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 13b

SELECT COUNT(*)
FROM movie_info_idx AS miidx
JOIN info_type AS it
    ON it.info = 'rating'
    AND it.id = miidx.info_type_id
JOIN title AS t
    ON t.title <> ''
    AND (t.title LIKE '%champion%' OR t.title LIKE '%loser%')
    AND miidx.movie_id = t.id
JOIN kind_type AS kt
    ON kt.kind = 'movie'
    AND kt.id = t.kind_id
JOIN movie_companies AS mc
    ON mc.movie_id = miidx.movie_id
JOIN company_type AS ct
    ON ct.kind = 'production companies'
    AND ct.id = mc.company_type_id
JOIN company_name AS cn
    ON cn.country_code = '[us]'
    AND cn.id = mc.company_id
JOIN movie_info AS mi
    ON mi.movie_id = mc.movie_id
JOIN info_type AS it2
    ON it2.info = 'release dates'
    AND it2.id = mi.info_type_id
