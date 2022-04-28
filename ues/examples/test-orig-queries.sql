-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 14b

select count(*)
from movie_info_idx as mi_idx
join info_type as it2
    on (it2.info = 'rating'
    and it2.id = mi_idx.info_type_id
    and mi_idx.info > '6.0')
join title as t
    on (t.production_year > 2010
    and (t.title like '%murder%' or t.title like '%murder%' or t.title like '%mord%')
    and t.id = mi_idx.movie_id)
join kind_type as kt
    on (kt.kind = 'movie'
    and kt.id = t.kind_id)
join (select movie_id
    from movie_keyword as mk
        join keyword as k
            on (k.keyword in ('murder', 'murder-in-title')
            and k.id = mk.keyword_id))
    as t_mk
    on(t_mk.movie_id = mi_idx.movie_id)
join movie_info as mi
    on(mi.movie_id = t_mk.movie_id
    and mi.info in ('sweden', 'norway', 'germany',  'denmark',  'swedish',  'denish', 'norwegian',  'german', 'usa',  'american'))
join info_type as it1
    on (it1.info = 'countries'
    and it1.id = mi.info_type_id);


-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 23c

select count(*)
from  movie_info as mi
join info_type as it1
    on (it1.info = 'release dates'
    and it1.id = mi.info_type_id
    and mi.note like '%internet%'
    and mi.info is not null
    and (mi.info like 'usa:% 199%' or mi.info like 'usa:% 200%'))
join title as t
    on (t.production_year > 1990
    and t.id = mi.movie_id)
join kind_type as kt
    on (kt.kind in ('movie', 'tv movie', 'video movie', 'video game')
    and kt.id = t.kind_id)
join (select movie_id
    from complete_cast as cc
        join comp_cast_type as cct1
            on (cct1.kind = 'complete+verified'
            and cct1.id = cc.status_id))
    as t_cc
    on(t_cc.movie_id = mi.movie_id)
join movie_companies as mc
    on(mc.movie_id = t_cc.movie_id)
join company_type as ct
    on (ct.id = mc.company_type_id)
join company_name as cn
    on (cn.country_code = '[us]'
    and cn.id = mc.company_id)
join movie_keyword as mk
    on(mk.movie_id = mc.movie_id)
join keyword as k
    on (k.id = mk.keyword_id);

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 9d

select count(*)
from  aka_name as an
join name as n
    on (n.gender ='f'
    and an.person_id = n.id)
join cast_info as ci
    on(ci.person_id = an.person_id
    and ci.note in ('(voice)', '(voice: japanese version)', '(voice) (uncredited)', '(voice: english version)'))
join role_type as rt
    on (rt.role ='actress'
    and ci.role_id = rt.id)
join char_name as chn
    on (chn.id = ci.person_role_id)
join title as t
    on (ci.movie_id = t.id)
join movie_companies as mc
    on(mc.movie_id = ci.movie_id)
join company_name as cn
    on (cn.country_code ='[us]'
    and mc.company_id = cn.id);

-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
-- 13b

select count(*)
from  movie_info_idx as miidx
join info_type as it
    on (it.info ='rating'
    and it.id = miidx.info_type_id)
join title as t
    on (t.title != ''
    and (t.title like '%champion%' or t.title like '%loser%')
    and miidx.movie_id = t.id)
join kind_type as kt
    on (kt.kind ='movie'
    and kt.id = t.kind_id)
join  (select movie_id
    from movie_companies as mc
        join company_type as ct
            on (ct.kind ='production companies'
            and ct.id = mc.company_type_id)
        join company_name as cn
            on (cn.country_code ='[us]'
            and cn.id = mc.company_id))
    as t_mc
    on(t_mc.movie_id = miidx.movie_id)
join  (select movie_id
    from movie_info as mi
        join info_type as it2
            on (it2.info ='release dates'
            and it2.id = mi.info_type_id))
    as t_mi
    on(t_mi.movie_id = t_mc.movie_id);
