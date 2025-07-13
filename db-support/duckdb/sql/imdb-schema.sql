
--
-- base tables
--

CREATE TABLE comp_cast_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE company_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(32) NOT NULL
);

CREATE TABLE info_type (
    id integer NOT NULL PRIMARY KEY,
    info character varying(32) NOT NULL
);

CREATE TABLE keyword (
    id integer NOT NULL PRIMARY KEY,
    keyword text NOT NULL,
    phonetic_code character varying(5)
);

CREATE TABLE kind_type (
    id integer NOT NULL PRIMARY KEY,
    kind character varying(15) NOT NULL
);

CREATE TABLE link_type (
    id integer NOT NULL PRIMARY KEY,
    link character varying(32) NOT NULL
);

CREATE TABLE role_type (
    id integer NOT NULL PRIMARY KEY,
    role character varying(32) NOT NULL
);

--
-- fact tables
--

CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32),

    constraint fk_t_kind_id foreign key (kind_id) references kind_type(id)
);

CREATE TABLE name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE char_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    imdb_index character varying(12),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);

CREATE TABLE company_name (
    id integer NOT NULL PRIMARY KEY,
    name text NOT NULL,
    country_code character varying(255),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);

--
-- dimension tables
--

CREATE TABLE aka_name (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    name text NOT NULL,
    imdb_index character varying(12),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32),

    constraint fk_an_person_id foreign key (person_id) references name(id)
);

CREATE TABLE aka_title (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    title text NOT NULL,
    imdb_index character varying(12),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note text,
    md5sum character varying(32),

    constraint fk_at_movie_id foreign key (movie_id) references title(id)
);

CREATE TABLE cast_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note text,
    nr_order integer,
    role_id integer NOT NULL,

    constraint fk_ci_movie_id foreign key (movie_id) references title(id),
    constraint fk_ci_person_id foreign key (person_id) references name(id),
    constraint fk_ci_person_role_id foreign key (person_role_id) references char_name(id),
    constraint fk_ci_role_id foreign key (role_id) references role_type(id)
);

CREATE TABLE complete_cast (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL,

    constraint fk_cc_subject_id foreign key (subject_id) references comp_cast_type(id),
    constraint fk_cc_status_id foreign key (status_id) references comp_cast_type(id),
    constraint fk_cc_movie_id foreign key (movie_id) references title(id)
);

CREATE TABLE movie_companies (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note text,

    constraint fk_mc_company_id foreign key (company_id) references company_name(id),
    constraint fk_mc_movie_id foreign key (movie_id) references title(id),
    constraint fk_mc_company_type_id foreign key (company_type_id) references company_type(id)
);

CREATE TABLE movie_info (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text,

    constraint fk_mi_movie_id foreign key (movie_id) references title(id),
    constraint fk_mi_info_type_id foreign key (info_type_id) references info_type(id)
);

CREATE TABLE movie_info_idx (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text,

    constraint fk_mi_idx_movie_id foreign key (movie_id) references title(id),
    constraint fk_mi_idx_info_type_id foreign key (info_type_id) references info_type(id)
);

CREATE TABLE movie_keyword (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL,

    constraint fk_mk_movie_id foreign key (movie_id) references title(id),
    constraint fk_mk_keyword_id foreign key (keyword_id) references keyword(id)
);

CREATE TABLE movie_link (
    id integer NOT NULL PRIMARY KEY,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL,

    constraint fk_ml_movie_id foreign key (movie_id) references title(id),
    constraint fk_ml_linked_movie_id foreign key (linked_movie_id) references title(id),
    constraint fk_ml_link_type_id foreign key (link_type_id) references link_type(id)
);

CREATE TABLE person_info (
    id integer NOT NULL PRIMARY KEY,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info text NOT NULL,
    note text,

    constraint fk_pi_info_type_id foreign key (info_type_id) references info_type(id),
    constraint fk_pi_person_id foreign key (person_id) references name(id)
);
