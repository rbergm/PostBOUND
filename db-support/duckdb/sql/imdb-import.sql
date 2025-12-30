
--
-- base tables
--

copy comp_cast_type from 'csv/comp_cast_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy company_type from 'csv/company_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy info_type from 'csv/info_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy keyword from 'csv/keyword.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy kind_type from 'csv/kind_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy link_type from 'csv/link_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy role_type from 'csv/role_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

--
-- fact tables
--

copy title from 'csv/title.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy name from 'csv/name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy char_name from 'csv/char_name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy company_name from 'csv/company_name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

--
-- dimension tables
--

copy aka_name from 'csv/aka_name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy aka_title from 'csv/aka_title.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy cast_info from 'csv/cast_info.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy complete_cast from 'csv/complete_cast.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy movie_info from 'csv/movie_info.csv' delimiter '|' csv  NULL AS '';

copy movie_info_idx from 'csv/movie_info_idx.csv' delimiter '|' csv  NULL AS '';

copy movie_keyword from 'csv/movie_keyword.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy movie_link from 'csv/movie_link.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy person_info from 'csv/person_info.csv' delimiter '|' csv  NULL AS '' quote E'\b';

copy movie_companies from 'csv/movie_companies.csv' delimiter '|' csv  NULL AS '' quote E'\b';
