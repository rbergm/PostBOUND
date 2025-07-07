
--
-- base tables
--

.print 'importing table comp_cast_type'
copy comp_cast_type from 'csv/comp_cast_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table company_type'
copy company_type from 'csv/company_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table info_type'
copy info_type from 'csv/info_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table keyword'
copy keyword from 'csv/keyword.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table kind_type'
copy kind_type from 'csv/kind_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table link_type'
copy link_type from 'csv/link_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table role_type'
copy role_type from 'csv/role_type.csv' delimiter '|' csv  NULL AS '' quote E'\b';

--
-- fact tables
--

.print 'importing table title'
copy title from 'csv/title.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table name'
copy name from 'csv/name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table char_name'
copy char_name from 'csv/char_name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table company_name'
copy company_name from 'csv/company_name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

--
-- dimension tables
--

.print 'importing table aka_name'
copy aka_name from 'csv/aka_name.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table aka_title'
copy aka_title from 'csv/aka_title.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table cast_info'
copy cast_info from 'csv/cast_info.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table complete_cast'
copy complete_cast from 'csv/complete_cast.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table movie_info'
copy movie_info from 'csv/movie_info.csv' delimiter '|' csv  NULL AS '';

.print 'importing table movie_info_idx'
copy movie_info_idx from 'csv/movie_info_idx.csv' delimiter '|' csv  NULL AS '';

.print 'importing table movie_keyword'
copy movie_keyword from 'csv/movie_keyword.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table movie_link'
copy movie_link from 'csv/movie_link.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table person_info'
copy person_info from 'csv/person_info.csv' delimiter '|' csv  NULL AS '' quote E'\b';

.print 'importing table movie_companies'
copy movie_companies from 'csv/movie_companies.csv' delimiter '|' csv  NULL AS '' quote E'\b';
