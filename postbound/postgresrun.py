
from postbound.db import postgres
from postbound.qal import base

conn_string = "dbname=imdbload user=postgres password=root host=127.0.0.1 port=5432"

postgres_db = postgres.connect(connect_string=conn_string)

result1 = postgres_db.execute_query("SELECT CURRENT_TIME", cache_enabled=False)
table_ref = base.TableReference('movie_info_idx')
column_ref = base.ColumnReference('id', table=table_ref)

#result1 = postgres_db.schema().lookup_column(column_ref, [table_ref])
#result2 = postgres_db.schema().is_primary_key(column_ref)
#result3 = postgres_db.schema().has_secondary_index(column_ref)
#result4 = postgres_db.schema().datatype(column_ref)
result5 = postgres_db.statistics()._retrieve_total_rows_from_stats(table_ref)
result6 = postgres_db.statistics()._retrieve_distinct_values_from_stats(column_ref)
result7 = postgres_db.statistics()._retrieve_min_max_values_from_stats(column_ref)
result8 = postgres_db.statistics()._retrieve_most_common_values_from_stats(column_ref,100)


print(result1)



