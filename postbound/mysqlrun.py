from postbound.db import mysql
from postbound.qal import base

conn_string = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "1234",
    "database": "imdbload",
    
}

mysql_db = mysql.connect(connect_string=conn_string)

table_ref = base.TableReference('movie_info_idx')
column_ref = base.ColumnReference('id', table=table_ref)

#result1 = mysql_db.schema().lookup_column(column_ref, [table_ref])
#result2 = mysql_db.schema().is_primary_key(column_ref)
#result3 = mysql_db.schema().has_secondary_index(column_ref)
result4 = mysql_db.schema().datatype(column_ref)
result5 = mysql_db.statistics()._retrieve_total_rows_from_stats(table_ref)
result6 = mysql_db.statistics()._retrieve_distinct_values_from_stats(column_ref)
result7 = mysql_db.statistics()._retrieve_min_max_values_from_stats(column_ref)

result8 = mysql_db.statistics()._retrieve_most_common_values_from_stats(column_ref,20)

print(result8)






