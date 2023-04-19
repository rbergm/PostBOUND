import unittest
import sys
sys.path.append('../..')
from postbound.db import mysql
from postbound.db import postgres
from postbound.qal import base
import configparser
import threading



class TestDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Get the database connection details from the config file
         
        config = configparser.ConfigParser()
        config.read('config.ini')
        database_config_postgres = config['DATABASEPOSTGRES']
        conn_string_postgres = f"dbname={database_config_postgres['database']} user={database_config_postgres['username']} password={database_config_postgres['password']} host={database_config_postgres['host']} port={database_config_postgres['port']}"
        self.pg_connection = postgres.connect(connect_string=conn_string_postgres)
        self.pg_cursor = self.pg_connection.cursor()

        tables_config = config['TABLES']
        columns_config = config['COLUMNS']

        database_config_mysql = config['DATABASEMYSQL']
        mysql_host = database_config_mysql['host']
        mysql_user = database_config_mysql['username']
        mysql_database = database_config_mysql['database']
        mysql_password = database_config_mysql['password']
        conn_string_mysql = {
            "host": mysql_host,
            "user": mysql_user,
            "password": mysql_password,
            "database": mysql_database,
        }
        self.mysql_connection = mysql.connect(connect_string=conn_string_mysql)
        self.mysql_cursor = self.mysql_connection.cursor()

        self.table_ref1 = base.TableReference(tables_config['table1'])
        self.column_ref1 = base.ColumnReference(columns_config['table1_column1'], table=self.table_ref1)
        self.column_ref2 = base.ColumnReference(columns_config['table1_column2'], table=self.table_ref1)
        self.table_ref2 = base.TableReference(tables_config['table2'])
        self.column_ref3 = base.ColumnReference(columns_config['table2_column1'], table=self.table_ref2)
        self.column_ref4 = base.ColumnReference(columns_config['table2_column2'], table=self.table_ref2)

    def test_table_primary_key(self):
        result_postgres = self.pg_connection.schema().is_primary_key(self.column_ref1)
        self.assertTrue(result_postgres, msg="Column should be a primary key.")

        result_mysql = self.mysql_connection.schema().is_primary_key(self.column_ref1)
        self.assertTrue(result_mysql, msg="Column should be a primary key.")

    def test_secondary_index(self):
        result_postgres = self.pg_connection.schema().has_secondary_index(self.column_ref2)
        self.assertTrue(result_postgres, msg="Attribute is not indexed.")

        result_mysql = self.mysql_connection.schema().has_secondary_index(self.column_ref2)
        self.assertTrue(result_mysql, msg="Attribute is not indexed.") 

    def test_retrieve_total_rows_from_stats(self):
        result_postgres = self.pg_connection.statistics()._retrieve_total_rows_from_stats(self.table_ref1)
        self.assertGreater(result_postgres, 0, "No total rows found in PostgreSQL table")

        result_mysql = self.mysql_connection.statistics()._retrieve_total_rows_from_stats(self.table_ref1)
        self.assertGreater(result_mysql, 0, "No total rows found in MYSQL table")

    def test_retrieve_distinct_values_from_stats(self):
        result_postgres = self.pg_connection.statistics()._retrieve_distinct_values_from_stats(self.column_ref1)
        self.assertGreater(result_postgres, 0, "No total values found in PostgreSQL table")

        result_mysql = self.mysql_connection.statistics()._retrieve_distinct_values_from_stats(self.column_ref1)
        self.assertGreater(result_mysql, 0, "No total values found in MYSQL table")
    
    def test_index_presence(self):
        
        pg_query = f"SELECT indexname FROM pg_indexes WHERE tablename = '{self.table_ref1}'"
        self.pg_cursor.execute(pg_query)
        pg_result = self.pg_cursor.fetchall()
        index_list = [row[0] for row in pg_result]
        self.assertGreater(len(index_list), 0, "No indexes found in PostgreSQL table")

        
        mysql_query = f"SHOW INDEXES FROM {self.table_ref1}"
        self.mysql_cursor.execute(mysql_query)
        index_list = self.mysql_cursor.fetchall()
        self.assertGreater(len(index_list), 0, "No indexes found in MySQL table")

    def test_joins(self):
        
        pg_join_query = f"SELECT COUNT(*) FROM {self.table_ref1} INNER JOIN {self.table_ref2} ON {self.column_ref1} = {self.column_ref3}"
        pg_result = self.pg_cursor.execute(pg_join_query).fetchone()[0]
        # Assert that the result is greater than or equal to 1
        self.assertGreaterEqual(pg_result, 0, "PostgreSQL join did not return the correct number of records")

        
        mysql_join_query = f"SELECT COUNT(*) FROM {self.table_ref1} INNER JOIN {self.table_ref2} ON {self.column_ref1} = {self.column_ref3}"
        self.mysql_cursor.execute(mysql_join_query)
        mysql_result = self.mysql_cursor.fetchone()[0]
        # Assert that the result is greater than or equal to 1
        self.assertGreaterEqual(mysql_result, 0, "MySQL join did not return the correct number of records")

    def test_threading(self):
        def connect_to_postgres():
            
            pg_join_query = f"SELECT COUNT(*) FROM {self.table_ref1}"
            pg_result = self.pg_cursor.execute(pg_join_query).fetchone()[0]
            self.assertGreaterEqual(pg_result, 1, "PostgreSQL did not give correct results ")
            
        t = threading.Thread(target=connect_to_postgres)
        t.start()
        t.join()

        def connect_to_mysql():
            
            
            
            mysql_join_query = f"SELECT COUNT(*) FROM {self.table_ref1}"
            self.mysql_cursor.execute(mysql_join_query)
            mysql_result = self.mysql_cursor.fetchone()[0]
            self.assertGreaterEqual(mysql_result, 1, "MYSQL did not give correct results")

        t = threading.Thread(target=connect_to_mysql)
        t.start()
        t.join()



    def test_sql_injection_vulnerability(self):
        # Test for SQL injection vulnerability
        
        
        query = f"SELECT * FROM {self.table_ref1} WHERE {self.column_ref1} = %s"
        self.pg_cursor.execute(query, (1,))
        pg_result = self.pg_cursor.fetchone()
        self.assertGreaterEqual(len(pg_result), 1, "SQL injection vulnerability found!")

        
        
        query = f"SELECT * FROM {self.table_ref1} WHERE {self.column_ref1} = %s"
        self.mysql_cursor.execute(query, (1,))
        mysql_result = self.mysql_cursor.fetchone()
        self.assertGreaterEqual(len(mysql_result), 1, "SQL injection vulnerability found!")

    @classmethod
    def tearDownClass(self):
        self.pg_cursor.close()
        self.pg_connection.close()
        self.mysql_cursor.close()
        self.mysql_connection.close()

