import postbound as pb

pg_instance = pb.postgres.connect(config_file=".psycopg_connection_stats")

print("===== Schema introspection: =====\n")
# Let's start by getting all real tables along with their columns.
schema = pg_instance.schema()

print("Tables:")
for table in schema.tables():
    if schema.is_view(table):
        continue

    columns = schema.columns(table)
    columns_str = ", ".join([col.name for col in columns])
    print(f"- {table.full_name} [{columns_str}]")
print()

print("===== Statistics: =====\n")
# We can also extract simple statistics about the data
stats = pg_instance.statistics()
stats.cache_enabled = False

some_table = next((tab for tab in schema.tables() if not schema.is_view(tab)))
for column in schema.columns(some_table):
    min, max = stats.min_max(column)
    print(f"- {column}: [{min}, {max}]")
