CREATE INDEX IF NOT EXISTS customer_lo_custkey ON lineorder(lo_custkey);
CREATE INDEX IF NOT EXISTS part_lo_partkey ON lineorder(lo_partkey);
CREATE INDEX IF NOT EXISTS supplier_lo_suppkey ON lineorder(lo_suppkey);
CREATE INDEX IF NOT EXISTS date_lo_orderdate ON lineorder(lo_orderdate);
CREATE INDEX IF NOT EXISTS date_lo_commitdate ON lineorder(lo_commitdate);
