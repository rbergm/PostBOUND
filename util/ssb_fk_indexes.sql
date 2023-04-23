create index if not exists customer_lo_custkey on lineorder(lo_custkey);
create index if not exists part_lo_partkey on lineorder(lo_partkey);
create index if not exists supplier_lo_suppkey on lineorder(lo_suppkey);
create index if not exists date_lo_orderdate on lineorder(lo_orderdate);
create index if not exists date_lo_commitdate on lineorder(lo_commitdate);

