#!/usr/bin/env python3

"""Experiment to determine the true cardinalities of all possible sub-joins of queries in the Join Order Benchmark."""

import argparse
import collections
import json
from typing import Dict, List

import pandas as pd

from analysis import workloads
from transform import db, mosp, util


def calculate_workload_cardinalities(workload: workloads.Workload, *,
                                     n_connections: int = 12,
                                     dbs: db.DBSchema = None, verbose: bool = False) -> pd.DataFrame:
    dbs = db.DBSchema.get_instance() if dbs is None else dbs
    query_pool = db.ParallelQueryExecutor(dbs.connect_string, n_connections, verbose=verbose)

    fragment_to_queries_map: Dict[str, List[mosp.MospQuery]] = collections.defaultdict(list)
    query_fragment_translator: Dict[str, mosp.MospQuery] = {}

    for query in workload.queries():
        join_map = query.predicates().joins
        for sub_join in util.powerset(query.collect_tables()):
            if not sub_join or not join_map.joins_tables(*sub_join):
                continue

            query_fragment = query.extract_fragment(list(sub_join)).as_count_star()
            if query_fragment in fragment_to_queries_map:
                fragment_to_queries_map[query_fragment].append(query)
                continue

            query_fragment_str = str(query_fragment)
            fragment_to_queries_map[query_fragment_str].append(query)
            query_fragment_translator[query_fragment_str] = query_fragment
            query_pool.queue_query(query_fragment_str)

    query_pool.drain_queue()
    cardinalities: Dict[str, int] = query_pool.result_set()

    query_to_fragment_map = util.dict_invert(fragment_to_queries_map)
    queries_list = []
    fragments_list = []
    cardinalities_list = []

    for query, fragments in query_to_fragment_map.items():
        for fragment in fragments:
            fragment_tables = query_fragment_translator[fragment].collect_tables()

            queries_list.append(str(query))
            fragments_list.append(json.dumps([str(tab) for tab in fragment_tables]))
            cardinalities_list.append(cardinalities[fragment])

    return pd.DataFrame({"query": queries_list, "query_fragment": fragments_list, "cardinality": cardinalities_list})


def run(out: str = "job-join-cardinalities.csv", n_connections: int = 12, verbose: bool = False):
    job = workloads.job()
    cardinalities = calculate_workload_cardinalities(job, n_connections=n_connections, verbose=verbose)
    cardinalities.to_csv(out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", "-p", type=int, default=12, help="The number of simultaneous DB connections")
    parser.add_argument("--out", "-o", default="job-join-cardinalities.csv", help="Path to the output file.")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Produce debugging output")

    args = parser.parse_args()

    run(args.out, args.proc, args.verbose)
