#!/usr/bin/env python3

import argparse
import collections
import math
import operator
import random
import sys
from typing import Dict, List, Set, Iterator, Tuple


attr_values = "abcdefghijklmnopqrstuvwxyz"


def print_stderr(*args, condition: bool = None, **kwargs):
    if condition is False:
        return
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


class TopKList:
    def __init__(self, value_frequencies: List[Tuple[str, int]], *, k: int = 5):
        sorted_frequencies = sorted(value_frequencies, key=operator.itemgetter(1), reverse=True)

        self.k = k
        self.entries = sorted_frequencies[:self.k]
        self.topk_list = dict(self.entries)

    def max_freq(self) -> int:
        return max(self.topk_list.values())

    def star_freq(self) -> int:
        return min(self.topk_list.values())

    def attribute_values(self) -> Set[str]:
        return set(self.topk_list.keys())

    def overlaps_with(self, other: "TopKList") -> bool:
        return len(self.attribute_values() & other.attribute_values()) > 0

    def __contains__(self, attr: str) -> bool:
        return attr in self.topk_list

    def __getitem__(self, attr: str) -> int:
        return self.topk_list.get(attr, self.star_freq())

    def __iter__(self) -> Iterator[str]:
        return [val for val, __ in self.entries].__iter__()

    def __len__(self) -> int:
        return len(self.entries)

    def __str__(self) -> str:
        return str(self.entries)


def generate_tuples(num_tuples: int, distinct_values: int) -> List[str]:
    available_values = attr_values[:distinct_values]
    weights = [random.randint(0, 10) for __ in range(distinct_values)]
    return random.choices(available_values, weights, k=num_tuples)


def count_value_occurrences(attribute_values: List[str]) -> Dict[str, int]:
    occurrence_counter = collections.defaultdict(int)
    for val in attribute_values:
        occurrence_counter[val] += 1
    return occurrence_counter


def max_frequency(attribute_values: List[str]) -> int:
    occurrence_counter = count_value_occurrences(attribute_values)
    return max(occurrence_counter.values())


def build_topk_list(attribute_values: List[str], *, k: int, exceed_value: int = 0) -> TopKList:
    exceed_value = exceed_value if exceed_value else 0
    occurrence_counter = count_value_occurrences(attribute_values)

    attribute_frequencies = [(attr, freq) for attr, freq in occurrence_counter.items()]
    attribute_frequencies = [(attr, freq + random.randint(0, exceed_value)) for attr, freq in attribute_frequencies]
    return TopKList(attribute_frequencies, k=k)


def calculate_ues_bound(max_freq_r: int, max_freq_s: int, num_tuples_r: int, num_tuples_s: int, *,
                        verbose: bool = False):
    distinct_r = math.ceil(num_tuples_r / max_freq_r)
    distinct_s = math.ceil(num_tuples_s / max_freq_s)
    bound = min(distinct_r, distinct_s) * max_freq_r * max_freq_s

    print_stderr("UES bounds:", condition=verbose)
    print_stderr(f"|R| = {num_tuples_r}; MF(R.a) = {max_freq_r}", condition=verbose)
    print_stderr(f"|S| = {num_tuples_s}; MF(S.b) = {max_freq_s}", condition=verbose)
    print_stderr(f"distinct(R.a) = {distinct_r}; distinct(S.b) = {distinct_s}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return round(bound)


def calculate_topk_bound_v1(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    # Top-k bound
    num_proc_r, num_proc_s = 0, 0
    topk_bound = 0

    for attr_val in topk_r:
        r_freq = topk_r[attr_val]
        s_freq = topk_s[attr_val]
        topk_bound += r_freq * s_freq
        num_proc_r += r_freq
        num_proc_s += s_freq

    for attr_val in [attr_val for attr_val in topk_s if attr_val not in topk_r]:
        r_freq = topk_r[attr_val]
        s_freq = topk_s[attr_val]
        topk_bound += r_freq * s_freq
        num_proc_r += r_freq
        num_proc_s += s_freq

    adjust_r, adjust_s = min(num_tuples_r / num_proc_r, 1), min(num_tuples_s / num_proc_s, 1)
    topk_bound = adjust_r * adjust_s * topk_bound

    hits_r = len(topk_r) + len(topk_s.attribute_values() - topk_r.attribute_values())
    hits_s = len(topk_s) + len(topk_r.attribute_values() - topk_s.attribute_values())

    # remainder UES bound
    distinct_rem_r = max((num_tuples_r - hits_r * topk_r.star_freq()) / topk_r.star_freq(), 0)
    distinct_rem_s = max((num_tuples_s - hits_s * topk_s.star_freq()) / topk_s.star_freq(), 0)
    ues_bound = min(distinct_rem_r, distinct_rem_s) * topk_r.star_freq() * topk_s.star_freq()

    return round(topk_bound + ues_bound)


def calculate_topk_bound_v2(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    # Top-k bound
    num_proc_r, num_proc_s = 0, 0
    topk_bound = 0

    for attr_val in topk_r:
        r_freq = topk_r[attr_val]
        s_freq = topk_s[attr_val]
        topk_bound += r_freq * s_freq

        # bookkeeping
        num_proc_r += r_freq
        num_proc_s += s_freq

    for attr_val in [attr_val for attr_val in topk_s if attr_val not in topk_r]:
        r_freq = topk_r[attr_val]
        s_freq = topk_s[attr_val]
        topk_bound += r_freq * s_freq

        # bookkeeping
        num_proc_r += r_freq
        num_proc_s += s_freq

    adjust_r, adjust_s = min(num_tuples_r / num_proc_r, 1), min(num_tuples_s / num_proc_s, 1)
    topk_bound = adjust_r * adjust_s * topk_bound

    hits_r = len(topk_r) + len(topk_s.attribute_values() - topk_r.attribute_values())
    hits_s = len(topk_s) + len(topk_r.attribute_values() - topk_s.attribute_values())

    # remainder UES bound
    distinct_rem_r = max(math.ceil(num_tuples_r / topk_r.star_freq() - hits_r), 0)
    distinct_rem_s = max(math.ceil(num_tuples_s / topk_s.star_freq() - hits_s), 0)
    ues_bound = min(distinct_rem_r, distinct_rem_s) * topk_r.star_freq() * topk_s.star_freq()

    print_stderr(f"hits(R.a) = {hits_r}; hits(S.b) = {hits_s}", condition=verbose)
    print_stderr(f"f*(R.a) = {topk_r.star_freq()}; f*(S.b) = {topk_s.star_freq()}", condition=verbose)
    print_stderr(f"distinct(R.a') = {distinct_rem_r}; distinct(S.b') = {distinct_rem_s}", condition=verbose)
    print_stderr(f"Top-k bound: {topk_bound}; UES* bound: {ues_bound}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return round(topk_bound + ues_bound)


def calcualte_topk_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                         version: int = 2, verbose: bool = False):
    print_stderr(f"Top-k bound v.{version}", condition=verbose)
    if version == 1:
        bound = calculate_topk_bound_v1(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    elif version == 2:
        bound = calculate_topk_bound_v2(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)

    return bound


def execute_join(tuples_r: List[str], tuples_s: List[str]) -> int:
    output_cardinality = 0
    occurrences_r = count_value_occurrences(tuples_r)
    occurrences_s = count_value_occurrences(tuples_s)

    for value, count_r in occurrences_r.items():
        count_s = occurrences_s.get(value, 0)
        output_cardinality += count_r * count_s

    return output_cardinality


SimulationResult = collections.namedtuple("SimulationResult", ["ues", "topk", "actual"])


def simulate(num_tuples_r: int, num_tuples_s: int, distinct_values_r: int, distinct_values_s: int, *,
             topk_length: int, exceed_value: int, version: int, verbose: bool, rand_seed: int):
    random.seed(rand_seed)

    tuples_r = generate_tuples(num_tuples_r, distinct_values_r)
    tuples_s = generate_tuples(num_tuples_s, distinct_values_s)
    topk_r = build_topk_list(tuples_r, k=topk_length, exceed_value=exceed_value)
    topk_s = build_topk_list(tuples_s, k=topk_length, exceed_value=exceed_value)

    print_stderr(".. Using exceed value of", exceed_value, condition=verbose and (exceed_value is not None))

    print_stderr("|R| =", num_tuples_r, condition=verbose)
    print_stderr("R.a:", sorted(tuples_r), condition=verbose)
    print_stderr("topk(R.a) =", topk_r, condition=verbose)
    print_stderr(condition=verbose)
    print_stderr("|S| = ", num_tuples_s, condition=verbose)
    print_stderr("S.b:", sorted(tuples_s), condition=verbose)
    print_stderr("topk(S.b) =", topk_s, condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    ues_bound = calculate_ues_bound(max_frequency(tuples_r), max_frequency(tuples_s), num_tuples_r, num_tuples_s,
                                    verbose=verbose)
    topk_bound = calcualte_topk_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, version=version, verbose=verbose)
    actual_values = execute_join(tuples_r, tuples_s)

    return SimulationResult(ues_bound, topk_bound, actual_values)


def find_regressions(num_tuples_r: int, num_tuples_s: int, distinct_values_r: int, distinct_values_s: int, *,
                     topk_length: int, exceed_value: int, version: int, num_seeds: int = 100):
    for seed in range(num_seeds):
        simulation_res = simulate(num_tuples_r, num_tuples_s, distinct_values_r, distinct_values_s,
                                  topk_length=topk_length, exceed_value=exceed_value, rand_seed=seed,
                                  version=version, verbose=False)
        if simulation_res.ues < simulation_res.topk:
            print("Regression found at seed ", seed, ": UES bound is smaller Top-k bound", sep="")
        elif simulation_res.topk < simulation_res.actual:
            print("Regression found at seed ", seed, ": Top-k bound is no upper bound", sep="")


def main():
    parser = argparse.ArgumentParser(description="Utility to quickly test the behaviour of different Top-K estimation"
                                     "algorithms")
    parser.add_argument("-n", action="store", type=int, required=False, default=10,
                        help="Number of tuples per relation")
    parser.add_argument("-k", action="store", type=int, default=3, help="Length of the Top-k lists")
    parser.add_argument("-d", action="store", type=int, required=False, default=5,
                        help="Number of distinct values per relation")
    parser.add_argument("-nr", action="store", type=int, required=False, help="Number of tuples in the R relation. "
                        "Overwrites -n.")
    parser.add_argument("-ns", action="store", type=int, required=False, help="Number of tuples in the S relation. "
                        "Overwrites -n.")
    parser.add_argument("-dr", action="store", type=int, required=False, help="Number of distinct values of the R "
                        "relation. Overwrites -d.")
    parser.add_argument("-ds", action="store", type=int, required=False, help="Number of distinct values of the S "
                        "relation. Overwrites -d.")
    parser.add_argument("--exceed", action="store", type=int, default=0, help="Frequency oversubsription. Each "
                        "attribute frequency will be increase by a  value in [0, exceed], which can create "
                        "frequencies that exceed the total number of tuples per relation.")
    parser.add_argument("--rand-seed", action="store", type=int, default=42, help="Seed for the RNG")
    parser.add_argument("--regression-mode", action="store_true", default=False)
    parser.add_argument("--version", action="store", type=int, default=2, help="")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Produce debugging ouput")
    # TODO: Primary Key/Foreign Key joins

    args = parser.parse_args()

    if not args.n and (not args.nr or not args.ns):
        parser.error("If -n is not given, both -nr and -ns must be specified")
    if not args.d and (not args.dr or not args.ds):
        parser.error("If -d is not given, both -dr and -ds must be specified")

    num_tuples_r = args.nr if args.nr else args.n
    num_tuples_s = args.ns if args.ns else args.n
    distinct_values_r = args.dr if args.dr else args.d
    distinct_values_s = args.ds if args.ds else args.d
    exceed_value = args.exceed if args.exceed else None
    topk_length = args.k
    version = args.version if args.version else 1

    if distinct_values_r > len(attr_values) or distinct_values_s > len(attr_values):
        parser.error(f"Maximum {len(attr_values)} distinct attribute values allowed")

    if args.regression_mode:
        find_regressions(num_tuples_r, num_tuples_s, distinct_values_r, distinct_values_s,
                         topk_length=topk_length, exceed_value=exceed_value, version=version)
        return

    simulation_result = simulate(num_tuples_r, num_tuples_s, distinct_values_r, distinct_values_s,
                                 topk_length=topk_length, exceed_value=exceed_value, rand_seed=args.rand_seed,
                                 version=version, verbose=args.verbose)
    ues_bound, topk_bound, actual_values = simulation_result
    print("UES bound:", ues_bound)
    print("MCV bound:", topk_bound)
    print("Actual:", actual_values)

    ues_bound_failure = ues_bound < actual_values
    topk_bound_failure = (topk_bound < actual_values) or (topk_bound > ues_bound)
    sys.exit(1 if ues_bound_failure or topk_bound_failure else 0)


if __name__ == "__main__":
    main()
