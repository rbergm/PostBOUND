#!/usr/bin/env python3

import abc
import argparse
import collections
import math
import itertools
import operator
import pprint
import random
import sys
from typing import Dict, List, Set, Iterator, Tuple, Union


attr_values = "abcdefghijklmnopqrstuvwxyz"

DEFAULT_TOPK_ESTIMATION_VER = 1
DEFAULT_NUM_REGRESSION_SEEDS = 100


def print_stderr(*args, condition: bool = None, **kwargs):
    if condition is False:
        return
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


class TopKList:
    @staticmethod
    def empty() -> "TopKList":
        return TopKList([])

    def __init__(self, value_frequencies: List[Tuple[str, int]], *, k: int = 5,
                 total_num_tuples: int = None, smart_fstar: bool = False):
        sorted_frequencies = sorted(value_frequencies, key=operator.itemgetter(1), reverse=True)

        self.k = k
        self.total_num_tuples = total_num_tuples
        self.entries = sorted_frequencies[:self.k]
        self.topk_list = dict(self.entries)
        self.remainder_frequency = (0 if smart_fstar and k >= len(value_frequencies)
                                    else min(self.topk_list.values(), default=0))

    def max_freq(self) -> int:
        return max(self.topk_list.values(), default=0)

    def star_freq(self) -> int:
        return self.remainder_frequency

    def total_frequency(self) -> int:
        return sum(self.topk_list.values())

    def min_value(self) -> str:
        return self.entries[-1][0]

    def is_uniform(self) -> bool:
        return self.max_freq() == self.star_freq()

    def attribute_values(self) -> Set[str]:
        return set(self.topk_list.keys())

    def overlaps_with(self, other: "TopKList") -> bool:
        return len(self.attribute_values() & other.attribute_values()) > 0

    def drop_values_from(self, other: "TopKList") -> "TopKList":
        unique_values = [(val, freq) for val, freq in self.entries if val not in other]
        return TopKList(unique_values, k=self.k)

    def snap_frequencies_to(self, max_frequency: int) -> "TopKList":
        return TopKList([(val, min(freq, max_frequency)) for val, freq in self.entries], k=self.k)

    def __contains__(self, attr: str) -> bool:
        return attr in self.topk_list

    def __getitem__(self, attr: str) -> int:
        return self.topk_list.get(attr, self.star_freq())

    def __iter__(self) -> Iterator[str]:
        return [val for val, __ in self.entries].__iter__()

    def __len__(self) -> int:
        return len(self.entries)

    def __sub__(self, other: "TopKList") -> "TopKList":
        return self.drop_values_from(other)

    def __str__(self) -> str:
        return f"{str(self.entries)}, f* = {self.remainder_frequency}"


class RelationGenerator(abc.ABC):
    @abc.abstractmethod
    def num_tuples(self) -> int:
        return NotImplemented

    @abc.abstractmethod
    def relation_contents(self) -> List[str]:
        return NotImplemented

    @abc.abstractmethod
    def reset(self) -> None:
        return NotImplemented


class RandomRelationGenerator(RelationGenerator):
    def __init__(self, num_tuples: int, distinct_values: int):
        self._num_tuples = num_tuples
        self._distinct_values = distinct_values
        self._contents = None

    def num_tuples(self) -> int:
        return self._num_tuples

    def relation_contents(self) -> List[str]:
        if self._contents is None:
            if self._distinct_values <= len(attr_values):
                available_values = attr_values[:self._distinct_values]
            else:
                num_usable_values = self._largest_divisor(self._distinct_values, len(attr_values))
                base_values = attr_values[:num_usable_values]
                subvalues_per_base_value = int(self._distinct_values / num_usable_values)
                available_values = [base + str(suffix) for base, suffix in
                                    itertools.product(base_values, range(subvalues_per_base_value))]
            weights = [random.randint(0, 10) for __ in range(self._distinct_values)]
            self._contents = random.choices(available_values, weights, k=self._num_tuples)

        return list(self._contents)

    def reset(self) -> None:
        self._contents = None

    def _largest_divisor(self, value: int, max_candidate: int) -> int:
        for candidate in reversed(range(1, max_candidate+1)):
            if value % candidate == 0:
                return candidate


class ManualRelationGenerator(RelationGenerator):
    def __init__(self, contents: str):
        self._contents = contents.split(",")

    def num_tuples(self) -> int:
        return len(self._contents)

    def relation_contents(self) -> List[str]:
        return list(self._contents)

    def reset(self) -> None:
        pass


class TopKListGenerator(abc.ABC):
    @abc.abstractmethod
    def topk_length(self) -> int:
        return NotImplemented

    @abc.abstractmethod
    def build_topk_list(self, tuples: List[str], k: int) -> TopKList:
        return NotImplemented


class DistributionBasedTopKListGenerator(TopKListGenerator):
    def __init__(self, *, exceed_value: int = 0):
        self._exceed_value = exceed_value

    def topk_length(self) -> int:
        return self._k

    def build_topk_list(self, tuples: List[str], k: int) -> TopKList:
        exceed_value = self._exceed_value if self._exceed_value else 0
        occurrence_counter = count_value_occurrences(tuples)

        attribute_frequencies = [(attr, freq) for attr, freq in occurrence_counter.items()]
        attribute_frequencies = [(attr, freq + random.randint(0, exceed_value))
                                 for attr, freq in attribute_frequencies]
        return TopKList(attribute_frequencies, k=k)


class ManualTopKListGenerator(TopKListGenerator):
    def __init__(self, contents: str):
        self._contents = contents

    def topk_length(self) -> int:
        return len(self._contents.split(","))

    def build_topk_list(self, tuples: List[str], k: int) -> TopKList:
        topk_entries: Dict[str, int] = {}
        for entry in self._contents.split(","):
            value, frequency = entry.split(":")
            topk_entries[value] = int(frequency)
        entry_list: List[Tuple[str, int]] = list(topk_entries.items())
        return TopKList(entry_list, k=k)


def count_value_occurrences(attribute_values: List[str]) -> Dict[str, int]:
    occurrence_counter = collections.defaultdict(int)
    for val in attribute_values:
        occurrence_counter[val] += 1
    return occurrence_counter


def max_frequency(attribute_values: List[str]) -> int:
    occurrence_counter = count_value_occurrences(attribute_values)
    return max(occurrence_counter.values())


def histogram(attribute_values: List[str]) -> List[Tuple[str, int]]:
    occurrences = count_value_occurrences(attribute_values)
    hist = sorted(occurrences.items())
    return hist


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
    fstar_hits_r, fstar_hits_s = 0, 0
    topk_bound = 0

    for attr_val in topk_r:
        r_freq = topk_r[attr_val]
        s_freq = topk_s[attr_val]
        topk_bound += r_freq * s_freq

        # bookkeeping
        num_proc_r += r_freq
        num_proc_s += s_freq
        fstar_hits_s += 1 if attr_val not in topk_s else 0

    for attr_val in [attr_val for attr_val in topk_s if attr_val not in topk_r]:
        r_freq = topk_r[attr_val]
        s_freq = topk_s[attr_val]
        topk_bound += r_freq * s_freq

        # bookkeeping
        num_proc_r += r_freq
        num_proc_s += s_freq
        fstar_hits_r += 1

    adjust_r, adjust_s = min(num_tuples_r / num_proc_r, 1), min(num_tuples_s / num_proc_s, 1)
    topk_bound = adjust_r * adjust_s * topk_bound

    # remainder UES bound
    num_rem_tuples_r = max(num_tuples_r - num_proc_r, 0)
    num_rem_tuples_s = max(num_tuples_s - num_proc_s, 0)
    distinct_rem_r = num_rem_tuples_r / topk_r.star_freq()
    distinct_rem_s = num_rem_tuples_s / topk_s.star_freq()
    ues_bound = min(distinct_rem_r, distinct_rem_s) * topk_r.star_freq() * topk_s.star_freq()

    print_stderr(f"hits*(R.a) = {fstar_hits_r}; hits*(S.b) = {fstar_hits_s}", condition=verbose)
    print_stderr(f"f*(R.a) = {topk_r.star_freq()}; f*(S.b) = {topk_s.star_freq()}", condition=verbose)
    print_stderr(f"|R'| = {num_rem_tuples_r}; |S'| = {num_rem_tuples_s}", condition=verbose)
    print_stderr(f"distinct(R.a') = {distinct_rem_r}; distinct(S.b') = {distinct_rem_s}", condition=verbose)
    print_stderr(f"Top-k bound: {topk_bound}; UES* bound: {ues_bound}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return round(topk_bound + ues_bound)


def calculate_topk_bound_v2(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    def topk_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        topk_bound = 0
        processed_tuples_r, processed_tuples_s = 0, 0

        for attribute_value in topk_r:
            topk_bound += topk_r[attribute_value] * topk_s[attribute_value]
            processed_tuples_r += topk_r[attribute_value]
            if attribute_value in topk_s:
                processed_tuples_s += topk_s[attribute_value]

        for attribute_value in topk_s - topk_r:
            topk_bound += topk_s[attribute_value] * topk_r.star_freq()
            processed_tuples_s += topk_s[attribute_value]

        adjustment_factor_r = min(num_tuples_r / processed_tuples_r, 1)
        adjustment_factor_s = min(num_tuples_s / processed_tuples_s, 1)
        adjusted_topk_bound = adjustment_factor_r * adjustment_factor_s * topk_bound

        print_stderr(f"Top-k adjust: a(R) = {adjustment_factor_r}; a(S) = {adjustment_factor_s}; "
                     f"original bound: {topk_bound}", condition=verbose)
        print_stderr(f"Top-k processed tuples: p(R) = {processed_tuples_r}; "
                     f"p(S) = {processed_tuples_s}", condition=verbose)

        return adjusted_topk_bound

    def ues_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        if topk_r.star_freq() == 0 or topk_s.star_freq() == 0:
            return 0

        topk_hits = len(topk_r.attribute_values() | topk_s.attribute_values())
        distinct_values_r = max(num_tuples_r / topk_r.star_freq() - topk_hits, 0)
        distinct_values_s = max(num_tuples_s / topk_s.star_freq() - topk_hits, 0)
        ues_bound = min(distinct_values_r, distinct_values_s) * topk_r.star_freq() * topk_s.star_freq()

        print_stderr(f"f*(R.a) = {topk_r.star_freq()}; f*(S.b) = {topk_s.star_freq()}", condition=verbose)
        print_stderr(f"distinct(R.a') = {distinct_values_r}; distinct(S.b') = {distinct_values_s}", condition=verbose)
        return ues_bound

    _topk_bound = topk_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    _ues_bound = ues_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    print_stderr(f"Top-k bound: {_topk_bound}; UES* bound: {_ues_bound}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return math.ceil(_topk_bound + _ues_bound)


def calculate_topk_bound_v3(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    def topk_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        topk_bound = 0
        processed_tuples_r, processed_tuples_s = 0, 0

        for attribute_value in topk_r:
            topk_bound += topk_r[attribute_value] * topk_s[attribute_value]
            processed_tuples_r += topk_r[attribute_value]
            if attribute_value in topk_s:
                processed_tuples_s += topk_s[attribute_value]

        for attribute_value in topk_s - topk_r:
            topk_bound += topk_s[attribute_value] * topk_r.star_freq()
            processed_tuples_s += topk_s[attribute_value]

        adjustment_factor_r = min(num_tuples_r / processed_tuples_r, 1)
        adjustment_factor_s = min(num_tuples_s / processed_tuples_s, 1)
        adjusted_topk_bound = adjustment_factor_r * adjustment_factor_s * topk_bound

        print_stderr(f"Top-k adjust: a(R) = {adjustment_factor_r}; a(S) = {adjustment_factor_s}; "
                     f"original bound: {topk_bound}", condition=verbose)
        print_stderr(f"Top-k processed tuples: p(R) = {processed_tuples_r}; "
                     f"p(S) = {processed_tuples_s}", condition=verbose)

        return adjusted_topk_bound

    def ues_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        if topk_r.star_freq() == 0 or topk_s.star_freq() == 0:
            return 0

        distinct_values_r, distinct_values_s = num_tuples_r / topk_r.star_freq(), num_tuples_s / topk_s.star_freq()
        ues_bound = min(distinct_values_r, distinct_values_s) * topk_r.star_freq() * topk_s.star_freq()

        print_stderr(f"f*(R.a) = {topk_r.star_freq()}; f*(S.b) = {topk_s.star_freq()}", condition=verbose)
        print_stderr(f"distinct(R.a') = {distinct_values_r}; distinct(S.b') = {distinct_values_s}", condition=verbose)
        return ues_bound

    _topk_bound = topk_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    _ues_bound = ues_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    print_stderr(f"Top-k bound: {_topk_bound}; UES* bound: {_ues_bound}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return math.ceil(_topk_bound + _ues_bound)


def calculate_topk_bound_v4(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    # Top-k bound
    num_proc_r, num_proc_s = 0, 0
    topk_bound = 0

    if not topk_r.is_uniform():
        for attr_val in topk_r:
            r_freq = topk_r[attr_val]
            s_freq = topk_s[attr_val]
            topk_bound += r_freq * s_freq

            # bookkeeping
            num_proc_r += r_freq
            num_proc_s += s_freq

    unique_topk_s = TopKList.empty() if topk_r.is_uniform() else topk_s.drop_values_from(topk_r)
    for attr_val in unique_topk_s:
        r_freq = topk_r[attr_val]
        s_freq = topk_s[attr_val]
        topk_bound += r_freq * s_freq

        # bookkeeping
        num_proc_r += r_freq
        num_proc_s += s_freq

    adjust_r = min(num_tuples_r / num_proc_r, 1) if num_proc_r else 1
    adjust_s = min(num_tuples_s / num_proc_s, 1) if num_proc_s else 1
    topk_bound = adjust_r * adjust_s * topk_bound

    # remainder UES bound
    rem_freq_r = topk_r.star_freq()
    rem_freq_s = topk_s.max_freq() if topk_r.is_uniform() else topk_s.star_freq()
    distinct_rem_r = num_tuples_r / rem_freq_r
    distinct_rem_s = num_tuples_s / rem_freq_s
    ues_bound = min(distinct_rem_r, distinct_rem_s) * topk_r.star_freq() * rem_freq_s

    print_stderr(f"f*(R.a) = {rem_freq_r}; f*(S.b) = {rem_freq_s}", condition=verbose)
    print_stderr(f"distinct(R.a') = {distinct_rem_r}; distinct(S.b') = {distinct_rem_s}", condition=verbose)
    print_stderr(f"Top-k bound: {topk_bound}; UES* bound: {ues_bound}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return round(topk_bound + ues_bound)


def calculate_topk_bound_v5(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):

    def topk_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        topk_bound = 0
        processed_tuples_r, processed_tuples_s = 0, 0

        for attribute_value in topk_r:
            topk_bound += topk_r[attribute_value] * topk_s[attribute_value]
            processed_tuples_r += topk_r[attribute_value]
            if attribute_value in topk_s:
                processed_tuples_s += topk_s[attribute_value]

        for attribute_value in topk_s - topk_r:
            topk_bound += topk_s[attribute_value] * topk_r.star_freq()
            processed_tuples_s += topk_s[attribute_value]

        adjustment_factor_r = min(num_tuples_r / processed_tuples_r, 1)
        adjustment_factor_s = min(num_tuples_s / processed_tuples_s, 1)
        adjusted_topk_bound = adjustment_factor_r * adjustment_factor_s * topk_bound

        print_stderr(f"Top-k adjust: a(R) = {adjustment_factor_r}; a(S) = {adjustment_factor_s}; "
                     f"original bound: {topk_bound}", condition=verbose)
        print_stderr(f"Top-k processed tuples: p(R) = {processed_tuples_r}; "
                     f"p(S) = {processed_tuples_s}", condition=verbose)

        return adjusted_topk_bound

    def ues_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        if topk_r.star_freq() == 0 or topk_s.star_freq() == 0:
            return 0

        effective_num_tuples_r = max(num_tuples_r - len(topk_r) * topk_r.star_freq(), 0)
        effective_num_tuples_s = max(num_tuples_s - len(topk_s) * topk_s.star_freq(), 0)

        distinct_values_r = effective_num_tuples_r / topk_r.star_freq()
        distinct_values_s = effective_num_tuples_s / topk_s.star_freq()
        ues_bound = min(distinct_values_r, distinct_values_s) * topk_r.star_freq() * topk_s.star_freq()

        print_stderr(f"|R*| = {effective_num_tuples_r}; |S*| = {effective_num_tuples_s}", condition=verbose)
        print_stderr(f"f*(R.a) = {topk_r.star_freq()}; f*(S.b) = {topk_s.star_freq()}", condition=verbose)
        print_stderr(f"distinct(R.a') = {distinct_values_r}; distinct(S.b') = {distinct_values_s}", condition=verbose)
        return ues_bound

    _topk_bound = topk_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    _ues_bound = ues_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    print_stderr(f"Top-k bound: {_topk_bound}; UES* bound: {_ues_bound}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return math.ceil(_topk_bound + _ues_bound)


def calculate_topk_bound_v6(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):

    def topk_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        topk_bound = 0
        processed_tuples_r, processed_tuples_s = 0, 0

        for attribute_value in topk_r:
            topk_bound += topk_r[attribute_value] * topk_s[attribute_value]
            processed_tuples_r += topk_r[attribute_value]
            if attribute_value in topk_s:
                processed_tuples_s += topk_s[attribute_value]

        for attribute_value in topk_s - topk_r:
            topk_bound += topk_s[attribute_value] * topk_r.star_freq()
            processed_tuples_s += topk_s[attribute_value]

        adjustment_factor_r = min(num_tuples_r / processed_tuples_r, 1)
        adjustment_factor_s = min(num_tuples_s / processed_tuples_s, 1)
        adjusted_topk_bound = min(adjustment_factor_r, adjustment_factor_s) * topk_bound

        print_stderr(f"Top-k adjust: a(R) = {adjustment_factor_r}; a(S) = {adjustment_factor_s}; "
                     f"original bound: {topk_bound}", condition=verbose)
        print_stderr(f"Top-k processed tuples: p(R) = {processed_tuples_r}; "
                     f"p(S) = {processed_tuples_s}", condition=verbose)

        return adjusted_topk_bound

    def ues_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *, verbose: bool = False):
        if topk_r.star_freq() == 0 or topk_s.star_freq() == 0:
            return 0

        distinct_values_r, distinct_values_s = num_tuples_r / topk_r.star_freq(), num_tuples_s / topk_s.star_freq()
        ues_bound = min(distinct_values_r, distinct_values_s) * topk_r.star_freq() * topk_s.star_freq()

        print_stderr(f"f*(R.a) = {topk_r.star_freq()}; f*(S.b) = {topk_s.star_freq()}", condition=verbose)
        print_stderr(f"distinct(R.a') = {distinct_values_r}; distinct(S.b') = {distinct_values_s}", condition=verbose)
        return ues_bound

    topk_r, topk_s = topk_r.snap_frequencies_to(num_tuples_r), topk_s.snap_frequencies_to(num_tuples_s)
    _topk_bound = topk_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    _ues_bound = ues_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    print_stderr(f"Top-k bound: {_topk_bound}; UES* bound: {_ues_bound}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    return math.ceil(_topk_bound + _ues_bound)


class TopKListV7:
    @staticmethod
    def initialize_from(base_topk: TopKList, num_tuples: int) -> "TopKListV7":
        return TopKListV7(base_topk.entries, num_tuples)

    def __init__(self, entries: List[Tuple[str, int]], num_tuples: int, *, star_frequency: Union[int, None] = None):
        self._entries = sorted(entries, key=operator.itemgetter(1), reverse=True) if num_tuples > 0 else []
        self._frequencies = dict(self._entries)
        self._total_tuples = num_tuples
        self._star_frequency = (min(star_frequency, num_tuples) if star_frequency is not None and num_tuples > 0
                                else star_frequency)
        if self._entries and self._star_frequency is None:
            self._star_frequency = self._entries[-1][1]
        elif self._entries and self._star_frequency is not None:
            self._star_frequency = min(self._star_frequency, self._entries[-1][1])
        elif not self._entries and self._star_frequency is None:
            self._star_frequency = 1

    def has_contents(self) -> bool:
        return len(self._entries) > 0

    def is_empty(self) -> bool:
        return not self.has_contents()

    def attribute_values(self) -> Set[str]:
        return set(self._frequencies.keys())

    def frequency_of(self, value: str) -> int:
        return self._frequencies.get(value, self.star_frequency)

    def head(self) -> Tuple[str, int]:
        return self._entries[0] if self._total_tuples > 0 else None

    def snap_to(self, num_tuples: int) -> "TopKListV7":
        snapped_tuples = min(self._total_tuples, num_tuples)
        snapped_entries = [(val, min(freq, num_tuples)) for val, freq in self._entries]
        snapped_star_freq = min(self.star_frequency, num_tuples)
        return TopKListV7(snapped_entries, snapped_tuples, star_frequency=snapped_star_freq)

    def shift_according_to(self, value: str) -> "TopKListV7":
        corresponding_frequency = self.frequency_of(value)
        remaining_tuples = max(self._total_tuples - corresponding_frequency, 0)
        remaining_entries = [(val, freq) for val, freq in self._entries if val != value]
        shifted_list = TopKListV7(remaining_entries, remaining_tuples, star_frequency=self.star_frequency)
        return shifted_list.snap_to(remaining_tuples)

    def _get_num_tuples(self) -> int:
        return self._total_tuples

    def _set_num_tuples(self, value: int) -> None:
        self._total_tuples = max(value, 0)

    def _get_star_frequency(self) -> int:
        return self._star_frequency

    remaining_tuples = property(_get_num_tuples, _set_num_tuples)
    star_frequency = property(_get_star_frequency)

    def __getitem__(self, key: str) -> int:
        return self.frequency_of(key)

    def __iter__(self) -> Iterator[str]:
        return [val for val, __ in self._entries].__iter__()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"card = {self._total_tuples}; MCV = {str(self._entries)}; f* = {self.star_frequency}"


def calculate_topk_bound_v7(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    topk_r: TopKListV7 = TopKListV7.initialize_from(topk_r, num_tuples_r)
    topk_s: TopKListV7 = TopKListV7.initialize_from(topk_s, num_tuples_s)
    topk_r = topk_r.snap_to(num_tuples_r)
    topk_s = topk_s.snap_to(num_tuples_s)

    print_stderr(f"Original MCV(R): {topk_r}", condition=verbose)
    print_stderr(f"Original MCV(S): {topk_s}", condition=verbose)

    bound = 0
    while ((topk_r.has_contents() or topk_s.has_contents())
           and topk_r.remaining_tuples > 0
           and topk_s.remaining_tuples > 0):
        highest_bound = 0
        highest_bound_value = None
        for attr_val in topk_r:
            candidate_bound = topk_r[attr_val] * topk_s[attr_val]
            if candidate_bound > highest_bound:
                highest_bound = candidate_bound
                highest_bound_value = attr_val
        for attr_val in topk_s:
            candidate_bound = topk_r[attr_val] * topk_s[attr_val]
            if candidate_bound > highest_bound:
                highest_bound = candidate_bound
                highest_bound_value = attr_val

        bound += highest_bound

        topk_r = topk_r.shift_according_to(highest_bound_value)
        topk_s = topk_s.shift_according_to(highest_bound_value)

        print_stderr(condition=verbose)
        print_stderr(".. Next iteration", condition=verbose)
        print_stderr(f".. Selected value {highest_bound_value} (bound = {highest_bound})", condition=verbose)
        print_stderr(f".. Adjusted MCV(R): {topk_r}", condition=verbose)
        print_stderr(f".. Adjusted MCV(S): {topk_s}", condition=verbose)

    print_stderr(f"|R*| = {topk_r.remaining_tuples}; |S*| = {topk_s.remaining_tuples}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    if topk_r.remaining_tuples > 0 and topk_s.remaining_tuples > 0:
        bound += (min(topk_r.remaining_tuples / topk_r.star_frequency, topk_s.remaining_tuples / topk_s.star_frequency)
                  * topk_r.star_frequency
                  * topk_s.star_frequency)

    return math.ceil(bound)


def calculate_topk_bound_v8(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    shorter_topk = topk_r if len(topk_r) < len(topk_s) else topk_s

    bound = 0
    for topk_idx in range(len(shorter_topk)):
        __, freq_r = topk_r.entries[topk_idx]
        __, freq_s = topk_s.entries[topk_idx]

        partial_bound = freq_r * freq_s
        if topk_idx == len(shorter_topk) - 1:
            partial_bound *= max(num_tuples_r / topk_r.star_freq(), num_tuples_s / topk_s.star_freq())
        bound += partial_bound

    return bound


def calculate_topk_bound_v9(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                            verbose: bool = False):
    bound = 0
    for val in topk_r:
        bound += topk_r[val] * topk_s[val]
    for val in topk_s.drop_values_from(topk_r):
        bound += topk_r[val] * topk_s[val]

    distinct_topk_values = len(topk_r.attribute_values() | topk_s.attribute_values())
    ues_distinct_values = min(num_tuples_r / topk_r.star_freq(), num_tuples_s / topk_s.star_freq())
    remaining_distinct_values = max(ues_distinct_values - distinct_topk_values, 0)
    bound += remaining_distinct_values * topk_r.star_freq() * topk_s.star_freq()

    return bound


def calculate_topk_bound_v10(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                             verbose: bool = False):
    def calculate_max_bound(topk_r: TopKListV7, topk_s: TopKListV7, *, current_bound: int = 0,
                            initial: bool = False) -> int:
        if topk_r.remaining_tuples == 0 or topk_s.remaining_tuples == 0:
            # if there are no tuples left, there is nothing more to do
            return current_bound

        if topk_r.is_empty() and topk_s.is_empty():
            # if there are still tuples left but our Top-K lists don't contain any more information, fall back to
            # a UES estimation
            distinct_values = min(topk_r.remaining_tuples / topk_r.star_frequency,
                                  topk_s.remaining_tuples / topk_s.star_frequency)
            remainder_bound = distinct_values * topk_r.star_frequency * topk_s.star_frequency
            return current_bound + remainder_bound

        max_bound = 0
        for value in topk_r.attribute_values() | topk_s.attribute_values():
            # otherwise compute for each remaining value the maximum bound that can originate from using it as the
            # next join value
            join_bound = current_bound + topk_r[value] * topk_s[value]
            candidate_bound = calculate_max_bound(topk_r.shift_according_to(value),
                                                  topk_s.shift_according_to(value),
                                                  current_bound=join_bound)
            if candidate_bound > max_bound:
                max_bound = candidate_bound
                print_stderr(f"New max bound {max_bound} based on value {value}", condition=verbose and initial)
        return max_bound

    return calculate_max_bound(TopKListV7.initialize_from(topk_r, num_tuples_r).snap_to(num_tuples_r),
                               TopKListV7.initialize_from(topk_s, num_tuples_s).snap_to(num_tuples_s),
                               initial=True)


TopkBoundVersions = {
    1: (calculate_topk_bound_v1, "The former (live) implementation"),
    2: (calculate_topk_bound_v2, "UES* pushdown: reduce distinct values based on TopK lists"),
    3: (calculate_topk_bound_v3, "No pushdown: adjusted Topk bound + full UES* bound"),
    4: (calculate_topk_bound_v4, "Short-lived experiment. Doesn't work"),
    5: (calculate_topk_bound_v5, "UES* pushdown: |R*| := |R| - f* * k"),
    6: (calculate_topk_bound_v6, "Mild TopK adjustment, no UES* pushdown"),
    7: (calculate_topk_bound_v7, "Integrated bound with inplace updates"),
    8: (calculate_topk_bound_v8, "Value-independent estimation"),
    9: (calculate_topk_bound_v9, "No TopK pushdown, UES distinct value pushdown"),
    10: (calculate_topk_bound_v10, "Greedy integrated bound with inplace updates")
}


def calcualte_topk_bound(topk_r: TopKList, topk_s: TopKList, num_tuples_r: int, num_tuples_s: int, *,
                         version: int = DEFAULT_TOPK_ESTIMATION_VER, verbose: bool = False):
    print_stderr(f"Top-k bound v.{version}", condition=verbose)
    bound = -math.inf
    if version in TopkBoundVersions:
        bound_calculator = TopkBoundVersions[version][0]
        bound = bound_calculator(topk_r, topk_s, num_tuples_r, num_tuples_s, verbose=verbose)
    else:
        print("Available versions are:")
        pprint.pprint({num: version[1] for num, version in TopkBoundVersions.items()})
        raise ValueError(f"Unknown version: {version}")

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


def simulate(generator_r: RelationGenerator, generator_s: RelationGenerator,
             topk_generator_r: TopKListGenerator, topk_generator_s: TopKListGenerator, *,
             topk_length: int, exceed_value: int, topk_version: int, verbose: bool,
             rand_seed: int) -> SimulationResult:
    random.seed(rand_seed)
    generator_r.reset()
    generator_s.reset()

    tuples_r, tuples_s = generator_r.relation_contents(), generator_s.relation_contents()

    num_tuples_r, num_tuples_s = generator_r.num_tuples(), generator_s.num_tuples()
    topk_r = topk_generator_r.build_topk_list(tuples_r, topk_length)
    topk_s = topk_generator_s.build_topk_list(tuples_s, topk_length)

    print_stderr(".. Using exceed value of", exceed_value, condition=verbose and (exceed_value is not None))

    print_stderr(f"|R| = {num_tuples_r}; distinct(R.a) = {len(set(tuples_r))}", condition=verbose)
    print_stderr("R.a:", sorted(tuples_r), condition=verbose and len(tuples_r) <= 30)
    print_stderr("topk(R.a) =", topk_r, condition=verbose)
    print_stderr(f"hist(R.a) = {histogram(tuples_r)}", condition=verbose)
    print_stderr(condition=verbose)
    print_stderr(f"|S| = {num_tuples_s}; distinct(S.b) = {len(set(tuples_s))}", condition=verbose)
    print_stderr("S.b:", sorted(tuples_s), condition=verbose and len(tuples_s) <= 30)
    print_stderr("topk(S.b) =", topk_s, condition=verbose)
    print_stderr(f"hist(S.b) = {histogram(tuples_s)}", condition=verbose)
    print_stderr("---- ---- ---- ----", condition=verbose)

    ues_bound = calculate_ues_bound(topk_r.max_freq(), topk_s.max_freq(), num_tuples_r, num_tuples_s, verbose=verbose)
    topk_bound = calcualte_topk_bound(topk_r, topk_s, num_tuples_r, num_tuples_s, version=topk_version,
                                      verbose=verbose)
    actual_values = execute_join(tuples_r, tuples_s)

    return SimulationResult(ues_bound, topk_bound, actual_values)


def find_regressions(generator_r: RelationGenerator, generator_s: RelationGenerator,
                     topk_generator_r: TopKListGenerator, topk_generator_s: TopKListGenerator, *,
                     exceed_value: int, version: int, topk_length: int = None, num_seeds: int = 100,
                     verbose: bool = False, progress: bool = False):
    num_tuples_r, num_tuples_s = generator_r.num_tuples(), generator_s.num_tuples()
    topk_range = range(topk_length, topk_length + 1) if topk_length else range(1, max(num_tuples_r, num_tuples_s) + 1)
    config_idx = 1
    configs = list(itertools.product(range(num_seeds), topk_range))
    for seed, topk_length in configs:
        print_stderr(f"Now simulating config {config_idx} of {len(configs)} "
                     f"(seed = {seed}, topk = {topk_length})", condition=progress)
        simulation_res = simulate(generator_r, generator_s, topk_generator_r, topk_generator_s,
                                  topk_length=topk_length, exceed_value=exceed_value, rand_seed=seed,
                                  topk_version=version, verbose=False)
        if simulation_res.ues < simulation_res.topk:
            print(f"Regression found at seed = {seed} and k = {topk_length}: UES bound is smaller Top-k bound")
        elif simulation_res.topk < simulation_res.actual:
            print(f"Regression found at seed = {seed} and k = {topk_length}: Top-k bound is no upper bound",
                  f"({simulation_res.topk} vs. {simulation_res.actual})")
        elif verbose:
            overestimation = simulation_res.topk - simulation_res.actual
            tightening = simulation_res.ues - simulation_res.topk
            print(f"At seed = {seed} and k = {topk_length}: "
                  f"Overestimation = {overestimation}; Tightening = {tightening}")


def main():
    parser = argparse.ArgumentParser(description="Utility to quickly test the behaviour of different Top-K estimation"
                                     "algorithms")
    parser.add_argument("-n", action="store", type=int, required=False, default=10,
                        help="Number of tuples per relation")
    parser.add_argument("-k", action="store", type=int, default=None, help="Length of the Top-k lists")
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
    parser.add_argument("--regression-seeds", action="store", type=int, default=DEFAULT_NUM_REGRESSION_SEEDS,
                        help="How many RNG seeds should be tried")
    parser.add_argument("--version", action="store", type=int, default=DEFAULT_TOPK_ESTIMATION_VER, help="")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Produce debugging ouput")
    parser.add_argument("--progress", action="store_true", default=False, help="Given progress info in regression "
                        "mode")
    parser.add_argument("--manual-rel-r", action="store", default="", help="Use given attribute values for relation "
                        "R instead of randomly generated ones. Format a,a,b,c")
    parser.add_argument("--manual-rel-s", action="store", default="", help="Use given attribute values for relation "
                        "S instead of randomly generated ones. Format a,a,b,c")
    parser.add_argument("--manual-topk-r", action="store", default="", help="Use the given Top-k list for attribute "
                        "R.a instead of the derived one. Cropping will happen according to -k. No sanity checks are "
                        "performed! Format a:7,b:2,c:1")
    parser.add_argument("--manual-topk-s", action="store", default="", help="Use the given Top-k list for attribute "
                        "S.b instead of the derived one. Cropping will happen according to -k. No sanity checks are "
                        "performed! Format a:7,b:2,c:1")
    # TODO: Primary Key/Foreign Key joins

    args = parser.parse_args()

    if args.regression_mode and (args.manual_rel_r or args.manual_rel_s or args.manual_topk_r or args.manual_topk_s):
        parser.error("Regression mode cannot use manual tuples/top-k lists.")

    num_tuples_r = args.nr if args.nr else args.n
    num_tuples_s = args.ns if args.ns else args.n
    distinct_values_r = args.dr if args.dr else args.d
    distinct_values_s = args.ds if args.ds else args.d
    generator_r = (ManualRelationGenerator(args.manual_rel_r) if args.manual_rel_r
                   else RandomRelationGenerator(num_tuples_r, distinct_values_r))
    generator_s = (ManualRelationGenerator(args.manual_rel_s) if args.manual_rel_s
                   else RandomRelationGenerator(num_tuples_s, distinct_values_s))

    exceed_value = args.exceed if args.exceed else None
    topk_generator_r = (ManualTopKListGenerator(args.manual_topk_r) if args.manual_topk_r
                        else DistributionBasedTopKListGenerator(exceed_value=exceed_value))
    topk_generator_s = (ManualTopKListGenerator(args.manual_topk_s) if args.manual_topk_s
                        else DistributionBasedTopKListGenerator(exceed_value=exceed_value))

    topk_length = args.k if args.k is not None else 3
    version = args.version if args.version else 1

    if args.regression_mode:
        find_regressions(generator_r, generator_s, topk_generator_r, topk_generator_s,
                         topk_length=topk_length if args.k else None, exceed_value=exceed_value, version=version,
                         num_seeds=args.regression_seeds, verbose=args.verbose, progress=args.progress)
        return

    simulation_result = simulate(generator_r, generator_s, topk_generator_r, topk_generator_s,
                                 topk_length=topk_length, exceed_value=exceed_value, rand_seed=args.rand_seed,
                                 topk_version=version, verbose=args.verbose)
    ues_bound, topk_bound, actual_values = simulation_result
    print("UES bound:", ues_bound)
    print("MCV bound:", topk_bound)
    print("Actual:", actual_values)

    ues_bound_failure = ues_bound < actual_values
    topk_bound_failure = (topk_bound < actual_values) or (topk_bound > ues_bound)
    sys.exit(1 if ues_bound_failure or topk_bound_failure else 0)


if __name__ == "__main__":
    main()
