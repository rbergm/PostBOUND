#!/usr/bin/env python3

import argparse

import pandas as pd

from transform import mosp
from postgres import hint


def read_input(src: str, query_col: str) -> pd.DataFrame:
    df = pd.read_csv(src)
    df[query_col] = df[query_col].apply(mosp.MospQuery.parse)
    return df


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("input", action="store", help="")
    parser.add_argument("--out", "-o", action="store", default="out.csv", help="")
    parser.add_argument("--query-col", action="store", default="query", help="")
    parser.add_argument("--hint-col", action="store", default="hint", help="")

    args = parser.parse_args()
    df = read_input(args.input, args.query_col)
    df[args.hint_col] = (df[args.query_col]
                         .apply(hint.idxnlj_subqueries)
                         .apply(hint.HintedMospQuery.generate_sqlcomment, strip_empty=True))

    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
