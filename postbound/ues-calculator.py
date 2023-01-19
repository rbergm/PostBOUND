#!/usr/bin/env python3

import argparse


def main():
    parser = argparse.ArgumentParser(description="Utility to quickly calculate UES bounds for arbitrary input data.")
    parser.add_argument("-ta", type=int, required=True, help="Total number of tuples in relation A")
    parser.add_argument("-tb", type=int, required=True, help="Total number of tuples in relation B")
    parser.add_argument("-mfa", type=int, required=True, help="Maximum attribute frequency of A's join attribute")
    parser.add_argument("-mfb", type=int, required=True, help="Maximum attribute frequency of B's join attribute")
    parser.add_argument("--verbose", "-v", action="store_true", default=False, help="Also print distinct value counts")

    args = parser.parse_args()
    da = args.ta / args.mfa
    db = args.tb / args.mfb

    bound = min(da, db) * args.mfa * args.mfb
    bound = round(bound)

    if args.verbose:
        print(" distinct(a) =", da)
        print(" distinct(b) =", db)
        print("upper(A ⋈ B) =", bound)
    else:
        print(bound)


if __name__ == "__main__":
    main()
