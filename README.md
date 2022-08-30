# Simplicity++

This repository modifies and extends the UES algorithm for upper bound-based join ordering of SQL queries, which was
originally proposed by Hertzschuch et al.[^0].

The implemented modifications have multiple goals:

1. support new query structures and workloads
2. enable modification of core components of the algorithm (base table cardinality estimation, join cardinality estimation, subquery generation)
3. generalize the original UES idea for calculating upper bounds of join cardinalities, thereby tightening the bounds

## Overview

TODO: description of various folders

## Architecture

![Interaction of the various UES components](doc/figures/ues-architecture.svg)

Lorem ipsum

## Improved upper bounds

![Example of Top-k based upper bound estimation](doc/figures/top-k-estimation.svg)

Lorem ipsum

## Literature

[^0]: Simplicity Done Right for Join Ordering - Hertzschuch et al., CIDR'21 ([paper](https://www.cidrdb.org/cidr2021/papers/cidr2021_paper01.pdf), [GitHub](https://github.com/axhertz/SimplicityDoneRight))
