# PostBOUND support tools

This directory contains some additional supporting utilities for working with PostBOUND. These include:

- utilities to setup a (relatively recent) Python version in the `python-3.10` directory
- `setup-py-venv.sh`, a utility to generate a Python virtual environment running PostBOUND
- `set-workload.sh`, a utility to update the currenctly active default Postgres database connection file
- `generate-workload.sh`, a utility to generate a CSV file containing workload queries. These can be consumed by the PostBOUND
  tools in the `experiments` module
- `ceb-generator.py`, a custom implementation of the Cardinality Estimation Benchmark generator [^1] to create workloads from
  query templates
- a utility to clean the OS page cache for runtime measurements in the `drop-caches` directory

All utilities should be generally run from the repositorie's root directory, e.g. as `tools/set-workload.sh --help`.
Furthermore, Python scripts have to be run as modules, e.g. `python3 -m tools.ceb-generator --help`.

---

## References

[^1]: Parimarjan Negi et al.: "Flow-Loss: Learning Cardinality Estimates That Matter" (PVLDB 2021)
