#!/usr/bin/env python3

import argparse
import os
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Literal, get_args

try:
    import quacklab
except ImportError:
    warnings.warn("quacklab not found. Trying vanilla duckdb package.")
    import duckdb as quacklab


SupportedDBs = Literal["imdb", "job", "stats"]

DBArchives = {
    "imdb": "https://db4701.inf.tu-dresden.de:8443/index.php/s/H7TKaEBr5JmdaNA/download/csv.zip",
    "job": "https://db4701.inf.tu-dresden.de:8443/index.php/s/H7TKaEBr5JmdaNA/download/csv.zip",
    "stats": "https://db4701.inf.tu-dresden.de:8443/public.php/dav/files/p8eRRMEERQE9nXC",
}

DBSchemas = {
    "imdb": "imdb-schema.sql",
    "job": "imdb-schema.sql",
    "stats": "stats-schema.sql",
}

DBImporters = {
    "imdb": "imdb-import.sql",
    "job": "imdb-import.sql",
    "stats": "stats-import.sql",
}


def log(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def fetch_archive(archive_file: Path, *, workload: SupportedDBs) -> None:
    archive_url = DBArchives[workload]

    log(".. Fetching raw data set for", workload)
    urllib.request.urlretrieve(archive_url, archive_file)

    log(".. Extracting data files for", workload)
    with zipfile.ZipFile(archive_file, "r") as zip:
        zip.extractall(archive_file.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        "-w",
        choices=get_args(SupportedDBs),
        required=True,
        help="The workload to import. JOB is just an alias for IMDB.",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=Path,
        required=False,
        default="",
        help="Path to the database file to create.",
    )
    parser.add_argument(
        "--dir",
        "-d",
        type=Path,
        required=False,
        default=None,
        help="Directory containing raw input data.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Overwrite any existing database file.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        default=False,
        help="Remove the raw data files after import.",
    )

    args = parser.parse_args()
    cwd = Path.cwd()
    root = Path(__file__).parent.resolve()

    target_db: Path = args.target or Path(args.workload)
    target_db = target_db.expanduser().resolve()
    if target_db.is_file() and not args.force:
        log(".. Database exists, doing nothing")
        sys.exit(1)
    elif target_db.is_file() and args.force:
        log(".. Removing existing DB file")
        target_db.unlink()

    data_dir: Path = args.dir or (Path.cwd() / f"{target_db}_data").parent
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_archive = data_dir / "csv.zip"

    if csv_archive.is_file():
        log(".. Re-using existing input data at", data_dir)
    else:
        log(".. Using data directory", data_dir)
        log(".. Source data does not exist")
        fetch_archive(csv_archive, workload=args.workload)

    if args.cleanup:
        csv_archive.unlink()

    os.chdir(data_dir)

    log(".. Creating", args.workload, "database at", target_db)
    duck = quacklab.connect(target_db)
    schema = root / "sql" / DBSchemas[args.workload]
    duck.execute(schema.read_text())

    log(".. Importing raw data")
    importer = root / "sql" / DBImporters[args.workload]
    duck.execute(importer.read_text())

    duck.close()
    os.chdir(cwd)


if __name__ == "__main__":
    main()
