#!/usr/bin/env python3

import argparse
import os
import pathlib
import textwrap


def load_online(out: str, source="http://homepages.cwi.nl/~boncz/job/imdb.tgz") -> None:
    print(".. Downloading data set")
    os.system("wget --output-document imdb_temp {source}".format(source=source))
    os.makedirs(out, exist_ok=True)
    print(".. Unpacking data")
    os.system("tar xvf imdb_temp --directory {out}".format(out=out))
    os.system("rm imdb_temp")
    print(".. Download done")


def postgres_setup(source_dir: str, db_name: str) -> None:
    print(".. Creating database")
    os.system(f"createdb {db_name}")
    os.system(f"psql {db_name} -f {source_dir}/schematext.sql")

    source_path = pathlib.Path(source_dir)
    data_files = source_path.glob("*.csv")
    for data_file in data_files:
        print(".. Now importing", data_file)
        table = data_file.stem
        os.system(f"""psql {db_name} -c "\copy {table} from {data_file} with csv quote '\\"' escape '\\';" """)


def create_fkeys(out: str, db_name: str) -> None:
    fkeys_spec = "https://raw.githubusercontent.com/gregrahn/join-order-benchmark/master/fkindexes.sql"
    fkey_src = "imdb-fkeys.sql"
    fkey_path = out + "/" + fkey_src
    print(f".. Fetching Foreign keys specification from {fkeys_spec}")
    os.system(f"wget -nv --output-document {fkey_path} {fkeys_spec}")
    print(".. Creating foreign key indices")
    os.system(f"psql {db_name} -f {fkey_path}")
    os.system(f"psql {db_name} -c \"create index if not exists subject_id_complete_cast on complete_cast(subject_id)\"")
    os.system(f"psql {db_name} -c \"create index if not exists status_id_complete_cast on complete_cast(status_id)\"")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent("""\
        Utility to create IMDB database instances for PostgreSQL

        When running this script, the PostgreSQL server has to be running and accessible via the `psql` command. Furthermore, utilities
        such as `createdb` have to be on the PATH as well."""))
    arg_grp = parser.add_mutually_exclusive_group()
    arg_grp.add_argument("--online", action="store_true", help="If set, download the IMDB data set")
    arg_grp.add_argument("--source", action="store", help="Directory to load the raw IMDB data set from")
    arg_grp.add_argument("--fkeys", action="store_true", help="Don't setup a new IMDB instance, but create foreign keys for an existing one.")
    parser.add_argument("--target", action="store", default="imdb", help="Directory to store the raw IMDB data in, if downloading from network")
    parser.add_argument("--db-name", action="store", default="imdb", help="Name of the database to work on")

    args = parser.parse_args()

    # if we should load foreign keys, only do so without the actual database setup
    if args.fkeys:
        create_fkeys(args.target, args.db_name)
        return

    source_dir = args.target if args.online else args.source
    if args.online:
        load_online(args.target)
    postgres_setup(source_dir, args.db_name)


if __name__ == "__main__":
    main()
