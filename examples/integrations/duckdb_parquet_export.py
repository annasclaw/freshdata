"""examples/integrations/duckdb_parquet_export.py — Partitioned Parquet In, Cleaned Parquet Out.

Problem:
--------
Data engineering pipelines rarely receive a single tidy file. They receive a
directory of partitioned Parquet files (Hive-style `dt=YYYY-MM-DD/` folders)
that must be read, cleaned, and written back out without leaving DuckDB. Doing
that by hand means a pile of `CASE WHEN` / `TRY_CAST` expressions, and reading
the raw dataset straight into SQL means the cleaning rules are not reusable or
testable.

This recipe keeps the whole round trip in DuckDB:

1. Write a small multi-partition dataset to disk (dirty, on purpose).
2. Read the whole partition tree in one query via `read_parquet(...,
   hive_partitioning=1)` so the partition column is promoted to a real column.
3. Clean once with FreshData, returning the quality report so the effect of the
   cleaning step is visible rather than assumed.
4. Export back out with `COPY ... TO ... (FORMAT PARQUET, PARTITION_BY ...)`,
   then re-register the cleaned tree and query it to prove the round trip.

Installation:
-------------
pip install "freshdata-cleaner[duckdb]"

Run:
----
python examples/integrations/duckdb_parquet_export.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import duckdb

import freshdata as fd

RAW_DIRNAME = "raw_events"
CLEAN_DIRNAME = "clean_events"
# Partition column produced by Hive-style partitioning on both sides.
PARTITION_COLUMN = "dt"


def build_raw_dataset(raw_dir: Path) -> None:
    """Write a deliberately dirty, multi-partition dataset to ``raw_dir``.

    Each partition directory holds one Parquet file, which is the shape most
    pipelines emit: ``raw_events/dt=2026-10-13/part-0.parquet``.
    """
    partitions = {
        "2026-10-13": [
            ("TX-1001", " Alice Corp ", "$1,250.00", "5%"),
            ("TX-1002", "Bob LLC", "N/A", "0%"),
        ],
        "2026-10-14": [
            ("TX-1003", "  Charlie Inc", "$980.50", "missing"),
            ("TX-1004", "Diana Tech", "$3,400.00", "10%"),
            ("TX-1005", "", "$-", "n/a"),
        ],
    }

    con = duckdb.connect()
    try:
        for partition_date, rows in partitions.items():
            part_dir = raw_dir / f"{PARTITION_COLUMN}={partition_date}"
            part_dir.mkdir(parents=True, exist_ok=True)
            values = ",\n".join(
                "    (" + ", ".join(f"'{value}'" for value in row) + ")" for row in rows
            )
            con.execute(
                f"""
                COPY (
                    SELECT *
                    FROM (VALUES
                        {values}
                    ) AS t(transaction_id, customer_name, order_amount, discount_pct)
                ) TO '{part_dir}/part-0.parquet' (FORMAT PARQUET);
                """
            )
    finally:
        con.close()


def main() -> None:
    print("=== Integration: Partitioned Parquet Cleaning & Export ===")

    workdir = Path(tempfile.mkdtemp(prefix="freshdata_duckdb_export_"))
    raw_dir = workdir / RAW_DIRNAME
    clean_dir = workdir / CLEAN_DIRNAME

    try:
        # 1. Materialise a multi-partition raw dataset on disk.
        build_raw_dataset(raw_dir)
        partition_dirs = sorted(p.name for p in raw_dir.iterdir() if p.is_dir())
        parquet_files = sorted(raw_dir.rglob("*.parquet"))
        print("\n[1] Raw partition tree:")
        print(f"  root          : {raw_dir}")
        print(f"  partitions    : {partition_dirs}")
        print(f"  parquet files : {len(parquet_files)}")

        con = duckdb.connect()
        try:
            # 2. Read the whole tree in one query. hive_partitioning=1 promotes
            #    the dt= directory segment to a real column.
            glob = (raw_dir / "**" / "*.parquet").as_posix()
            raw_relation = con.execute(
                f"""
                SELECT *
                FROM read_parquet('{glob}', hive_partitioning=1)
                ORDER BY transaction_id
                """
            )
            raw_df = raw_relation.df()
            print("\n[2] Raw dataset read by DuckDB:")
            print(raw_df.to_string(index=False))

            # 3. Clean with FreshData. return_report=True keeps the before/after
            #    quality metrics visible instead of assuming the clean succeeded.
            cleaned_df, report = fd.clean(raw_df, return_report=True)
            print("\n[3] Cleaned dataset:")
            print(cleaned_df.to_string(index=False))
            print("\n    Quality report:")
            print(f"    {report}")

            # 4a. Export back out as a partitioned Parquet tree. COPY TO takes a
            #     directory (not a glob) and PARTITION_BY writes one subdirectory
            #     per partition value, reproducing the input layout -- so the
            #     export is a drop-in replacement for the raw dataset.
            con.register("events_clean", cleaned_df)
            con.execute(
                f"""
                COPY (
                    SELECT * FROM events_clean
                    WHERE {PARTITION_COLUMN} IS NOT NULL
                )
                TO '{clean_dir.as_posix()}'
                (FORMAT PARQUET, PARTITION_BY ({PARTITION_COLUMN}), OVERWRITE_OR_IGNORE);
                """
            )
            cleaned_glob = (clean_dir / "**" / "*.parquet").as_posix()
            exported_files = sorted(clean_dir.rglob("*.parquet"))
            print("\n[4] Exported partition tree:")
            print(f"  root          : {clean_dir}")
            print(f"  parquet files : {len(exported_files)}")

            # 4b. Prove the round trip: re-read what was just written and query
            #     it with plain SQL. No in-memory state carried over.
            #
            #     hive_partitioning=1 infers a TIMESTAMP for the dt= directory
            #     value, so cast it back to DATE for a readable report. The
            #     amount is still a formatted string here -- stripping currency
            #     symbols and separators is the kind of SQL that FreshData's
            #     cleaning already did for the sentinel/blank cases, and doing
            #     it in the query keeps the example honest about what the
            #     library does NOT do (it does not parse currency).
            round_trip = con.execute(
                f"""
                SELECT
                    CAST({PARTITION_COLUMN} AS DATE)                       AS partition_day,
                    COUNT(*)                                              AS rows,
                    COUNT(order_amount)                                   AS rows_with_amount,
                    ROUND(
                        SUM(TRY_CAST(REGEXP_REPLACE(order_amount, '[^0-9.-]', '', 'g') AS DOUBLE)),
                        2
                    )                                                     AS total_amount
                FROM read_parquet('{cleaned_glob}', hive_partitioning=1)
                GROUP BY 1
                ORDER BY 1
                """
            ).df()
            print("\n[5] Analytical SQL over the re-read export:")
            print(round_trip.to_string(index=False))
        finally:
            con.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        print(f"\nCleaned up temporary directory: {workdir}")


if __name__ == "__main__":
    main()
