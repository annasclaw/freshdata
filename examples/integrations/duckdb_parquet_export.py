"""examples/integrations/duckdb_parquet_export.py — Partitioned Parquet in, Parquet out.

Problem:
--------
Analytical pipelines usually keep their facts on disk as a partitioned Parquet
dataset (``region=us/part-0.parquet``, ``region=eu/part-0.parquet``, ...) and
query it with DuckDB. The files arrive exactly as messy as a CSV would: padded
strings, currency/percent sentinels, text-typed numerics, and duplicate rows
from an upstream join.

FreshData repairs that data *once*, and DuckDB carries it back to disk or into
an analytical table — no second loader, no bespoke ``CASE WHEN`` chain.

Installation:
-------------
pip install "freshdata-cleaner[duckdb]"

Run:
----
python examples/integrations/duckdb_parquet_export.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb

import freshdata as fd

# Raw rows land in three partitions. Every partition carries the same schema so
# DuckDB's hive_partitioning read can union them into one relation.
PARTITIONS = {
    "us": [
        ("TX-001", "  Alice Corp ", "$12,450.00", "5%"),
        ("TX-002", "Bob LLC", "N/A", "0%"),
        ("TX-003", "Charlie Inc", "$8,900.50", "missing"),
    ],
    "eu": [
        ("TX-004", " Diana Tech ", "€34,000.00", "10%"),
        ("TX-005", "Epsilon Ltd", "N/A", "-"),
        ("TX-006", "Zeta GmbH", "€1,250.00", "missing"),
    ],
    "apac": [
        ("TX-007", "Eta KK", "$7,300.25", "2%"),
        ("TX-008", "  Theta Inc  ", "$990.00", "1%"),
        ("TX-007", "Eta KK", "$7,300.25", "2%"),  # duplicate from a re-run of the source job
    ],
}


def write_partitioned_parquet(root: Path) -> Path:
    """Lay down ``root/region=<name>/part-0.parquet`` for every partition."""
    for region, rows in PARTITIONS.items():
        partition_dir = root / f"region={region}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        try:
            con.execute(
                """
                CREATE TABLE sales_raw (
                    transaction_id VARCHAR,
                    customer_name VARCHAR,
                    order_amount VARCHAR,
                    discount_pct VARCHAR
                );
                """
            )
            con.executemany("INSERT INTO sales_raw VALUES (?, ?, ?, ?)", rows)
            con.execute(
                "COPY sales_raw TO ? (FORMAT PARQUET)",
                [str(partition_dir / "part-0.parquet")],
            )
        finally:
            con.close()
    return root


def read_dataset(con: duckdb.DuckDBPyConnection, dataset_root: Path):
    """Read every partition back as one relation, keyed by the hive column."""
    return con.execute(
        "SELECT * FROM read_parquet(?, hive_partitioning = true)",
        [str(dataset_root / "**" / "*.parquet")],
    ).df()


def main() -> None:
    print("=== Integration: Partitioned Parquet <-> DuckDB <-> FreshData ===")

    con = duckdb.connect()
    with tempfile.TemporaryDirectory(prefix="fd-parquet-") as tmp:
        raw_root = write_partitioned_parquet(Path(tmp) / "raw")

        print("\n[1] Partitioned dataset on disk:")
        for path in sorted(raw_root.rglob("*.parquet")):
            print(f"  {path.relative_to(raw_root)}")

        # 2. One query fans out across all partitions; freshdata then repairs the
        #    union in a single pass instead of once per file.
        raw_df = read_dataset(con, raw_root)
        print("\n[2] Raw rows read across partitions:")
        print(raw_df.to_string(index=False))

        cleaned_df, report = fd.clean(
            raw_df,
            preserve_columns=("transaction_id", "region"),
            drop_duplicates=True,
            strategy="conservative",
            return_report=True,
        )

        print("\n[3] Cleaning report:")
        print(f"  rows        {report.rows_before} -> {report.rows_after}")
        print(f"  cells fixed {report.cells_changed}")
        print(f"  duplicates  {report.duplicates_removed} removed")
        print(f"  warnings    {len(report.warnings)}")
        for warning in report.warnings:
            print(f"    - {warning}")

        print("\n[4] Cleaned rows:")
        print(cleaned_df.to_string(index=False))

        # 3a. Export path A: back to a partitioned Parquet dataset, partitioned
        #     by the same hive column so downstream readers see the same layout.
        clean_root = Path(tmp) / "clean"
        con.register("sales_clean_view", cleaned_df)
        con.execute(
            "COPY sales_clean_view TO ? (FORMAT PARQUET, PARTITION_BY (region))",
            [str(clean_root)],
        )
        exported = sorted(p.relative_to(clean_root) for p in clean_root.rglob("*.parquet"))
        print("\n[5] Exported Parquet dataset (partitioned by region):")
        for path in exported:
            print(f"  {path}")

        # 3b. Export path B: register as a DuckDB table for analytical SQL.
        con.execute("CREATE OR REPLACE TABLE sales_clean AS SELECT * FROM sales_clean_view")
        print("\n[6] Analytical SQL on the exported table:")
        print(
            con.execute(
                """
                -- order_amount came back numeric, so it aggregates directly.
                -- discount_pct stayed VARCHAR on purpose: freshdata only coerces a
                -- column to numeric when enough values parse, and a percent column
                -- carrying sentinels does not clear that bar. Coercing it by hand
                -- would have been a guess, so DuckDB casts per row instead.
                SELECT
                    region,
                    COUNT(*)                                          AS orders,
                    ROUND(SUM(order_amount), 2)                      AS revenue,
                    ROUND(AVG(TRY_CAST(REPLACE(discount_pct, '%', '') AS DOUBLE)), 1)    AS avg_discount
                FROM sales_clean
                GROUP BY region
                ORDER BY revenue DESC NULLS LAST
                """
            )
            .df()
            .to_string(index=False)
        )

        # 3c. Prove the round trip: re-read the exported dataset from disk and
        #     confirm it matches the in-memory cleaned frame.
        round_trip = read_dataset(con, clean_root)
        same = round_trip.sort_values(["transaction_id"]).reset_index(drop=True).equals(
            cleaned_df.sort_values(["transaction_id"]).reset_index(drop=True)
        )
        print("\n[7] Round-trip check (cleaned frame == re-read Parquet dataset):")
        print(f"  {round_trip.shape[0]} rows, identical={same}")

    con.close()


if __name__ == "__main__":
    main()