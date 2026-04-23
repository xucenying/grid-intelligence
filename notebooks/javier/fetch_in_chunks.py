"""
fetch_in_chunks.py
==================
Fetches ENTSO-E / weather / gas data in configurable time chunks
and writes each chunk to its own CSV file.

Usage
-----
python fetch_in_chunks.py --entsoe_api_key YOUR_KEY
python fetch_in_chunks.py --entsoe_api_key YOUR_KEY --start 2021-01-01 --end 2022-01-01 --chunk_months 3
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import pandas as pd
from requests.exceptions import HTTPError

try:
    from fetcher import EntsoeSource, WeatherSource, GasSource, DataFetcher
except ImportError:
    sys.exit("Could not import fetcher.py — stelle sicher dass beide Scripts im gleichen Ordner sind.")

OUTPUT_DIR = "raw_data/chunks"


def fetch_with_retry(func, *args, max_retries=5, base_wait=60, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 503 and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"    503 error — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise


def fetch_chunk(fetcher, start, end, country):
    return fetch_with_retry(fetcher._fetch_range, start, end, country)


def daterange_chunks(start, end, months):
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + relativedelta(months=months), end)
        yield cursor, next_cursor
        cursor = next_cursor


def output_path(start, end):
    fmt = "%Y-%m-%d"
    return os.path.join(OUTPUT_DIR, f"data_{start.strftime(fmt)}_{end.strftime(fmt)}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entsoe_api_key", required=True)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-04-21")
    parser.add_argument("--chunk_months", type=int, default=3)
    parser.add_argument("--country", default="DE_LU")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fetcher = DataFetcher(entsoe_api_key=args.entsoe_api_key)

    chunks = list(daterange_chunks(start_dt, end_dt, args.chunk_months))
    print(f"Total chunks: {len(chunks)}  ({args.chunk_months}-Monats-Fenster)\n")

    success, skipped, failed = [], [], []

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        path = output_path(chunk_start, chunk_end)
        label = f"[{i}/{len(chunks)}] {chunk_start.date()} → {chunk_end.date()}"

        if args.skip_existing and os.path.exists(path):
            print(f"  ⏭  {label}  — bereits vorhanden, übersprungen")
            skipped.append(path)
            continue

        print(f"  ⬇  {label} ...")
        try:
            df = fetch_chunk(fetcher, chunk_start, chunk_end, args.country)
            df.to_csv(path, index=True)
            print(f"     ✓  gespeichert → {path}  ({len(df):,} rows)")
            success.append(path)
        except Exception as exc:
            print(f"     ✗  FEHLER: {exc}")
            failed.append((path, str(exc)))

    print("\n" + "=" * 60)
    print(f"Fertig.  ✓ {len(success)} gespeichert  |  ⏭ {len(skipped)} übersprungen  |  ✗ {len(failed)} fehlgeschlagen")
    if failed:
        print("\nFehlgeschlagene Chunks:")
        for p, err in failed:
            print(f"  {p}: {err}")
    else:
        print(f"\nAlle CSVs in: {OUTPUT_DIR}/")
        print("Danach merge_chunks.py ausführen.")


if __name__ == "__main__":
    main()
