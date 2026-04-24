"""
merge_chunks.py — Merges all chunk CSVs into consolidated_full.csv
"""
import argparse, glob, os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks_dir", default="raw_data/chunks")
    parser.add_argument("--output", default="raw_data/consolidated_full.csv")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.chunks_dir, "data_*.csv")))
    if not files:
        print(f"Keine Chunk-Dateien gefunden in: {args.chunks_dir}")
        return

    print(f"{len(files)} Chunks gefunden — merging...\n")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            print(f"  ✓ {os.path.basename(f)}  ({len(df):,} rows)")
            dfs.append(df)
        except Exception as e:
            print(f"  ✗ {os.path.basename(f)}  — skipped: {e}")

    combined = pd.concat(dfs)
    before = len(combined)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    after = len(combined)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    combined.to_csv(args.output)

    print(f"\n{'='*50}")
    print(f"Gesamt: {before:,} rows → nach Deduplizierung: {after:,} rows")
    print(f"Gespeichert: {args.output}")

if __name__ == "__main__":
    main()
