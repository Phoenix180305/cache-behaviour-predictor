"""
generate_dataset.py
Merges program_metadata.csv + perf_results.csv → dataset.csv
"""

import csv
import statistics
import argparse
from pathlib import Path

METADATA_CSV  = Path("program_metadata.csv")
PERF_CSV      = Path("perf_results.csv")
DATASET_CSV   = Path("dataset.csv")

# Features to drop before ML (they directly encode the label or are IDs)
LEAKAGE_COLS  = {"cache_references", "cache_misses", "miss_rate",
                 "wall_time_s", "status", "src_path"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def auto_cast(v):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--friendly-threshold", type=float, default=None,
                    help="Override global miss-rate threshold (0-1). "
                         "Default: per-pattern median split.")
    args = ap.parse_args()

    print("[dataset] Reading metadata …")
    meta_rows = read_csv(METADATA_CSV)
    print(f"          {len(meta_rows)} programs")

    print("[dataset] Reading perf results …")
    perf_rows = read_csv(PERF_CSV)
    print(f"          {len(perf_rows)} measurements")

    # Index perf by prog_id
    perf_by_id = {int(r["prog_id"]): r for r in perf_rows}

    # Merge
    merged = []
    skipped = 0
    for m in meta_rows:
        pid = int(m["prog_id"])
        p   = perf_by_id.get(pid)
        if p is None or p.get("status") != "ok" or not p.get("miss_rate"):
            skipped += 1
            continue
        row = {k: auto_cast(v) for k, v in m.items()}
        row["miss_rate_raw"]  = float(p["miss_rate"])
        row["wall_time_s"]    = float(p["wall_time_s"])
        merged.append(row)

    print(f"[dataset] {len(merged)} usable rows ({skipped} dropped — missing/timeout)")

    # ── Labeling ──────────────────────────────────────────────────────────────
    if args.friendly_threshold is not None:
        # Global threshold override
        threshold = args.friendly_threshold
        for row in merged:
            row["cache_friendly"] = int(row["miss_rate_raw"] <= threshold)
        print(f"[dataset] Global threshold: miss_rate ≤ {threshold:.4f} → friendly")
    else:
        # Per-pattern-type median split (default, balanced)
        patterns = set(r["pattern"] for r in merged)
        thresholds = {}
        for pat in patterns:
            rates = [r["miss_rate_raw"] for r in merged if r["pattern"] == pat]
            thresholds[pat] = statistics.median(rates) if len(rates) >= 10 else None

        global_median = statistics.median(r["miss_rate_raw"] for r in merged)

        for row in merged:
            pat = row["pattern"]
            thr = thresholds.get(pat) or global_median
            row["cache_friendly"] = int(row["miss_rate_raw"] <= thr)

        print("[dataset] Per-pattern median thresholds:")
        for pat, thr in sorted(thresholds.items()):
            count   = sum(1 for r in merged if r["pattern"] == pat)
            n_frien = sum(1 for r in merged
                          if r["pattern"] == pat and r["cache_friendly"] == 1)
            t = thr if thr else global_median
            print(f"  {pat:20s}  threshold={t:.4f}  "
                  f"friendly={n_frien}/{count} ({100*n_frien/count:.0f}%)")

    friendly_total = sum(r["cache_friendly"] for r in merged)
    print(f"\n[dataset] Overall — friendly: {friendly_total} "
          f"({100*friendly_total/len(merged):.1f}%)  "
          f"hostile: {len(merged)-friendly_total} "
          f"({100*(len(merged)-friendly_total)/len(merged):.1f}%)")

    # Drop leakage columns
    # keep miss_rate_raw for the label summary but drop before ML output
    keep_leakage_in_dataset = False   # set True to inspect raw perf in dataset
    for row in merged:
        if not keep_leakage_in_dataset:
            row.pop("miss_rate_raw", None)
            row.pop("wall_time_s", None)

    # Drop src_path, prog_id is kept as an index
    for row in merged:
        row.pop("src_path", None)

    # Write final dataset
    fieldnames = list(merged[0].keys())
    # Move target to last column
    if "cache_friendly" in fieldnames:
        fieldnames.remove("cache_friendly")
        fieldnames.append("cache_friendly")

    with open(DATASET_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged)

    print(f"\n[dataset] Saved {len(merged)} rows × {len(fieldnames)} cols → {DATASET_CSV}")
    print(f"[dataset] Feature columns: {len(fieldnames) - 2} "
          f"(excluding prog_id and cache_friendly)")


if __name__ == "__main__":
    main()
