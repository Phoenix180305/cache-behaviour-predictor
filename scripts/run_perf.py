"""
run_perf.py
Runs each compiled binary under `perf stat` and collects cache metrics.
Writes: perf_results.csv
"""

import csv
import re
import subprocess
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BIN_DIR     = Path("binaries")
RESULTS_CSV = Path("perf_results.csv")

PERF_EVENTS = "cache-references,cache-misses"
DEFAULT_TIMEOUT = 30    # seconds per binary
DEFAULT_JOBS    = 4     # perf stat can be CPU-intensive; keep low


def run_perf(prog_id: int, bin_path: Path, timeout: int) -> dict:
    """Run perf stat on one binary, return parsed metrics."""
    cmd = [
        "perf", "stat",
        "-e", PERF_EVENTS,
        "-x", ",",
        "--",
        str(bin_path),
    ]

    result = dict(
        prog_id=prog_id,
        cache_references=None,
        cache_misses=None,
        miss_rate=None,
        wall_time_s=None,
        status="ok",
    )

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        result["wall_time_s"] = round(time.perf_counter() - t0, 4)

        # perf -x , writes counter values to stderr, one per line:
        refs  = None
        misses = None
        for line in proc.stderr.decode(errors="replace").splitlines():
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            val_str = parts[0].replace(".", "").replace(",", "").strip()
            event   = parts[2].strip()
            try:
                val = int(val_str)
            except ValueError:
                continue
            if "cache-references" in event:
                refs = val
            elif "cache-misses" in event:
                misses = val

        result["cache_references"] = refs
        result["cache_misses"]     = misses
        if refs and refs > 0 and misses is not None:
            result["miss_rate"] = round(misses / refs, 6)

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except FileNotFoundError:
        result["status"] = "perf_not_found"
        print("[ERROR] `perf` not found. Install with: sudo apt install linux-perf")
        sys.exit(1)
    except Exception as e:
        result["status"] = f"error:{e}"

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs",    type=int, default=DEFAULT_JOBS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--limit",   type=int, default=None,
                    help="Only measure first N programs (for testing)")
    args = ap.parse_args()

    binaries = sorted(BIN_DIR.glob("prog_*"))
    if args.limit:
        binaries = binaries[:args.limit]

    print(f"[perf] Measuring {len(binaries)} binaries with {args.jobs} workers")

    tasks = []
    for p in binaries:
        # extract prog_id from filename prog_000042
        try:
            pid = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            pid = -1
        tasks.append((pid, p))

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(run_perf, pid, p, args.timeout): pid
                for pid, p in tasks}
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(tasks)} done …")

    rows.sort(key=lambda r: r["prog_id"])

    fieldnames = ["prog_id", "cache_references", "cache_misses",
                  "miss_rate", "wall_time_s", "status"]
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    ok       = sum(1 for r in rows if r["status"] == "ok")
    timeouts = sum(1 for r in rows if r["status"] == "timeout")
    print(f"[perf] Done — {ok} ok, {timeouts} timeouts → {RESULTS_CSV}")


if __name__ == "__main__":
    main()
