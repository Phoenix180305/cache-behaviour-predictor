"""
generate_programs.py
Generates synthetic C programs with varied memory access patterns.
Writes: programs/<id>.c and program_metadata.csv
"""

import os
import csv
import random
import itertools
from pathlib import Path

PROGRAM_DIR = Path("programs")
METADATA_FILE = Path("program_metadata.csv")
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

ARRAY_SIZES   = [1024, 4096, 16384, 65536, 262144, 1048576]   # elements
STRIDES       = [1, 2, 4, 8, 16, 32, 64, 128]
ITER_COUNTS   = [100, 1000, 10000]
MATRIX_SIZES  = [32, 64, 128, 256, 512]
TILE_SIZES    = [16, 32, 64]
WORKING_SETS  = [256, 1024, 4096, 16384]

PATTERNS = [
    "sequential", "strided", "random",
    "matrix_row", "matrix_col", "matrix_tiled",
    "pointer_chase", "histogram",
    "gather_scatter", "working_set",
]

TARGET_PROGRAMS = 30_000

def tmpl_sequential(n, iters):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define N {n}
#define ITERS {iters}
int main(void) {{
    int *a = (int*)malloc(N * sizeof(int));
    if (!a) return 1;
    long long s = 0;
    for (int it = 0; it < ITERS; it++)
        for (int i = 0; i < N; i++) {{ a[i] = i; s += a[i]; }}
    printf("%lld\\n", s);
    free(a); return 0;
}}
"""

def tmpl_strided(n, stride, iters):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define N {n}
#define STRIDE {stride}
#define ITERS {iters}
int main(void) {{
    int *a = (int*)malloc(N * sizeof(int));
    if (!a) return 1;
    long long s = 0;
    for (int it = 0; it < ITERS; it++)
        for (int i = 0; i < N; i += STRIDE) {{ a[i] = i; s += a[i]; }}
    printf("%lld\\n", s);
    free(a); return 0;
}}
"""

def tmpl_random(n, iters):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define N {n}
#define ITERS {iters}
int main(void) {{
    int *a = (int*)malloc(N * sizeof(int));
    int *idx = (int*)malloc(N * sizeof(int));
    if (!a || !idx) return 1;
    srand(42);
    for (int i = 0; i < N; i++) idx[i] = rand() % N;
    long long s = 0;
    for (int it = 0; it < ITERS; it++)
        for (int i = 0; i < N; i++) {{ a[idx[i]] = i; s += a[idx[i]]; }}
    printf("%lld\\n", s);
    free(a); free(idx); return 0;
}}
"""

def tmpl_matrix_row(m):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define M {m}
int main(void) {{
    int (*a)[M] = (int(*)[M])malloc(M * M * sizeof(int));
    if (!a) return 1;
    long long s = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) {{ a[i][j] = i+j; s += a[i][j]; }}
    printf("%lld\\n", s);
    free(a); return 0;
}}
"""

def tmpl_matrix_col(m):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define M {m}
int main(void) {{
    int (*a)[M] = (int(*)[M])malloc(M * M * sizeof(int));
    if (!a) return 1;
    long long s = 0;
    for (int j = 0; j < M; j++)
        for (int i = 0; i < M; i++) {{ a[i][j] = i+j; s += a[i][j]; }}
    printf("%lld\\n", s);
    free(a); return 0;
}}
"""

def tmpl_matrix_tiled(m, t):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define M {m}
#define T {t}
int main(void) {{
    int (*a)[M] = (int(*)[M])malloc(M * M * sizeof(int));
    if (!a) return 1;
    long long s = 0;
    for (int ii = 0; ii < M; ii += T)
        for (int jj = 0; jj < M; jj += T)
            for (int i = ii; i < ii+T && i < M; i++)
                for (int j = jj; j < jj+T && j < M; j++)
                    {{ a[i][j] = i+j; s += a[i][j]; }}
    printf("%lld\\n", s);
    free(a); return 0;
}}
"""

def tmpl_pointer_chase(n):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define N {n}
int main(void) {{
    int *next = (int*)malloc(N * sizeof(int));
    if (!next) return 1;
    // build random permutation
    int *perm = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) perm[i] = i;
    srand(42);
    for (int i = N-1; i > 0; i--) {{
        int j = rand() % (i+1); int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }}
    for (int i = 0; i < N; i++) next[perm[i]] = perm[(i+1) % N];
    free(perm);
    long long s = 0; int cur = 0;
    for (int i = 0; i < N; i++) {{ cur = next[cur]; s += cur; }}
    printf("%lld\\n", s);
    free(next); return 0;
}}
"""

def tmpl_histogram(n, bins=256):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define N {n}
#define BINS {bins}
int main(void) {{
    unsigned char *data = (unsigned char*)malloc(N);
    int *hist = (int*)calloc(BINS, sizeof(int));
    if (!data || !hist) return 1;
    srand(42);
    for (int i = 0; i < N; i++) data[i] = rand() % BINS;
    for (int i = 0; i < N; i++) hist[data[i]]++;
    long long s = 0;
    for (int i = 0; i < BINS; i++) s += hist[i];
    printf("%lld\\n", s);
    free(data); free(hist); return 0;
}}
"""

def tmpl_gather_scatter(n, iters):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define N {n}
#define ITERS {iters}
int main(void) {{
    int *src = (int*)malloc(N * sizeof(int));
    int *dst = (int*)malloc(N * sizeof(int));
    int *idx = (int*)malloc(N * sizeof(int));
    if (!src || !dst || !idx) return 1;
    srand(42);
    for (int i = 0; i < N; i++) {{ src[i] = i; idx[i] = rand() % N; }}
    long long s = 0;
    for (int it = 0; it < ITERS; it++)
        for (int i = 0; i < N; i++) {{ dst[idx[i]] = src[i]; s += dst[idx[i]]; }}
    printf("%lld\\n", s);
    free(src); free(dst); free(idx); return 0;
}}
"""

def tmpl_working_set(ws, iters):
    return f"""\
#include <stdio.h>
#include <stdlib.h>
#define WS {ws}
#define ITERS {iters}
int main(void) {{
    int *a = (int*)malloc(WS * sizeof(int));
    if (!a) return 1;
    long long s = 0;
    for (int it = 0; it < ITERS; it++)
        for (int i = 0; i < WS; i++) {{ a[i] = it+i; s += a[i]; }}
    printf("%lld\\n", s);
    free(a); return 0;
}}
"""

# Feature Extraction
def features_for(pattern, params):
    """Return a dict of static features for a given program."""
    f = dict(pattern=pattern)

    array_size   = params.get("array_size", 0)
    stride       = params.get("stride", 1)
    iters        = params.get("iters", 1)
    matrix_size  = params.get("matrix_size", 0)
    tile_size    = params.get("tile_size", 0)
    working_set  = params.get("working_set", 0)

    f["array_size"]         = array_size
    f["stride"]             = stride
    f["iters"]              = iters
    f["matrix_size"]        = matrix_size
    f["tile_size"]          = tile_size
    f["working_set"]        = working_set

    L1_CACHE_BYTES  = 32   * 1024     # 32 KB
    L2_CACHE_BYTES  = 256  * 1024     # 256 KB
    L3_CACHE_BYTES  = 8192 * 1024     # 8 MB (typical)
    CACHE_LINE      = 64              # bytes
    INT_SIZE        = 4               # bytes

    eff_size  = (matrix_size * matrix_size) if matrix_size > 0 else (working_set if working_set > 0 else array_size)
    bytes_ws  = eff_size * INT_SIZE

    f["working_set_bytes"]        = bytes_ws
    f["fits_in_L1"]               = int(bytes_ws <= L1_CACHE_BYTES)
    f["fits_in_L2"]               = int(bytes_ws <= L2_CACHE_BYTES)
    f["fits_in_L3"]               = int(bytes_ws <= L3_CACHE_BYTES)
    f["L1_ratio"]                 = bytes_ws / L1_CACHE_BYTES
    f["L2_ratio"]                 = bytes_ws / L2_CACHE_BYTES
    f["L3_ratio"]                 = bytes_ws / L3_CACHE_BYTES

    f["stride_in_cache_lines"]    = max(1, stride * INT_SIZE) / CACHE_LINE
    f["elements_per_cache_line"]  = CACHE_LINE / INT_SIZE          # 16 for int
    f["stride_crosses_line"]      = int((stride * INT_SIZE) >= CACHE_LINE)

    # spatial locality score (1=perfect sequential, 0=pure random)
    pat_scores = {
        "sequential":    1.00,
        "matrix_row":    1.00,
        "matrix_tiled":  0.95,
        "histogram":     0.70,
        "working_set":   0.90,
        "strided":       max(0.0, 1.0 - (stride - 1) / 128.0),
        "matrix_col":    max(0.0, 1.0 - (matrix_size - 1) / 512.0),
        "gather_scatter":0.20,
        "random":        0.05,
        "pointer_chase": 0.02,
    }
    f["spatial_locality_score"]   = pat_scores.get(pattern, 0.5)

    # temporal locality score
    temp_scores = {
        "sequential":    0.30,
        "matrix_row":    0.30,
        "matrix_col":    0.30,
        "working_set":   1.00,
        "histogram":     0.80,
        "matrix_tiled":  0.85,
        "strided":       0.30,
        "gather_scatter":0.10,
        "random":        0.10,
        "pointer_chase": 0.05,
    }
    f["temporal_locality_score"]  = temp_scores.get(pattern, 0.5)

    # access regularity (1=regular/predictable, 0=irregular)
    reg_scores = {
        "sequential":    1.00,
        "matrix_row":    1.00,
        "strided":       1.00,
        "matrix_col":    1.00,
        "matrix_tiled":  1.00,
        "working_set":   1.00,
        "histogram":     0.50,
        "gather_scatter":0.20,
        "random":        0.00,
        "pointer_chase": 0.00,
    }
    f["access_regularity"]        = reg_scores.get(pattern, 0.5)

    # pattern type flags (one-hot)
    for p in PATTERNS:
        f[f"pat_{p}"] = int(pattern == p)

    # tile effectiveness
    f["tile_size"]                = tile_size
    f["tile_fits_L1"]             = int(tile_size > 0 and (tile_size * tile_size * INT_SIZE) <= L1_CACHE_BYTES)
    f["tile_efficiency"]          = (tile_size / matrix_size) if (matrix_size > 0 and tile_size > 0) else 0.0

    # estimated total accesses
    if pattern in ("sequential", "random", "gather_scatter"):
        f["total_accesses"]       = array_size * iters
    elif pattern == "strided":
        f["total_accesses"]       = (array_size // max(stride, 1)) * iters
    elif pattern in ("matrix_row", "matrix_col", "matrix_tiled"):
        f["total_accesses"]       = matrix_size * matrix_size
    elif pattern == "pointer_chase":
        f["total_accesses"]       = array_size
    elif pattern == "histogram":
        f["total_accesses"]       = array_size
    elif pattern == "working_set":
        f["total_accesses"]       = working_set * iters
    else:
        f["total_accesses"]       = array_size * iters

    return f


def generate_all(target=TARGET_PROGRAMS, out_dir=PROGRAM_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    prog_id = 0

    # Build a balanced pool of parameter combos per pattern
    combos = {
        "sequential":    list(itertools.product(ARRAY_SIZES, ITER_COUNTS)),
        "strided":       list(itertools.product(ARRAY_SIZES, STRIDES, ITER_COUNTS)),
        "random":        list(itertools.product(ARRAY_SIZES, ITER_COUNTS)),
        "matrix_row":    list(itertools.product(MATRIX_SIZES)),
        "matrix_col":    list(itertools.product(MATRIX_SIZES)),
        "matrix_tiled":  list(itertools.product(MATRIX_SIZES, TILE_SIZES)),
        "pointer_chase": list(itertools.product(ARRAY_SIZES)),
        "histogram":     list(itertools.product(ARRAY_SIZES)),
        "gather_scatter":list(itertools.product(ARRAY_SIZES, ITER_COUNTS)),
        "working_set":   list(itertools.product(WORKING_SETS, ITER_COUNTS)),
    }

    per_pattern = target // len(combos)

    for pattern, pool in combos.items():
        # repeat pool to hit per_pattern count
        expanded = []
        while len(expanded) < per_pattern:
            expanded.extend(pool)
        random.shuffle(expanded)
        expanded = expanded[:per_pattern]

        for params_tuple in expanded:
            if pattern == "sequential":
                n, iters = params_tuple
                code = tmpl_sequential(n, iters)
                params = dict(array_size=n, iters=iters)
            elif pattern == "strided":
                n, s, iters = params_tuple
                code = tmpl_strided(n, s, iters)
                params = dict(array_size=n, stride=s, iters=iters)
            elif pattern == "random":
                n, iters = params_tuple
                code = tmpl_random(n, iters)
                params = dict(array_size=n, iters=iters)
            elif pattern == "matrix_row":
                m, = params_tuple
                code = tmpl_matrix_row(m)
                params = dict(matrix_size=m)
            elif pattern == "matrix_col":
                m, = params_tuple
                code = tmpl_matrix_col(m)
                params = dict(matrix_size=m)
            elif pattern == "matrix_tiled":
                m, t = params_tuple
                if t >= m:
                    t = m // 2 or 1
                code = tmpl_matrix_tiled(m, t)
                params = dict(matrix_size=m, tile_size=t)
            elif pattern == "pointer_chase":
                n, = params_tuple
                code = tmpl_pointer_chase(n)
                params = dict(array_size=n)
            elif pattern == "histogram":
                n, = params_tuple
                code = tmpl_histogram(n)
                params = dict(array_size=n)
            elif pattern == "gather_scatter":
                n, iters = params_tuple
                code = tmpl_gather_scatter(n, iters)
                params = dict(array_size=n, iters=iters)
            elif pattern == "working_set":
                ws, iters = params_tuple
                code = tmpl_working_set(ws, iters)
                params = dict(working_set=ws, iters=iters)

            src_path = out_dir / f"prog_{prog_id:06d}.c"
            src_path.write_text(code)

            row = dict(prog_id=prog_id, src_path=str(src_path))
            row.update(features_for(pattern, params))
            records.append(row)
            prog_id += 1

    # Write metadata CSV
    fieldnames = list(records[0].keys())
    with open(METADATA_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)

    print(f"Generated {prog_id} programs → {out_dir}/")
    print(f"Metadata  → {METADATA_FILE}  ({len(fieldnames)} columns)")
    return records


if __name__ == "__main__":
    generate_all()
