# Cache Behavior Predictor

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Predicts whether a C program is **cache-friendly** or **cache-hostile** using only static code features — no execution required at inference time.

---

## How it works

Memory access patterns (sequential, random, strided, etc.) directly influence CPU cache performance. This project generates synthetic C programs with known patterns, measures their real cache behavior using `perf`, builds a labeled dataset, and trains ML models to predict cache behavior from static features alone.

```
generate_programs.py → compile_programs.sh → run_perf.py → generate_dataset.py → train.py
```

---

## Project structure

```
ML-Project/
├── scripts/
│   ├── generate_programs.py   # Generate ~30k synthetic C programs
│   ├── compile_programs.sh    # Compile all programs with gcc
│   ├── run_perf.py            # Measure cache behavior with perf
│   ├── generate_dataset.py    # Build labeled dataset.csv
│   └── train.py               # Train models and produce results
├── ml/results/                # Output plots and metrics
├── programs/                  # Generated .c source files
├── binaries/                  # Compiled binaries
├── program_metadata.csv       # Static features per program
├── perf_results.csv           # Raw cache measurements
└── dataset.csv                # Final ML-ready dataset
```

---

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
# [package-manager install] gcc linux-perf   # Linux only
```

---

## Usage

Run all commands from the project root.

```bash
# 1. Generate synthetic C programs (~30,000)
python3 scripts/generate_programs.py

# 2. Compile
bash scripts/compile_programs.sh

# 3. Measure cache behavior (requires Linux + perf)
python3 scripts/run_perf.py

# 4. Build dataset
python3 scripts/generate_dataset.py

# 5. Train and evaluate
python3 scripts/train.py
```

---

## Memory access patterns

| Pattern | Cache behavior |
|---|---|
| Sequential | Friendly |
| Matrix row-major | Friendly |
| Matrix tiled | Mostly friendly |
| Working set | Friendly if fits in cache |
| Histogram | Moderate |
| Strided | Depends on stride size |
| Matrix column-major | Hostile |
| Gather/scatter | Hostile |
| Random | Hostile |
| Pointer chasing | Hostile |

---

## Models & results

Five classifiers trained with 5-fold stratified cross-validation:

| Model | Accuracy | F1 |
|---|---|---|
| Random Forest | ~0.91 | ~0.91 |
| Decision Tree | ~0.91 | ~0.91 |
| KNN | ~0.90 | ~0.90 |
| SVM (RBF) | ~0.86 | ~0.85 |
| Logistic Regression | ~0.77 | ~0.77 |

Results are saved to `ml/results/` — confusion matrices, metric bar chart, metrics table, and Random Forest feature importances.

---

## Key design decisions

- **Labeling**: Per-pattern median miss rate split, producing a balanced 50/50 dataset within each pattern class.
- **No data leakage**: Raw perf values (`cache_misses`, `miss_rate`) are excluded from features. The model only sees static structural features at inference time.
- **Feature engineering**: Raw parameters are supplemented with derived features — `fits_in_L1/L2/L3`, `spatial_locality_score`, `temporal_locality_score`, `stride_in_cache_lines` — which carry most of the predictive signal.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
