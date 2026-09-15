"""
train.py
Trains 5 classifiers on dataset.csv, evaluates with stratified k-fold CV,
and outputs metrics + plots to ml/results/.

Models:
  1. Logistic Regression
  2. K-Nearest Neighbours
  3. Support Vector Machine (RBF)
  4. Decision Tree
  5. Random Forest

Run: python3 ml/train.py [--data dataset.csv] [--folds 5]
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ── config ────────────────────────────────────────────────────────────────────
DATASET_CSV  = Path("dataset.csv")
OUT_DIR      = Path("ml/results")
RANDOM_STATE = 42
CV_FOLDS     = 5

NON_FEATURE_COLS = {"prog_id", "pattern", "cache_friendly"}


def load_data(path: Path):
    df = pd.read_csv(path)
    print(f"[train] Loaded {len(df)} rows x {df.shape[1]} cols from {path}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    y = df["cache_friendly"].astype(int)

    X.dropna(axis=1, how="all", inplace=True)
    X = X.loc[:, X.nunique() > 1]
    X.fillna(0, inplace=True)
    X = X.select_dtypes(include=[np.number])

    print(f"[train] Features after cleaning: {X.shape[1]}")
    print(f"[train] Class balance -- friendly: {y.mean():.2%}  "
          f"hostile: {1-y.mean():.2%}")
    return X, y


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                max_depth=12, min_samples_leaf=5,
                random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=None,
                min_samples_leaf=2, n_jobs=-1,
                random_state=RANDOM_STATE)),
        ]),
    }


# ── cross-validation ──────────────────────────────────────────────────────────

def run_cv(models, X, y, folds=CV_FOLDS):
    skf     = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1"]

    cv_results = {}
    for name, pipe in models.items():
        print(f"  [cv] {name} ...", end=" ", flush=True)
        res = cross_validate(pipe, X, y, cv=skf, scoring=scoring,
                             return_train_score=False, n_jobs=-1)
        cv_results[name] = {
            "accuracy":     res["test_accuracy"].mean(),
            "precision":    res["test_precision"].mean(),
            "recall":       res["test_recall"].mean(),
            "f1":           res["test_f1"].mean(),
            "accuracy_std": res["test_accuracy"].std(),
            "f1_std":       res["test_f1"].std(),
        }
        r = cv_results[name]
        print(f"acc={r['accuracy']:.3f}  f1={r['f1']:.3f}")
    return cv_results


# ── OOF predictions (for confusion matrices) ──────────────────────────────────

def get_oof_predictions(models, X, y, folds=CV_FOLDS):
    from sklearn.base import clone
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    predictions = {name: [] for name in models}
    truth_list  = []

    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        truth_list.append(y_te.values)
        for name, pipe in models.items():
            p = clone(pipe)
            p.fit(X_tr, y_tr)
            predictions[name].append(p.predict(X_te))

    y_true_all = np.concatenate(truth_list)
    y_pred_all = {n: np.concatenate(predictions[n]) for n in models}
    return y_true_all, y_pred_all


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_metrics_table(cv_results, out_dir):
    names   = list(cv_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    data    = [[cv_results[n][m] for m in metrics] for n in names]

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1"]
    rows = [[n] + [f"{v:.4f}" for v in row] for n, row in zip(names, data)]

    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    # Highlight best value per metric column
    col_max_idx = [np.argmax([float(rows[r][c]) for r in range(len(rows))])
                   for c in range(1, 5)]
    for col, best_row in enumerate(col_max_idx):
        tbl[(best_row + 1, col + 1)].set_facecolor("#d4edda")

    ax.set_title("Cross-Validation Metrics (5-Fold, mean)", pad=12, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] metrics_table.png")


def plot_metric_bars(cv_results, out_dir):
    names   = list(cv_results.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(names))
    w = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (m, lbl) in enumerate(zip(metrics, labels)):
        vals = [cv_results[n][m] for n in names]
        ax.bar(x + i * w, vals, w, label=lbl, alpha=0.85)

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison -- All Metrics (5-Fold CV)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "metric_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] metric_bars.png")


def plot_confusion_matrices(y_true, y_pred_all, out_dir):
    names = list(y_pred_all.keys())
    n     = len(names)
    cols  = 3
    rows  = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, name in enumerate(names):
        cm = confusion_matrix(y_true, y_pred_all[name])
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[i],
                    cmap="Blues", cbar=False,
                    xticklabels=["hostile", "friendly"],
                    yticklabels=["hostile", "friendly"])
        acc = accuracy_score(y_true, y_pred_all[name])
        f1  = f1_score(y_true, y_pred_all[name])
        axes[i].set_title(f"{name}\nacc={acc:.3f}  f1={f1:.3f}", fontsize=9)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion Matrices (5-Fold OOF predictions)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] confusion_matrices.png")


def plot_feature_importance(models, X, y, out_dir, top_n=20):
    fig, ax = plt.subplots(figsize=(9, 6))

    pipe = models["Random Forest"]
    pipe.fit(X, y)
    clf  = pipe.named_steps["clf"]
    imps = clf.feature_importances_
    idxs = np.argsort(imps)[::-1][:top_n]

    ax.barh(range(top_n), imps[idxs][::-1], color="#4C72B0", alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([X.columns[i] for i in idxs][::-1], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Random Forest -- Top {top_n} Features")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[plot] feature_importance.png")


def save_metrics_csv(cv_results, out_dir):
    rows = []
    for name, m in cv_results.items():
        rows.append({
            "model":        name,
            "accuracy":     round(m["accuracy"],     4),
            "precision":    round(m["precision"],    4),
            "recall":       round(m["recall"],       4),
            "f1":           round(m["f1"],           4),
            "accuracy_std": round(m["accuracy_std"], 4),
            "f1_std":       round(m["f1_std"],       4),
        })
    pd.DataFrame(rows).to_csv(out_dir / "cv_metrics.csv", index=False)
    print("[train] cv_metrics.csv saved")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  type=Path, default=DATASET_CSV)
    ap.add_argument("--folds", type=int,  default=CV_FOLDS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data)
    models = build_models()

    print(f"\n[train] Running {args.folds}-fold stratified CV ...")
    cv_results = run_cv(models, X, y, folds=args.folds)

    print("\n[train] Getting OOF predictions for confusion matrices ...")
    y_true, y_pred_all = get_oof_predictions(models, X, y, folds=args.folds)

    print("\n[plots] Generating ...")
    plot_metrics_table(cv_results, OUT_DIR)
    plot_metric_bars(cv_results, OUT_DIR)
    plot_confusion_matrices(y_true, y_pred_all, OUT_DIR)
    plot_feature_importance(models, X, y, OUT_DIR)

    save_metrics_csv(cv_results, OUT_DIR)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    sorted_models = sorted(cv_results.items(),
                           key=lambda x: x[1]["f1"], reverse=True)
    for rank, (name, m) in enumerate(sorted_models, 1):
        print(f"  {rank}. {name:<22}  "
              f"F1={m['f1']:.4f}  "
              f"Acc={m['accuracy']:.4f}  "
              f"(+-{m['f1_std']:.4f})")
    print("=" * 60)
    print(f"[train] All outputs -> {OUT_DIR}/")


if __name__ == "__main__":
    main()

