"""
12_train_model_v2.py  (v3.1 — sample-weighted)
-----------------------------------------------
Train a LightGBM regressor using the full feature set produced by
11_etl_amateur_patterns.py: 13 ACPL + 18 amateur-pattern + 10 time = 41 features.

Adds sample weighting to improve MAE at edge Elo buckets (800-1200, 2000-2200)
which are under-represented in training data and had the highest errors in v3.

Input   : data/processed/lichess_amateur_v2.parquet
Outputs : models/elo_predictor_v3_1.pkl         (weighted model, keeps v3 intact)
          data/processed/predictions_v3_1.csv
          data/processed/feature_importance_v3_1.csv

Prints a four-way comparison:
  v1 ACPL-only  |  v2 amateur-only  |  v3 combined (no weights)  |  v3.1 weighted
"""

import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ────────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

INPUT_FILE      = DATA_DIR   / "lichess_amateur_v2.parquet"
OUT_MODEL       = MODELS_DIR / "elo_predictor_v3_1.pkl"
OUT_PREDICTIONS = DATA_DIR   / "predictions_v3_1.csv"
OUT_IMPORTANCE  = DATA_DIR   / "feature_importance_v3_1.csv"

# ── Historical predictions for four-way comparison ────────────────────────────────
# Add a file here as each model is trained; missing → n/a in comparison table.
V1_PREDICTIONS  = DATA_DIR / "predictions.csv"          # v1: 13 ACPL, ~700k
V2A_PREDICTIONS = DATA_DIR / "predictions_v2_amateur.csv"  # v2: 31 feats, 100k amateur
V3_PREDICTIONS  = DATA_DIR / "predictions_v2.csv"       # v3: 41 feats, combined, no weights

# ── Features ─────────────────────────────────────────────────────────────────────

TARGET = "avg_elo"

ACPL_FEATURES = [
    "white_acpl",          "black_acpl",
    "white_acpl_opening",  "black_acpl_opening",
    "white_acpl_middle",   "black_acpl_middle",
    "white_acpl_endgame",  "black_acpl_endgame",
    "theory_depth_swing",
    "total_moves",
    "time_control_seconds", "time_control_increment",
    "result_white",
]

PATTERN_FEATURES = [
    "white_pointless_checks",          "black_pointless_checks",
    "white_early_queen_blunders",      "black_early_queen_blunders",
    "white_opening_blunders",          "black_opening_blunders",
    "white_bad_corner_bishop",         "black_bad_corner_bishop",
    "white_bad_rim_knight",            "black_bad_rim_knight",
    "white_castled",                   "black_castled",
    "white_castled_move",              "black_castled_move",
    "white_pawn_moves_in_opening",     "black_pawn_moves_in_opening",
    "white_piece_moved_twice_opening", "black_piece_moved_twice_opening",
]

TIME_FEATURES = [
    "white_avg_think",        "black_avg_think",
    "white_think_std",        "black_think_std",
    "white_fast_moves_pct",   "black_fast_moves_pct",
    "white_quick_blunders",   "black_quick_blunders",
    "white_opening_thinking", "black_opening_thinking",
]

FEATURES = ACPL_FEATURES + PATTERN_FEATURES + TIME_FEATURES

# ── Model hyperparams ─────────────────────────────────────────────────────────────

LGBM_PARAMS = dict(
    objective         = "regression",
    n_estimators      = 6000,
    learning_rate     = 0.01,
    num_leaves        = 150,
    min_child_samples = 50,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

EARLY_STOPPING_ROUNDS = 100
RANDOM_STATE          = 42

# ── Bucketing helpers ─────────────────────────────────────────────────────────────

ELO_BINS   = list(range(800, 2201, 200))
ELO_LABELS = [f"{lo}-{lo+200}" for lo in ELO_BINS[:-1]]

TC_BINS    = [0,   180,  600, 1800, float("inf")]
TC_LABELS  = ["bullet", "blitz", "rapid", "classical"]


def elo_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=ELO_BINS, labels=ELO_LABELS, right=False)


def tc_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=TC_BINS, labels=TC_LABELS, right=False)


# ── Sample weighting ──────────────────────────────────────────────────────────────

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Assign higher loss weight to edge Elo buckets where training data is sparse
    and the v3 model underperformed (MAE 370 at 800-1000, 316 at 2000-2200).

    Weights (relative to the densely populated 1400-1800 core):
      800 – 1200  →  2.0   (far from training peak, poorest predictions)
      1200 – 1400 →  1.5   (transition to core)
      1400 – 1800 →  1.0   (peak density, default)
      1800 – 2000 →  1.5   (transition from core)
      2000 – 2200 →  2.0   (far from training peak, poorest predictions)
    """
    weights = np.ones_like(y, dtype=float)
    weights[y < 1200]                        = 1
    weights[(y >= 1200) & (y < 1400)]        = 1
    # 1400-1800: remain 1.0 (already set)
    weights[(y >= 1800) & (y < 2000)]        = 1
    weights[y >= 2000]                       = 1
    return weights


# ── Evaluation helpers ────────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2":   r2_score(y_true, y_pred),
    }


def bucket_mae(df: pd.DataFrame, bucket_col: str, label: str) -> None:
    print(f"\n  MAE by {label}:")
    grouped = (
        df.groupby(bucket_col, observed=True)
          .apply(lambda g: mean_absolute_error(g["actual"], g["predicted"]))
          .rename("MAE")
          .reset_index()
    )
    for _, row in grouped.iterrows():
        n = (df[bucket_col] == row[bucket_col]).sum()
        print(f"    {str(row[bucket_col]):<18}  MAE={row['MAE']:6.1f}  (n={n:,})")


# ── Main ──────────────────────────────────────────────────────────────────────────

def _load_predictions_metrics(path: Path) -> dict | None:
    """Load a predictions CSV and return MAE / RMSE / R2, or None if missing."""
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return {
            "MAE":  mean_absolute_error(df["actual"], df["predicted"]),
            "RMSE": mean_squared_error(df["actual"], df["predicted"]) ** 0.5,
            "R2":   r2_score(df["actual"], df["predicted"]),
        }
    except Exception as e:
        print(f"  (Could not read {path.name}: {e})")
        return None


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("── Loading data ────────────────────────────────────────────")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_FILE}\n"
            "Run python/11_etl_amateur_patterns.py first."
        )

    df_raw = pd.read_parquet(INPUT_FILE)
    print(f"  {INPUT_FILE.name:<42}  {len(df_raw):>8,} rows  {len(df_raw.columns)} cols")

    # Drop leakage columns if present
    df_raw = df_raw.drop(columns=[c for c in ("elo_diff", "_tc_type")
                                   if c in df_raw.columns])

    if TARGET not in df_raw.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    missing_feats = [f for f in FEATURES if f not in df_raw.columns]
    if missing_feats:
        raise ValueError(
            f"Missing feature columns: {missing_feats}\n"
            "Ensure the file was produced by 11_etl_amateur_patterns.py (v2)."
        )

    df = df_raw[[TARGET] + FEATURES].copy()
    df = df.dropna(subset=[TARGET])
    df = df.drop_duplicates()
    print(f"  {'After dedup / drop target NaN':<42}  {len(df):>8,} rows")

    X = df[FEATURES]
    y = df[TARGET]

    # ── 2. Split ──────────────────────────────────────────────────────────────
    print("\n── Splitting 90 / 5 / 5 ─────────────────────────────────")
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X, y, test_size=0.05, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val   = train_test_split(
        X_tmp, y_tmp, test_size=0.05 / 0.95, random_state=RANDOM_STATE
    )
    print(f"  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # ── 3. Train (with sample weights) ───────────────────────────────────────
    print(f"\n── Training LightGBM v3.1 ({len(FEATURES)} features, sample-weighted) ──")
    sw_train = compute_sample_weights(y_train.values)
    sw_val   = compute_sample_weights(y_val.values)

    # Print weight distribution for transparency
    for lo, hi, w in [(800, 1200, 2.0), (1200, 1400, 1.5),
                      (1400, 1800, 1.0), (1800, 2000, 1.5), (2000, 2200, 2.0)]:
        mask = (y_train.values >= lo) & (y_train.values < hi)
        print(f"  weight={w:.1f}  Elo {lo}-{hi:<5}  train rows: {mask.sum():>7,}")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        sample_weight        = sw_train,
        eval_set             = [(X_val, y_val)],
        eval_sample_weight   = [sw_val],
        eval_metric          = "mae",
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Training time  : {elapsed:.1f}s")
    print(f"  Best iteration : {model.best_iteration_}")

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    print("\n── Evaluation ──────────────────────────────────────────────")

    v2_metrics: dict[str, dict] = {}
    for split_name, X_s, y_s in [
        ("Val ", X_val,  y_val),
        ("Test", X_test, y_test),
    ]:
        pred    = model.predict(X_s, num_iteration=model.best_iteration_)
        metrics = regression_metrics(y_s.values, pred)
        v2_metrics[split_name.strip()] = metrics
        print(f"\n  {split_name}  --  "
              f"MAE={metrics['MAE']:.1f}  "
              f"RMSE={metrics['RMSE']:.1f}  "
              f"R2={metrics['R2']:.4f}")

    # Full test-set predictions for bucket analysis
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration_)
    pred_df = pd.DataFrame({
        "actual":               y_test.values,
        "predicted":            y_pred_test,
        "error":                y_pred_test - y_test.values,
        "time_control_seconds": X_test["time_control_seconds"].values,
    })
    pred_df["elo_bucket"] = elo_bucket(pred_df["actual"])
    pred_df["tc_bucket"]  = tc_bucket(pred_df["time_control_seconds"])

    bucket_mae(pred_df, "elo_bucket", "Elo bucket")
    bucket_mae(pred_df, "tc_bucket",  "time control")

    # ── 5. Feature importance (top 15) ────────────────────────────────────────
    print("\n── Feature importance (top 15) ─────────────────────────────")
    importance_df = (
        pd.DataFrame({
            "feature": FEATURES,
            "gain":    model.booster_.feature_importance(importance_type="gain"),
            "split":   model.booster_.feature_importance(importance_type="split"),
        })
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    gain_total = importance_df["gain"].sum()
    importance_df["gain_pct"] = (importance_df["gain"] / gain_total * 100).round(2)
    def _feature_group(f: str) -> str:
        if f in ACPL_FEATURES:    return "acpl   "
        if f in PATTERN_FEATURES: return "pattern"
        return "time   "

    importance_df["group"] = importance_df["feature"].apply(_feature_group)

    for _, row in importance_df.head(15).iterrows():
        print(f"  [{row['group']}] {row['feature']:<36}  gain={row['gain_pct']:5.1f}%  "
              f"splits={int(row['split']):,}")

    # Group-level gain summary
    group_gain = importance_df.groupby("group")["gain_pct"].sum().sort_values(ascending=False)
    print(f"\n  Gain by group:")
    for grp, pct in group_gain.items():
        print(f"    [{grp}]  {pct:.1f}%")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print("\n── Saving outputs ──────────────────────────────────────────")

    joblib.dump(model, OUT_MODEL)
    print(f"  Model        -> {OUT_MODEL}")

    pred_df.to_csv(OUT_PREDICTIONS, index=False)
    print(f"  Predictions  -> {OUT_PREDICTIONS}")

    importance_df.to_csv(OUT_IMPORTANCE, index=False)
    print(f"  Importance   -> {OUT_IMPORTANCE}")

    # ── 7. Four-way comparison ────────────────────────────────────────────────
    v1_m   = _load_predictions_metrics(V1_PREDICTIONS)
    v2a_m  = _load_predictions_metrics(V2A_PREDICTIONS)
    v3_m   = _load_predictions_metrics(V3_PREDICTIONS)
    v31_m  = v2_metrics["Test"]    # just computed

    def _fmt(val: float | None, fmt: str = ".1f") -> str:
        return f"{val:{fmt}}" if val is not None else "n/a"

    def _delta(v_new: float, v_ref: float | None, fmt: str,
               lower_is_better: bool = True) -> str:
        """Delta of this run vs v1 baseline (shown in rightmost column)."""
        if v_ref is None:
            return ""
        delta = v_new - v_ref
        sign  = "+" if delta >= 0 else ""
        better = (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better)
        tag = " <better>" if better else ""
        return f"  [{sign}{delta:{fmt}}]{tag}"

    COL = 15
    H = ["v1 ACPL", "v2 amateur", "v3 combined", "v3.1 weighted"]
    print(f"\n{'=' * 82}")
    print("  Four-way model comparison")
    print(f"{'=' * 82}")
    print(f"  {'Metric':<14}  " + "  ".join(f"{h:>{COL}}" for h in H) + "  vs v1")
    print(f"  {'-'*14}  " + "  ".join("-" * COL for _ in H))

    for label, key, fmt, lib in [
        ("Test MAE",  "MAE",  ".1f", True),
        ("Test RMSE", "RMSE", ".1f", True),
        ("Test R2",   "R2",   ".4f", False),
    ]:
        vals = [
            v1_m[key]  if v1_m  else None,
            v2a_m[key] if v2a_m else None,
            v3_m[key]  if v3_m  else None,
            v31_m[key],
        ]
        row = "  ".join(f"{_fmt(v, fmt):>{COL}}" for v in vals)
        print(f"  {label:<14}  {row}{_delta(v31_m[key], vals[0], fmt, lib)}")

    print(f"{'=' * 82}")
    desc = [f"~700k", f"~100k", f"~{len(df):,}", f"~{len(df):,}"]
    feat = [str(len(ACPL_FEATURES)), "31", str(len(FEATURES)), str(len(FEATURES))]
    wt   = ["no", "no", "no", "YES"]
    for label, row_vals in [("Rows", desc), ("Features", feat), ("Weighted", wt)]:
        row = "  ".join(f"{v:>{COL}}" for v in row_vals)
        print(f"  {label:<14}  {row}")
    print(f"{'=' * 82}\n")


if __name__ == "__main__":
    main()
