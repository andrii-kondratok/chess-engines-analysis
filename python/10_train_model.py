"""
10_train_model.py
-----------------
Train a LightGBM regressor to predict avg_elo from per-game ACPL features.

Inputs  : data/processed/lichess_train.parquet
          data/processed/lichess_rapid.parquet
          data/processed/lichess_classical.parquet
Outputs : models/elo_predictor.pkl
          data/processed/predictions.csv
          data/processed/feature_importance.csv
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

# ── Paths ───────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

INPUTS = [
    DATA_DIR / "lichess_train.parquet",
    DATA_DIR / "lichess_rapid.parquet",
    DATA_DIR / "lichess_classical.parquet",
]

OUT_MODEL       = MODELS_DIR / "elo_predictor.pkl"
OUT_PREDICTIONS = DATA_DIR   / "predictions.csv"
OUT_IMPORTANCE  = DATA_DIR   / "feature_importance.csv"

# ── Features ────────────────────────────────────────────────────────────────────

TARGET = "avg_elo"

FEATURES = [
    "white_acpl", "black_acpl",
    "white_acpl_opening", "black_acpl_opening",
    "white_acpl_middle",  "black_acpl_middle",
    "white_acpl_endgame", "black_acpl_endgame",
    "theory_depth_swing",
    "total_moves",
    "time_control_seconds", "time_control_increment",
    "result_white",
]

DROP_COLS = {"elo_diff", "_tc_type"}   # leakage / internal routing key

# ── Model hyperparams ────────────────────────────────────────────────────────────

LGBM_PARAMS = dict(
    objective        = "regression",      # L2/MSE — less median-biased than L1
    n_estimators     = 5000,
    learning_rate    = 0.03,
    num_leaves       = 127,
    min_child_samples= 50,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 0.1,
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)

EARLY_STOPPING_ROUNDS = 50
RANDOM_STATE          = 42


# ── Elo / TC bucket helpers ──────────────────────────────────────────────────────

ELO_BINS   = list(range(800, 2201, 200))
ELO_LABELS = [f"{lo}-{lo+200}" for lo in ELO_BINS[:-1]]

TC_BINS   = [0,   180,  600, 1800, float("inf")]
TC_LABELS = ["bullet", "blitz", "rapid", "classical"]


def elo_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=ELO_BINS, labels=ELO_LABELS, right=False)


def tc_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=TC_BINS, labels=TC_LABELS, right=False)


# ── Evaluation helpers ───────────────────────────────────────────────────────────

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R²":   r2_score(y_true, y_pred),
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


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & combine ──────────────────────────────────────────────────────
    print("── Loading data ────────────────────────────────────────────")
    frames = []
    for path in INPUTS:
        if not path.exists():
            print(f"  SKIP (not found): {path.name}")
            continue
        df = pd.read_parquet(path)
        print(f"  {path.name:<40}  {len(df):>8,} rows")
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No input parquets found. Run ETL scripts first.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  {'After concat':<40}  {len(combined):>8,} rows")

    combined = combined.drop(columns=[c for c in DROP_COLS if c in combined.columns])
    combined = combined.drop_duplicates()
    print(f"  {'After dedup':<40}  {len(combined):>8,} rows")

    # ── 2. Validate features ───────────────────────────────────────────────────
    missing_feats = [f for f in FEATURES if f not in combined.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")
    if TARGET not in combined.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    df = combined[[TARGET] + FEATURES].copy()
    df = df.dropna(subset=[TARGET])   # target must not be NaN
    print(f"  {'After dropping missing target':<40}  {len(df):>8,} rows")

    X = df[FEATURES]
    y = df[TARGET]

    # ── 3. Split ───────────────────────────────────────────────────────────────
    print("\n── Splitting 70 / 15 / 15 ─────────────────────────────────")
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val   = train_test_split(
        X_tmp, y_tmp, test_size=0.15 / 0.85, random_state=RANDOM_STATE
    )
    print(f"  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # ── 4. Train ───────────────────────────────────────────────────────────────
    print("\n── Training LightGBM ───────────────────────────────────────")
    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set         = [(X_val, y_val)],
        eval_metric      = "mae",
        callbacks        = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Training time  : {elapsed:.1f}s")
    print(f"  Best iteration : {model.best_iteration_}")

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    print("\n── Evaluation ──────────────────────────────────────────────")

    for split_name, X_s, y_s in [
        ("Val ", X_val,  y_val),
        ("Test", X_test, y_test),
    ]:
        pred    = model.predict(X_s, num_iteration=model.best_iteration_)
        metrics = regression_metrics(y_s.values, pred)
        print(f"\n  {split_name}  —  "
              f"MAE={metrics['MAE']:.1f}  "
              f"RMSE={metrics['RMSE']:.1f}  "
              f"R²={metrics['R²']:.4f}")

    # Full test-set predictions for bucket analysis
    y_pred_test = model.predict(X_test, num_iteration=model.best_iteration_)
    pred_df = pd.DataFrame({
        "actual":              y_test.values,
        "predicted":           y_pred_test,
        "error":               y_pred_test - y_test.values,
        "time_control_seconds": X_test["time_control_seconds"].values,
    })
    pred_df["elo_bucket"] = elo_bucket(pred_df["actual"])
    pred_df["tc_bucket"]  = tc_bucket(pred_df["time_control_seconds"])

    bucket_mae(pred_df, "elo_bucket", "Elo bucket")
    bucket_mae(pred_df, "tc_bucket",  "time control")

    # ── 6. Feature importance ──────────────────────────────────────────────────
    print("\n── Feature importance (top 10) ─────────────────────────────")
    importance_df = (
        pd.DataFrame({
            "feature":   FEATURES,
            "gain":      model.booster_.feature_importance(importance_type="gain"),
            "split":     model.booster_.feature_importance(importance_type="split"),
        })
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    gain_total = importance_df["gain"].sum()
    importance_df["gain_pct"] = (importance_df["gain"] / gain_total * 100).round(2)

    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<28}  gain={row['gain_pct']:5.1f}%  splits={int(row['split']):,}")

    # ── 7. Save ────────────────────────────────────────────────────────────────
    print("\n── Saving outputs ──────────────────────────────────────────")

    joblib.dump(model, OUT_MODEL)
    print(f"  Model       → {OUT_MODEL}")

    pred_df.to_csv(OUT_PREDICTIONS, index=False)
    print(f"  Predictions → {OUT_PREDICTIONS}")

    importance_df.to_csv(OUT_IMPORTANCE, index=False)
    print(f"  Importance  → {OUT_IMPORTANCE}")

    print(f"\n{'═' * 56}")
    print(f"  Done. Best MAE on test: "
          f"{mean_absolute_error(pred_df['actual'], pred_df['predicted']):.1f} Elo points")
    print(f"{'═' * 56}")


if __name__ == "__main__":
    main()
