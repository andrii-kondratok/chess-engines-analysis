"""
12_train_model_v4_stacking.py  (v4 — two-level stacking)
---------------------------------------------------------
Level-1: two independent LightGBM regressors
  • acpl_model   : ACPL_FEATURES + TIME_FEATURES  (23 feats)
  • pattern_model: PATTERN_FEATURES               (18 feats)
 
Level-2: meta LightGBM that sees
  • pred_acpl, pred_pattern          (level-1 outputs)
  • time_control_seconds, total_moves (lightweight game-context)
  ... and learns how much to trust each level-1 model per sample.
 
Key design choices:
  - Level-1 models are trained on X_train only.
  - Meta features for X_train are generated via 5-fold OOF to avoid
    leakage (each fold's predictions come from a model that never saw
    those rows during training).
  - X_val / X_test always use the full level-1 models (no OOF needed).
  - Sample weights are applied at each level.
 
Input   : data/processed/lichess_amateur_v2.parquet
Outputs : models/elo_predictor_v4_acpl.pkl
          models/elo_predictor_v4_pattern.pkl
          models/elo_predictor_v4_meta.pkl
          data/processed/predictions_v4.csv
          data/processed/feature_importance_v4_acpl.csv
          data/processed/feature_importance_v4_pattern.csv
          data/processed/feature_importance_v4_meta.csv
 
Prints a five-way comparison:
  v1 ACPL-only | v2 amateur-only | v3 combined | v3.1 weighted | v4 stacking
"""
 
import time
import warnings
from pathlib import Path
 
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
 
warnings.filterwarnings("ignore", category=UserWarning)
 
# ── Paths ────────────────────────────────────────────────────────────────────────
 
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
 
INPUT_FILE            = DATA_DIR   / "lichess_amateur_v2.parquet"
OUT_MODEL_ACPL        = MODELS_DIR / "elo_predictor_v4_acpl.pkl"
OUT_MODEL_PATTERN     = MODELS_DIR / "elo_predictor_v4_pattern.pkl"
OUT_MODEL_META        = MODELS_DIR / "elo_predictor_v4_meta.pkl"
OUT_PREDICTIONS       = DATA_DIR   / "predictions_v4.csv"
OUT_IMPORTANCE_ACPL   = DATA_DIR   / "feature_importance_v4_acpl.csv"
OUT_IMPORTANCE_PATTERN= DATA_DIR   / "feature_importance_v4_pattern.csv"
OUT_IMPORTANCE_META   = DATA_DIR   / "feature_importance_v4_meta.csv"
 
# ── Historical predictions for five-way comparison ────────────────────────────────
V1_PREDICTIONS  = DATA_DIR / "predictions.csv"
V2A_PREDICTIONS = DATA_DIR / "predictions_v2_amateur.csv"
V3_PREDICTIONS  = DATA_DIR / "predictions_v2.csv"
V31_PREDICTIONS = DATA_DIR / "predictions_v3_1.csv"
 
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
 
TIME_FEATURES = [
    "white_avg_think",        "black_avg_think",
    "white_think_std",        "black_think_std",
    "white_fast_moves_pct",   "black_fast_moves_pct",
    "white_quick_blunders",   "black_quick_blunders",
    "white_opening_thinking", "black_opening_thinking",
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
 
# All features needed to load from parquet
FEATURES = ACPL_FEATURES + TIME_FEATURES + PATTERN_FEATURES
 
L1_ACPL_FEATS    = ACPL_FEATURES + TIME_FEATURES   # 23 feats → acpl model
L1_PATTERN_FEATS = PATTERN_FEATURES                 # 18 feats → pattern model
 
# Meta-model context features passed alongside level-1 predictions
META_CONTEXT_FEATS = ["time_control_seconds", "total_moves"]
META_FEATURES      = ["pred_acpl", "pred_pattern"] + META_CONTEXT_FEATS
 
OOF_FOLDS = 5
 
# ── Hyperparams ───────────────────────────────────────────────────────────────────
 
_COMMON = dict(
    objective         = "regression",
    learning_rate     = 0.01,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)
 
ACPL_PARAMS = dict(
    **_COMMON,
    n_estimators      = 6000,
    num_leaves        = 150,
    min_child_samples = 50,
)
 
PATTERN_PARAMS = dict(
    **_COMMON,
    n_estimators      = 4000,
    num_leaves        = 100,
    min_child_samples = 50,
)
 
# Meta-model is intentionally shallow — it only needs to blend two signals
META_PARAMS = dict(
    **_COMMON,
    n_estimators      = 2000,
    num_leaves        = 31,
    min_child_samples = 30,
)
 
EARLY_STOPPING_ROUNDS = 100
RANDOM_STATE          = 42
 
# ── Bucketing helpers ─────────────────────────────────────────────────────────────
 
ELO_BINS   = list(range(800, 2201, 200))
ELO_LABELS = [f"{lo}-{lo+200}" for lo in ELO_BINS[:-1]]
TC_BINS    = [0, 180, 600, 1800, float("inf")]
TC_LABELS  = ["bullet", "blitz", "rapid", "classical"]
 
 
def elo_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=ELO_BINS, labels=ELO_LABELS, right=False)
 
 
def tc_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=TC_BINS, labels=TC_LABELS, right=False)
 
 
# ── Sample weighting ──────────────────────────────────────────────────────────────
 
def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Higher weight for edge Elo buckets that are sparse and harder to predict."""
    weights = np.ones_like(y, dtype=float)
    weights[y < 1200]                        = 2.0
    weights[(y >= 1200) & (y < 1400)]        = 1.5
    weights[(y >= 1800) & (y < 2000)]        = 1.5
    weights[y >= 2000]                       = 2.0
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
 
 
def save_importance(model: lgb.LGBMRegressor, feat_names: list[str],
                    path: Path, label: str) -> None:
    imp = pd.DataFrame({
        "feature": feat_names,
        "gain":    model.booster_.feature_importance(importance_type="gain"),
        "split":   model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    total = imp["gain"].sum()
    imp["gain_pct"] = (imp["gain"] / total * 100).round(2)
    print(f"\n── Feature importance — {label} (top 10) ────────────────────")
    for _, row in imp.head(10).iterrows():
        print(f"  {row['feature']:<36}  gain={row['gain_pct']:5.1f}%  "
              f"splits={int(row['split']):,}")
    imp.to_csv(path, index=False)
    print(f"  Saved → {path}")
 
 
def _load_predictions_metrics(path: Path) -> dict | None:
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
 
 
# ── Level-1 training helper ───────────────────────────────────────────────────────
 
def train_l1(
    name: str,
    feats: list[str],
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[lgb.LGBMRegressor, float]:
    """Train a level-1 model, return (fitted model, elapsed seconds)."""
    sw_train = compute_sample_weights(y_train.values)
    sw_val   = compute_sample_weights(y_val.values)
 
    model = lgb.LGBMRegressor(**params)
    t0 = time.perf_counter()
    model.fit(
        X_train[feats], y_train,
        sample_weight      = sw_train,
        eval_set           = [(X_val[feats], y_val)],
        eval_sample_weight = [sw_val],
        eval_metric        = "mae",
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    elapsed = time.perf_counter() - t0
    print(f"  [{name}]  time={elapsed:.1f}s  best_iter={model.best_iteration_}")
    return model, elapsed
 
 
# ── OOF predictions for meta-model training ───────────────────────────────────────
 
def make_oof_predictions(
    feats: list[str],
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> np.ndarray:
    """
    5-fold OOF: each fold trains on 4/5 of X_train and predicts the held-out 1/5.
    This ensures the meta-model's training inputs are never seen by the models
    that produced them — i.e. no leakage from level-1 into level-2.
    """
    oof = np.zeros(len(X_train))
    kf  = KFold(n_splits=OOF_FOLDS, shuffle=True, random_state=RANDOM_STATE)
 
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_vl = X_train[feats].iloc[tr_idx], X_train[feats].iloc[val_idx]
        y_tr, y_vl = y_train.iloc[tr_idx],         y_train.iloc[val_idx]
        sw_tr = compute_sample_weights(y_tr.values)
        sw_vl = compute_sample_weights(y_vl.values)
 
        fold_model = lgb.LGBMRegressor(**params)
        fold_model.fit(
            X_tr, y_tr,
            sample_weight      = sw_tr,
            eval_set           = [(X_vl, y_vl)],
            eval_sample_weight = [sw_vl],
            eval_metric        = "mae",
            callbacks = [
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(period=500),
            ],
        )
        oof[val_idx] = fold_model.predict(
            X_vl, num_iteration=fold_model.best_iteration_
        )
        print(f"    fold {fold}/{OOF_FOLDS}  best_iter={fold_model.best_iteration_}  "
              f"MAE={mean_absolute_error(y_vl, oof[val_idx]):.1f}")
 
    return oof
 
 
# ── Main ──────────────────────────────────────────────────────────────────────────
 
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
 
    df_raw = df_raw.drop(columns=[c for c in ("elo_diff", "_tc_type")
                                   if c in df_raw.columns])
    if TARGET not in df_raw.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")
 
    missing_feats = [f for f in FEATURES if f not in df_raw.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns: {missing_feats}")
 
    df = df_raw[[TARGET] + FEATURES].copy()
    df = df.dropna(subset=[TARGET]).drop_duplicates()
    print(f"  {'After dedup / drop target NaN':<42}  {len(df):>8,} rows")
 
    X = df[FEATURES]
    y = df[TARGET]
 
    # ── 2. Split 90 / 5 / 5 ──────────────────────────────────────────────────
    print("\n── Splitting 90 / 5 / 5 ─────────────────────────────────")
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
        X, y, test_size=0.05, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.05 / 0.95, random_state=RANDOM_STATE
    )
    print(f"  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")
 
    # ── 3. Level-1: OOF predictions for meta training ────────────────────────
    print(f"\n── Level-1 OOF ({OOF_FOLDS}-fold) — ACPL model ─────────────────────")
    t_oof0 = time.perf_counter()
    oof_acpl = make_oof_predictions(L1_ACPL_FEATS, ACPL_PARAMS, X_train, y_train)
    print(f"\n── Level-1 OOF ({OOF_FOLDS}-fold) — Pattern model ──────────────────")
    oof_pattern = make_oof_predictions(L1_PATTERN_FEATS, PATTERN_PARAMS, X_train, y_train)
    print(f"\n  Total OOF time: {time.perf_counter() - t_oof0:.1f}s")
 
    oof_mae_acpl    = mean_absolute_error(y_train, oof_acpl)
    oof_mae_pattern = mean_absolute_error(y_train, oof_pattern)
    print(f"\n  OOF MAE — acpl={oof_mae_acpl:.1f}  pattern={oof_mae_pattern:.1f}")
 
    # ── 4. Level-1: full models (trained on all of X_train) ──────────────────
    print("\n── Level-1 full training ────────────────────────────────────")
    acpl_model,    _ = train_l1("acpl   ", L1_ACPL_FEATS,    ACPL_PARAMS,
                                 X_train, y_train, X_val, y_val)
    pattern_model, _ = train_l1("pattern", L1_PATTERN_FEATS, PATTERN_PARAMS,
                                 X_train, y_train, X_val, y_val)
 
    def l1_predict(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p_acpl    = acpl_model.predict(
            X[L1_ACPL_FEATS],    num_iteration=acpl_model.best_iteration_)
        p_pattern = pattern_model.predict(
            X[L1_PATTERN_FEATS], num_iteration=pattern_model.best_iteration_)
        return p_acpl, p_pattern
 
    # Val & test predictions from full level-1 models
    val_acpl,  val_pattern  = l1_predict(X_val)
    test_acpl, test_pattern = l1_predict(X_test)
 
    print("\n  Level-1 val MAE:")
    print(f"    acpl   : {mean_absolute_error(y_val, val_acpl):.1f}")
    print(f"    pattern: {mean_absolute_error(y_val, val_pattern):.1f}")
 
    # ── 5. Build meta-model feature sets ─────────────────────────────────────
    def meta_df(pred_acpl, pred_pattern, X_ref: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "pred_acpl":            pred_acpl,
            "pred_pattern":         pred_pattern,
            "time_control_seconds": X_ref["time_control_seconds"].values,
            "total_moves":          X_ref["total_moves"].values,
        })
 
    # Training: use OOF (no leakage)
    X_meta_train = meta_df(oof_acpl, oof_pattern, X_train)
    # Val / test: use full level-1 model predictions
    X_meta_val   = meta_df(val_acpl,  val_pattern,  X_val)
    X_meta_test  = meta_df(test_acpl, test_pattern, X_test)
 
    # ── 6. Level-2: meta-model ────────────────────────────────────────────────
    print("\n── Level-2 meta-model training ──────────────────────────────")
    sw_meta_train = compute_sample_weights(y_train.values)
    sw_meta_val   = compute_sample_weights(y_val.values)
 
    meta_model = lgb.LGBMRegressor(**META_PARAMS)
    t0 = time.perf_counter()
    meta_model.fit(
        X_meta_train, y_train,
        sample_weight      = sw_meta_train,
        eval_set           = [(X_meta_val, y_val)],
        eval_sample_weight = [sw_meta_val],
        eval_metric        = "mae",
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"\n  Meta training time : {time.perf_counter() - t0:.1f}s")
    print(f"  Meta best iteration: {meta_model.best_iteration_}")
 
    # ── 7. Evaluate stacking ensemble ─────────────────────────────────────────
    print("\n── Evaluation ──────────────────────────────────────────────")
 
    v4_metrics: dict[str, dict] = {}
    for split_name, X_meta_s, y_s in [
        ("Val ", X_meta_val,  y_val),
        ("Test", X_meta_test, y_test),
    ]:
        pred    = meta_model.predict(X_meta_s, num_iteration=meta_model.best_iteration_)
        metrics = regression_metrics(y_s.values, pred)
        v4_metrics[split_name.strip()] = metrics
        print(f"\n  {split_name}  --  "
              f"MAE={metrics['MAE']:.1f}  "
              f"RMSE={metrics['RMSE']:.1f}  "
              f"R2={metrics['R2']:.4f}")
 
    # Bucket analysis on test set
    y_pred_test = meta_model.predict(X_meta_test, num_iteration=meta_model.best_iteration_)
    pred_df = pd.DataFrame({
        "actual":               y_test.values,
        "predicted":            y_pred_test,
        "error":                y_pred_test - y_test.values,
        "pred_acpl":            test_acpl,
        "pred_pattern":         test_pattern,
        "time_control_seconds": X_test["time_control_seconds"].values,
    })
    pred_df["elo_bucket"] = elo_bucket(pred_df["actual"])
    pred_df["tc_bucket"]  = tc_bucket(pred_df["time_control_seconds"])
 
    bucket_mae(pred_df, "elo_bucket", "Elo bucket")
    bucket_mae(pred_df, "tc_bucket",  "time control")
 
    # ── 8. Feature importance ─────────────────────────────────────────────────
    save_importance(acpl_model,    L1_ACPL_FEATS,    OUT_IMPORTANCE_ACPL,    "ACPL model")
    save_importance(pattern_model, L1_PATTERN_FEATS, OUT_IMPORTANCE_PATTERN, "Pattern model")
    save_importance(meta_model,    META_FEATURES,    OUT_IMPORTANCE_META,    "Meta model")
 
    # Meta gain% reveals how much the meta-model trusts each l1 prediction
    print("\n  Meta-model gain breakdown (how much each input is trusted):")
    meta_imp = pd.read_csv(OUT_IMPORTANCE_META)
    for _, row in meta_imp.iterrows():
        print(f"    {row['feature']:<26}  gain={row['gain_pct']:5.1f}%")
 
    # ── 9. Save models & predictions ─────────────────────────────────────────
    print("\n── Saving outputs ──────────────────────────────────────────")
    joblib.dump(acpl_model,    OUT_MODEL_ACPL)
    joblib.dump(pattern_model, OUT_MODEL_PATTERN)
    joblib.dump(meta_model,    OUT_MODEL_META)
    print(f"  ACPL model    -> {OUT_MODEL_ACPL}")
    print(f"  Pattern model -> {OUT_MODEL_PATTERN}")
    print(f"  Meta model    -> {OUT_MODEL_META}")
 
    pred_df.to_csv(OUT_PREDICTIONS, index=False)
    print(f"  Predictions   -> {OUT_PREDICTIONS}")
 
    # ── 10. Five-way comparison ───────────────────────────────────────────────
    v1_m  = _load_predictions_metrics(V1_PREDICTIONS)
    v2a_m = _load_predictions_metrics(V2A_PREDICTIONS)
    v3_m  = _load_predictions_metrics(V3_PREDICTIONS)
    v31_m = _load_predictions_metrics(V31_PREDICTIONS)
    v4_m  = v4_metrics["Test"]
 
    def _fmt(val, fmt=".1f"):
        return f"{val:{fmt}}" if val is not None else "n/a"
 
    def _delta(v_new, v_ref, fmt, lower_is_better=True):
        if v_ref is None:
            return ""
        delta = v_new - v_ref
        sign  = "+" if delta >= 0 else ""
        better = (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better)
        tag = " <better>" if better else ""
        return f"  [{sign}{delta:{fmt}}]{tag}"
 
    COL = 14
    H = ["v1 ACPL", "v2 amateur", "v3 combined", "v3.1 weighted", "v4 stacking"]
    print(f"\n{'=' * 95}")
    print("  Five-way model comparison")
    print(f"{'=' * 95}")
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
            v31_m[key] if v31_m else None,
            v4_m[key],
        ]
        row = "  ".join(f"{_fmt(v, fmt):>{COL}}" for v in vals)
        print(f"  {label:<14}  {row}{_delta(v4_m[key], vals[0], fmt, lib)}")
 
    print(f"{'=' * 95}")
    desc = ["~700k", "~100k", f"~{len(df):,}", f"~{len(df):,}", f"~{len(df):,}"]
    feat = [str(len(ACPL_FEATURES)), "31", "41", "41",
            f"{len(L1_ACPL_FEATS)}+{len(L1_PATTERN_FEATS)}+meta"]
    arch = ["single", "single", "single", "single", "stacking"]
    for label, row_vals in [("Rows", desc), ("Features", feat), ("Architecture", arch)]:
        row = "  ".join(f"{v:>{COL}}" for v in row_vals)
        print(f"  {label:<14}  {row}")
    print(f"{'=' * 95}\n")
 
 
if __name__ == "__main__":
    main()
 