import os
import pandas as pd
import numpy as np
import pygeohash as gh
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import joblib
import warnings
warnings.filterwarnings("ignore")

TRAIN_PATH = "e88186124ec611f1/dataset/train.csv"
TEST_PATH  = "e88186124ec611f1/dataset/test.csv"
OUT_PATH   = "submission.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

target = train["demand"].copy()
train.drop(columns=["demand"], inplace=True)

combined = pd.concat([train, test], axis=0).reset_index(drop=True)
n_train  = len(train)

# ── Feature Engineering ───────────────────────────────────────────────────────
def decode_geohash(gh_str):
    try:
        lat, lng = gh.decode(gh_str)
        return lat, lng
    except Exception:
        return np.nan, np.nan

coords = combined["geohash"].apply(decode_geohash)
combined["lat"] = coords.apply(lambda x: x[0])
combined["lng"] = coords.apply(lambda x: x[1])

def parse_timestamp(ts):
    try:
        h, m = str(ts).split(":")
        return int(h), int(m)
    except Exception:
        return np.nan, np.nan

ts_parsed = combined["timestamp"].apply(parse_timestamp)
combined["hour"]   = ts_parsed.apply(lambda x: x[0])
combined["minute"] = ts_parsed.apply(lambda x: x[1])

# Cyclical time encoding
combined["hour_sin"] = np.sin(2 * np.pi * combined["hour"] / 24)
combined["hour_cos"] = np.cos(2 * np.pi * combined["hour"] / 24)
combined["min_sin"]  = np.sin(2 * np.pi * combined["minute"] / 60)
combined["min_cos"]  = np.cos(2 * np.pi * combined["minute"] / 60)
combined["day_sin"]  = np.sin(2 * np.pi * combined["day"] / 7)
combined["day_cos"]  = np.cos(2 * np.pi * combined["day"] / 7)

# Binary categoricals
combined["LargeVehicles"] = (combined["LargeVehicles"] == "Allowed").astype(int)
combined["Landmarks"]     = (combined["Landmarks"] == "Yes").astype(int)

# Label encode categoricals
for col in ["RoadType", "Weather"]:
    combined[col] = combined[col].fillna("Unknown")
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col].astype(str))

# Fill numeric NAs with train median
num_cols = ["Temperature", "NumberofLanes", "lat", "lng"]
for col in num_cols:
    median_val = combined.iloc[:n_train][col].median()
    combined[col] = combined[col].fillna(median_val)

# Keep raw geohash + hour key for target encoding (before dropping)
combined["geohash_raw"]      = combined["geohash"]
combined["geohash_hour_key"] = combined["geohash"] + "_" + combined["hour"].astype(str)
combined["geohash_day_key"]  = combined["geohash"] + "_" + combined["day"].astype(str)

combined.drop(columns=["geohash", "timestamp", "Index"], inplace=True)

# ── Split back ────────────────────────────────────────────────────────────────
X_train = combined.iloc[:n_train].reset_index(drop=True)
X_test  = combined.iloc[n_train:].reset_index(drop=True)
y_train = target.reset_index(drop=True)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ── Target Encoding helper (applied inside folds to prevent leakage) ──────────
def add_target_encodings(X_tr, y_tr, X_val, X_te, global_mean):
    for key_col, new_col in [
        ("geohash_raw",      "geohash_enc"),
        ("geohash_hour_key", "geohash_hour_enc"),
        ("geohash_day_key",  "geohash_day_enc"),
    ]:
        enc_map = X_tr.join(y_tr.rename("demand"))[key_col].map(
            X_tr.join(y_tr.rename("demand")).groupby(key_col)["demand"].mean()
        )
        # build map from tr
        mean_map = X_tr.copy()
        mean_map["demand"] = y_tr.values
        mean_map = mean_map.groupby(key_col)["demand"].mean()

        X_tr  = X_tr.copy();  X_tr[new_col]  = X_tr[key_col].map(mean_map).fillna(global_mean)
        X_val = X_val.copy(); X_val[new_col] = X_val[key_col].map(mean_map).fillna(global_mean)
        X_te  = X_te.copy();  X_te[new_col]  = X_te[key_col].map(mean_map).fillna(global_mean)

    return X_tr, X_val, X_te

# Drop raw key cols after encoding (done inside fold)
KEY_COLS = ["geohash_raw", "geohash_hour_key", "geohash_day_key"]

# ── LightGBM params ───────────────────────────────────────────────────────────
lgb_params = {
    "objective":         "regression",
    "metric":            "rmse",
    "learning_rate":     0.03,
    "num_leaves":        127,
    "max_depth":         -1,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_estimators":      2000,
    "random_state":      42,
    "verbose":           -1,
}

# ── XGBoost params ────────────────────────────────────────────────────────────
xgb_params = {
    "objective":        "reg:squarederror",
    "learning_rate":    0.03,
    "max_depth":        7,
    "n_estimators":     2000,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "verbosity":        0,
    "tree_method":      "hist",
}

# ── Cross-validation ──────────────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb  = np.zeros(n_train)
oof_xgb  = np.zeros(n_train)
oof_cat  = np.zeros(n_train)
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

global_mean = y_train.mean()

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    X_tr_raw  = X_train.iloc[tr_idx]
    X_val_raw = X_train.iloc[val_idx]
    y_tr      = y_train.iloc[tr_idx]
    y_val     = y_train.iloc[val_idx]

    # Add target encodings (leak-free)
    X_tr, X_val, X_te = add_target_encodings(X_tr_raw, y_tr, X_val_raw, X_test, global_mean)

    # Drop raw key cols
    X_tr  = X_tr.drop(columns=KEY_COLS)
    X_val = X_val.drop(columns=KEY_COLS)
    X_te  = X_te.drop(columns=KEY_COLS)

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
    )
    oof_lgb[val_idx] = lgb_model.predict(X_val)
    test_lgb        += lgb_model.predict(X_te) / kf.n_splits
    joblib.dump(lgb_model, f"{MODELS_DIR}/lgbm_fold{fold+1}.joblib")

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb_model = xgb.XGBRegressor(**xgb_params, early_stopping_rounds=100, eval_metric="rmse")
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = xgb_model.predict(X_val)
    test_xgb        += xgb_model.predict(X_te) / kf.n_splits
    joblib.dump(xgb_model, f"{MODELS_DIR}/xgb_fold{fold+1}.joblib")

    # ── CatBoost ──────────────────────────────────────────────────────────────
    cat_model = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=7,
        loss_function="RMSE", random_seed=42,
        early_stopping_rounds=100, verbose=False,
    )
    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    oof_cat[val_idx] = cat_model.predict(X_val)
    test_cat        += cat_model.predict(X_te) / kf.n_splits
    joblib.dump(cat_model, f"{MODELS_DIR}/cat_fold{fold+1}.joblib")

    lgb_r2 = r2_score(y_val, oof_lgb[val_idx])
    xgb_r2 = r2_score(y_val, oof_xgb[val_idx])
    cat_r2 = r2_score(y_val, oof_cat[val_idx])
    print(f"Fold {fold+1}  LGB: {lgb_r2:.4f}  XGB: {xgb_r2:.4f}  CAT: {cat_r2:.4f}")

# ── Blend LGB + XGB + CatBoost (equal weights) ───────────────────────────────
oof_blend  = (oof_lgb  + oof_xgb  + oof_cat)  / 3
test_blend = (test_lgb + test_xgb + test_cat) / 3

print(f"\nOOF R²  →  LGB: {r2_score(y_train, oof_lgb):.4f}  "
      f"XGB: {r2_score(y_train, oof_xgb):.4f}  "
      f"CAT: {r2_score(y_train, oof_cat):.4f}  "
      f"Blend: {r2_score(y_train, oof_blend):.4f}")
print(f"Score ≈ {max(0, 100 * r2_score(y_train, oof_blend)):.2f}")

# ── Save submission ───────────────────────────────────────────────────────────
test_index = pd.read_csv(TEST_PATH)["Index"]
submission = pd.DataFrame({"Index": test_index, "demand": test_blend})
submission.to_csv(OUT_PATH, index=False)
print(f"\nSaved {OUT_PATH} with {len(submission)} rows.")
print(submission.head())
