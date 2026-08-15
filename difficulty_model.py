"""
difficulty_model.py  — C+ 版（2難易度）
難易度を KSI 用と CDI 用で分けて出力する

diff_k (KSI/PDI 用): C+ モデル
  特徴量: rel_speed + break + pitch_type + VAA/HAA + release position/angle
  ターゲット: P(whiff | swing)
  AUC ≈ 0.727

diff_c (CDI 用): 純スタッフ C モデル
  特徴量: rel_speed + break + pitch_type のみ
  ターゲット: P(whiff | swing)
  AUC ≈ 0.648
  → 投球軌道・リリース位置に依存しない「球の本来の質」

出力:
  data/difficulty_model.pkl
  data/pitches_with_difficulty.parquet  (diff_k, diff_c 両列を含む)
"""
import numpy as np
import pandas as pd
import pickle
import warnings
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

RAW_DIR  = Path("C:/Users/yasuhiro_tahara/Desktop/Project/98. Input-Data")
DATA_DIR = Path("C:/Users/yasuhiro_tahara/Desktop/Project/3. Player(Feilder)/##. Player Development/R/data")
DATA_DIR.mkdir(exist_ok=True)

SWING_CALLS = {"StrikeSwinging", "InPlay", "FoulBall", "FoulBallNotFieldable",
               "FoulBallFieldable", "FoulTip"}
WHIFF_CALLS = {"StrikeSwinging"}

KEEP_COLS = [
    "batter", "batter_code", "pitcher_code", "season_x", "balls_x", "strikes_x",
    "pitch_call", "new_pitch_type", "pitch_type",
    "rel_speed", "induced_vert_break", "horz_break", "spin_rate",
    "plate_loc_height", "plate_loc_side",
    "vert_appr_angle", "horz_appr_angle",
    "rel_height", "rel_side", "vert_rel_angle", "horz_rel_angle",
    "pitching_lr", "batting_lr", "fast_spe_ave",
    "top_farm",
]

# ── Step 1: データ読み込み ────────────────────────────────
print("Step 1: 全年度データ読み込み")

def normalize_name(n):
    n = n.lower()
    for a, b in [("ou","o"),("oh","o"),("uu","u"),("oo","o")]:
        n = n.replace(a, b)
    return n

pl = pd.read_csv(RAW_DIR / "player_list20-26.csv", encoding="cp932")
pl = pl.dropna(subset=["height", "player_name_en"])
pl["name_norm"] = pl["player_name_en"].apply(normalize_name)
height_map = pl.groupby("name_norm")["height"].first()

parts = []
for yr in range(17, 27):
    p = RAW_DIR / f"ALL_F_combined{yr}.parquet"
    if p.exists():
        avail = [c for c in KEEP_COLS if c in pd.read_parquet(p, columns=[]).columns
                 or True]
        try:
            tmp = pd.read_parquet(p, columns=KEEP_COLS)
        except Exception:
            cols_fb = [c for c in KEEP_COLS if c not in
                       ["vert_appr_angle","horz_appr_angle","rel_height","rel_side",
                        "vert_rel_angle","horz_rel_angle","pitcher_code"]]
            tmp = pd.read_parquet(p, columns=cols_fb)
            for c in ["vert_appr_angle","horz_appr_angle","rel_height","rel_side",
                      "vert_rel_angle","horz_rel_angle","pitcher_code"]:
                tmp[c] = np.nan
        tmp["season_x"] = yr + 2000
        parts.append(tmp)
        print(f"  {yr}: {len(tmp):,}行")

pitches = pd.concat(parts, ignore_index=True)
print(f"  合計: {len(pitches):,}行")

# 身長・ストライクゾーン正規化
pitches["batter_ps"] = (
    pitches["batter"].str.split(", ")
    .apply(lambda x: " ".join(reversed(x))
           if isinstance(x, list) and len(x) == 2 else
           (x[0] if isinstance(x, list) else ""))
)
pitches["name_norm"] = pitches["batter_ps"].apply(normalize_name)
pitches["height_cm"] = pitches["name_norm"].map(height_map)
pitches["height_m"]  = pitches["height_cm"].fillna(176.0) / 100.0

sz_bot = pitches["height_m"] * 0.270
sz_top = pitches["height_m"] * 0.535
sz_mid = (sz_bot + sz_top) / 2.0
sz_h   = sz_top - sz_bot

pitches["loc_h_norm"] = (pitches["plate_loc_height"] - sz_mid) / sz_h
pitches["loc_s_norm"] = pitches["plate_loc_side"] / 0.2535

# ── Step 2: 基本フラグ ───────────────────────────────────
print("\nStep 2: スイング/空振り/ゾーン判定")

pitches["is_swing"]   = pitches["pitch_call"].isin(SWING_CALLS)
pitches["is_whiff"]   = pitches["pitch_call"].isin(WHIFF_CALLS)
pitches["is_ball"]    = pitches["pitch_call"] == "BallCalled"
pitches["is_in_zone"] = (
    (np.abs(pitches["loc_h_norm"]) <= 1.0) &
    (np.abs(pitches["loc_s_norm"]) <= 1.0)
).astype(int)

print(f"  スイング数   : {pitches['is_swing'].sum():,}")
print(f"  空振り数     : {pitches['is_whiff'].sum():,}")
print(f"  リーグ空振率 : {pitches['is_whiff'].sum()/pitches['is_swing'].sum():.3f}")

# ── Step 3: 球種エンコード (スイング投球) ───────────────────
print("\nStep 3: 球種エンコード")

pitches["pt_clean"] = pitches["new_pitch_type"].fillna("other").str.lower()
swings = pitches[pitches["is_swing"]].copy()
print(f"  スイングサンプル: {len(swings):,}")

TOP_TYPES = swings["pt_clean"].value_counts().head(10).index.tolist()
pt_sw = swings["pt_clean"].map(lambda x: x if x in TOP_TYPES else "other")
pt_dummies_sw = pd.get_dummies(pt_sw, prefix="pt", dtype=np.float32)

y_sw = swings["is_whiff"].values.astype(int)
valid_sw = np.isfinite(swings["rel_speed"].values)

# ── Step 4: 特徴量構築 ──────────────────────────────────
STUFF_COLS   = ["rel_speed", "induced_vert_break", "horz_break"]
APPROACH_COLS= ["vert_appr_angle", "horz_appr_angle"]
RELEASE_COLS = ["rel_height", "rel_side", "vert_rel_angle", "horz_rel_angle"]
CP_COLS = STUFF_COLS + APPROACH_COLS + RELEASE_COLS   # C+ 全特徴量
C_COLS  = STUFF_COLS                                   # 純スタッフ

def build_X(cols, src=swings, dummies=pt_dummies_sw):
    base = src[cols].copy().astype(np.float32)
    X = pd.concat([base.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X.fillna(X.median())

X_cp = build_X(CP_COLS)[valid_sw].reset_index(drop=True)
X_c  = build_X(C_COLS)[valid_sw].reset_index(drop=True)
y    = y_sw[valid_sw]

print(f"\nStep 4: 特徴量")
print(f"  C+特徴量: {X_cp.shape[1]}列  ({', '.join(CP_COLS[:5])}...)")
print(f"  C 特徴量: {X_c.shape[1]}列   ({', '.join(C_COLS)})")
print(f"  有効サンプル: {len(X_cp):,}")

# ── Step 5: モデル学習 ──────────────────────────────────
print("\nStep 5: 2モデル学習")

rng = np.random.default_rng(42)
n_sample = min(300_000, len(X_cp))
idx = rng.choice(len(X_cp), size=n_sample, replace=False)
mask_test = np.ones(len(X_cp), dtype=bool)
mask_test[idx] = False

GBC_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  subsample=0.8, min_samples_leaf=50, random_state=42, verbose=0)

print(f"\n  [diff_k] C+ モデル 学習中...")
clf_k = GradientBoostingClassifier(**GBC_PARAMS)
clf_k.fit(X_cp.iloc[idx], y[idx])
if mask_test.sum() > 1000:
    prob_k = clf_k.predict_proba(X_cp[mask_test])[:, 1]
    auc_k  = roc_auc_score(y[mask_test], prob_k)
    print(f"  AUC (holdout) = {auc_k:.4f}")

print(f"\n  特徴量重要度 (diff_k):")
fn_k = X_cp.columns.tolist()
for i in np.argsort(clf_k.feature_importances_)[::-1][:12]:
    print(f"    {fn_k[i]:<28} {clf_k.feature_importances_[i]:.4f}")

print(f"\n  [diff_c] 純スタッフ C モデル 学習中...")
clf_c = GradientBoostingClassifier(**GBC_PARAMS)
clf_c.fit(X_c.iloc[idx], y[idx])
if mask_test.sum() > 1000:
    prob_c = clf_c.predict_proba(X_c[mask_test])[:, 1]
    auc_c  = roc_auc_score(y[mask_test], prob_c)
    print(f"  AUC (holdout) = {auc_c:.4f}")

print(f"\n  特徴量重要度 (diff_c):")
fn_c = X_c.columns.tolist()
for i in np.argsort(clf_c.feature_importances_)[::-1][:8]:
    print(f"    {fn_c[i]:<28} {clf_c.feature_importances_[i]:.4f}")

# ── Step 6: 全投球に難易度付与 ─────────────────────────
print("\nStep 6: 全投球に難易度スコア付与")

pt_all = pitches["pt_clean"].map(lambda x: x if x in TOP_TYPES else "other")
pt_all_dum = pd.get_dummies(pt_all, prefix="pt", dtype=np.float32)

def predict_all(clf, cols, col_order):
    base = pitches[cols].copy().astype(np.float32)
    Xa = pd.concat([base.reset_index(drop=True), pt_all_dum.reset_index(drop=True)], axis=1)
    Xa = Xa.reindex(columns=col_order, fill_value=0.0).fillna(0.0)
    return clf.predict_proba(Xa)[:, 1]

pitches["diff_k"] = predict_all(clf_k, CP_COLS, X_cp.columns.tolist())
pitches["diff_c"] = predict_all(clf_c, C_COLS,  X_c.columns.tolist())

print(f"  diff_k: mean={pitches['diff_k'].mean():.3f}  std={pitches['diff_k'].std():.3f}")
print(f"  diff_c: mean={pitches['diff_c'].mean():.3f}  std={pitches['diff_c'].std():.3f}")

print("\n  球種別 diff_k / diff_c:")
type_diff = pitches.groupby("pt_clean")[["diff_k","diff_c"]].mean().sort_values("diff_k", ascending=False)
print(type_diff.round(3).to_string())

# 後方互換: difficulty列はdiff_kと同じ
pitches["difficulty"] = pitches["diff_k"]

# ── Step 7: 保存 ───────────────────────────────────────
print("\nStep 7: 保存")

save_cols = [
    "batter_code", "pitcher_code", "season_x", "balls_x", "strikes_x",
    "pitch_call", "new_pitch_type", "pitch_type",
    "plate_loc_height", "plate_loc_side",
    "loc_h_norm", "loc_s_norm",
    "height_m",
    "diff_k", "diff_c",
    "difficulty",          # 後方互換
    "is_swing", "is_whiff", "is_ball", "is_in_zone",
    "top_farm",
    # CI contact model (16変数) 用
    "rel_speed", "induced_vert_break", "horz_break", "spin_rate",
    "rel_height", "rel_side",
    "vert_appr_angle", "horz_appr_angle",
    "pitching_lr", "batting_lr", "fast_spe_ave",
]
out = pitches[save_cols].copy()
out["pitcher_code"] = out["pitcher_code"].astype(str)
out.to_parquet(DATA_DIR / "pitches_with_difficulty.parquet", index=False)
print(f"  保存: pitches_with_difficulty.parquet  ({len(pitches):,}行)")
print(f"    diff_k: C+ モデル (KSI/PDI 用)")
print(f"    diff_c: 純スタッフ C モデル (CDI 用)")

with open(DATA_DIR / "difficulty_model.pkl", "wb") as f:
    pickle.dump({
        "model_k":    clf_k,
        "model_c":    clf_c,
        "features_k": X_cp.columns.tolist(),
        "features_c": X_c.columns.tolist(),
        "top_types":  TOP_TYPES,
        "cp_cols":    CP_COLS,
        "c_cols":     C_COLS,
    }, f)
print("  保存: difficulty_model.pkl")
print("\n[OK] 完了")
