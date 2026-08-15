"""
CI Walk-forward（拡張ウィンドウ）検証
各年 T に対して: 2017〜T-1 で XGBoost 学習 → T年の P̂(contact) 予測 → AUC + CI 計算
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss
import xgboost as xgb

DATA = Path("C:/Users/yasuhiro_tahara/Desktop/Project/3. Player(Feilder)/##. Player Development/R/data")
SEP = "=" * 65

FEAT = [
    "rel_speed","induced_vert_break","horz_break","spin_rate",
    "rel_height","rel_side","plate_loc_height","plate_loc_side",
    "balls_x","strikes_x","pitching_lr_enc","batting_lr_enc",
    "pitch_type_enc","fast_spe_ave","vert_appr_angle","horz_appr_angle"
]

print("データ読み込み中...")
df = pd.read_parquet(DATA / "pitches_with_difficulty.parquet",
                     columns=["batter_code","season_x","top_farm","is_swing","is_whiff"] + FEAT)

df = df[df["is_swing"] == 1].copy()
df["is_contact"] = (1 - df["is_whiff"]).astype(np.int8)
df = df.dropna(subset=FEAT + ["is_contact"]).reset_index(drop=True)

years = sorted(df["season_x"].unique())
print(f"年度: {years}")
print(f"スイング投球 総数: {len(df):,}")

# ── Walk-forward: 2018 以降を順に検証 ─────────────────────
print(f"\n{SEP}")
print("Walk-forward AUC（各年 T: 2017〜T-1 で学習 → T年でテスト）")
print(SEP)
print(f"\n{'T年':>6}  {'学習年':>14}  {'学習n':>10}  {'テストn':>10}  {'AUC':>8}  {'Brier':>8}")
print("-" * 65)

results = []
for i, test_yr in enumerate(years):
    if test_yr <= min(years):  # 最初の年はスキップ（学習データなし）
        continue
    train_yrs = [y for y in years if y < test_yr]
    train = df[df["season_x"].isin(train_yrs)]
    test  = df[df["season_x"] == test_yr]
    if len(train) < 5000 or len(test) < 500:
        continue

    X_tr, y_tr = train[FEAT].values, train["is_contact"].values
    X_te, y_te = test[FEAT].values, test["is_contact"].values

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", use_label_encoder=False,
        random_state=42, n_jobs=4
    )
    model.fit(X_tr, y_tr, verbose=False)
    prob = model.predict_proba(X_te)[:,1]

    auc = roc_auc_score(y_te, prob)
    bs  = brier_score_loss(y_te, prob)
    print(f"{test_yr:>6}  {min(train_yrs)}〜{max(train_yrs):>4}  {len(train):>10,}  {len(test):>10,}  {auc:>8.4f}  {bs:>8.4f}")
    results.append((test_yr, auc, bs, prob, test))

# ── CI スコア（Walk-forward 版）─────────────────────────────
print(f"\n{SEP}")
print("Walk-forward CI スコア vs 全データ学習 CI スコアの相関")
print(SEP)

# 全データ学習モデルのexp_contact（ci_new.parquetで代用: whiff_pctを直接使用）
ci_ref = pd.read_parquet(DATA / "ci_new.parquet")

# Walk-forwardで計算したCI
ci_wf_list = []
for (test_yr, auc, bs, prob, test_df) in results:
    tmp = test_df[["batter_code","season_x","is_contact","top_farm"]].copy()
    tmp["prob_wf"] = prob
    tmp["residual_wf"] = tmp["is_contact"] - tmp["prob_wf"]
    ci_wf = tmp.groupby(["batter_code","season_x"]).agg(
        CI_wf=("residual_wf","mean"),
        n_swings=("is_contact","count"),
        top_farm=("top_farm","first")
    ).reset_index()
    ci_wf_list.append(ci_wf)

ci_wf_all = pd.concat(ci_wf_list).reset_index(drop=True)
ci_wf_all = ci_wf_all[ci_wf_all["n_swings"] >= 100].copy()

# 一軍基準パーセンタイル化
ichi_wf = ci_wf_all[ci_wf_all["top_farm"] == "一軍"]["CI_wf"].values
from scipy.stats import percentileofscore
ci_wf_all["CI_wf_pct"] = ci_wf_all["CI_wf"].apply(
    lambda v: percentileofscore(ichi_wf, v, kind="rank"))

# 全データモデルのCI_new と比較
merged = ci_wf_all.merge(ci_ref[["batter_code","season_x","CI_new","whiff_pct"]],
                          on=["batter_code","season_x"], how="inner")
print(f"\nマッチ件数: {len(merged)}")
r_ci, _ = stats.pearsonr(merged["CI_wf"], merged["CI_new"])
print(f"r(CI_wf, CI_fullmodel) = {r_ci:+.4f}  ← Walk-forward vs 全データ学習の一致度")

for tier in ["一軍","ファーム"]:
    sub = merged[merged["top_farm"] == tier]
    if len(sub) < 20: continue
    r_w, _ = stats.pearsonr(sub["CI_wf"], sub["whiff_pct"])
    r_f, _ = stats.pearsonr(sub["CI_new"], sub["whiff_pct"])
    print(f"\n{tier} (n={len(sub)}):")
    print(f"  r(CI_wf,    Whiff%) = {r_w:+.4f}  ← Walk-forward版")
    print(f"  r(CI_full,  Whiff%) = {r_f:+.4f}  ← 全データ学習版（参照）")

# ── Walk-forward 翌年予測 ──────────────────────────────────
print(f"\n{SEP}")
print("Walk-forward 翌年予測 (CI_wf_Y → Whiff%_Y+1, 一軍のみ)")
print(SEP)

ichi_wf2 = ci_wf_all[ci_wf_all["top_farm"] == "一軍"].copy()
ichi_wf2 = ichi_wf2.merge(ci_ref[["batter_code","season_x","whiff_pct"]],
                            on=["batter_code","season_x"], how="inner")
pro_wf = (ichi_wf2[["batter_code","season_x","CI_wf","whiff_pct"]]
          .assign(join_yr=lambda d: d["season_x"]+1)
          .merge(ichi_wf2[["batter_code","season_x","whiff_pct"]]
                 .rename(columns={"whiff_pct":"whiff_next","season_x":"ny"}),
                 left_on=["batter_code","join_yr"], right_on=["batter_code","ny"], how="inner"))

print(f"\n連続年ペア数: {len(pro_wf)}")
if len(pro_wf) >= 10:
    r_next, _ = stats.pearsonr(pro_wf["CI_wf"], pro_wf["whiff_next"])
    print(f"r(CI_wf_Y, Whiff%_Y+1) = {r_next:+.4f}  ← Walk-forward版翌年予測")
