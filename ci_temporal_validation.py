"""CI モデル 年度分割検証"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

DATA = Path("C:/Users/yasuhiro_tahara/Desktop/Project/3. Player(Feilder)/##. Player Development/R/data")
SEP = "=" * 62

# ── 1. ピッチレベル年度別モデル精度 ─────────────────────────
print(SEP)
print("1. ピッチレベル 年度別精度（AUC / Brier / LogLoss）")
print(SEP)

df = pd.read_parquet(DATA / "pitches_with_difficulty.parquet",
                     columns=["season_x","is_swing","is_whiff","exp_contact","top_farm"])

sw = df[df["is_swing"] == 1].copy()
sw["is_contact"] = (1 - sw["is_whiff"]).astype(int)

# 全体
auc_all = roc_auc_score(sw["is_contact"], sw["exp_contact"])
bs_all  = brier_score_loss(sw["is_contact"], sw["exp_contact"])
ll_all  = log_loss(sw["is_contact"], sw["exp_contact"])
print(f"\n全体 (n={len(sw):,})  AUC={auc_all:.4f}  Brier={bs_all:.4f}  LogLoss={ll_all:.4f}")

# 一軍のみ年度別
print(f"\n{'年':>6}  {'Tier':>8}  {'n':>8}  {'AUC':>8}  {'Brier':>8}  {'LogLoss':>9}")
print("-" * 57)
rows = []
for (yr, tier), grp in sw.groupby(["season_x","top_farm"]):
    if len(grp) < 300: continue
    auc = roc_auc_score(grp["is_contact"], grp["exp_contact"])
    bs  = brier_score_loss(grp["is_contact"], grp["exp_contact"])
    ll  = log_loss(grp["is_contact"], grp["exp_contact"])
    rows.append((yr, tier, len(grp), auc, bs, ll))
    print(f"{yr:>6}  {tier:>8}  {len(grp):>8,}  {auc:>8.4f}  {bs:>8.4f}  {ll:>9.4f}")

# ── 2. 一軍限定のキャリブレーション確認 ─────────────────────
print(f"\n{SEP}")
print("2. キャリブレーション（10分位）― 一軍 全年度合算")
print(SEP)
ichi_sw = sw[sw["top_farm"] == "一軍"].copy()
ichi_sw["bin"] = pd.qcut(ichi_sw["exp_contact"], 10, labels=False)
cal = ichi_sw.groupby("bin").agg(
    mean_pred=("exp_contact","mean"),
    mean_actual=("is_contact","mean"),
    n=("is_contact","count")
).reset_index()
cal["diff"] = cal["mean_actual"] - cal["mean_pred"]
print(f"\n{'分位':>5}  {'予測値':>8}  {'実測値':>8}  {'差':>8}  {'n':>7}")
print("-" * 42)
for _, r in cal.iterrows():
    print(f"{int(r['bin'])+1:>5}  {r['mean_pred']:>8.4f}  {r['mean_actual']:>8.4f}  {r['diff']:>+8.4f}  {int(r['n']):>7,}")

# ── 3. バッターシーズンレベル 年度別相関 ─────────────────────
print(f"\n{SEP}")
print("3. バッターシーズンレベル 年度別 CI vs Whiff% 相関")
print(SEP)
ci = pd.read_parquet(DATA / "ci_new.parquet")
ci100 = ci[ci["n_swings"] >= 100].copy()

print(f"\n{'年':>6}  {'n':>6}  {'r(CI,Whiff%)':>14}  {'Tier'}")
print("-" * 45)
for (yr, tier), grp in ci100.groupby(["season_x","top_farm"]):
    sub = grp.dropna(subset=["CI_new","whiff_pct"])
    if len(sub) < 15: continue
    r, _ = stats.pearsonr(sub["CI_new"], sub["whiff_pct"])
    print(f"{yr:>6}  {len(sub):>6}  {r:>+14.4f}  {tier}")

# ── 4. 翌年予測妥当性（一軍） ────────────────────────────────
print(f"\n{SEP}")
print("4. 翌年予測妥当性: CI(年Y, 一軍) → Whiff%(年Y+1, 一軍)")
print(SEP)
ichi_ci = ci100[ci100["top_farm"] == "一軍"].copy()
# 左: 年Y, 右: 年Y+1 の Whiff% → join key を +1 してマッチ
pro = (ichi_ci[["batter_code","season_x","CI_new","whiff_pct"]]
       .assign(join_yr=lambda d: d["season_x"] + 1)
       .merge(
           ichi_ci[["batter_code","season_x","whiff_pct"]]
               .rename(columns={"whiff_pct":"whiff_next","season_x":"ny"}),
           left_on=["batter_code","join_yr"],
           right_on=["batter_code","ny"],
           how="inner"
       ))

print(f"\n連続年ペア数: {len(pro)}")
r_same, _ = stats.pearsonr(pro["CI_new"],   pro["whiff_pct"])
r_next, _ = stats.pearsonr(pro["CI_new"],   pro["whiff_next"])
r_ww,   _ = stats.pearsonr(pro["whiff_pct"],pro["whiff_next"])
print(f"  同年    r(CI_Y , Whiff%_Y)   = {r_same:+.4f}  ← 参照")
print(f"  翌年    r(CI_Y , Whiff%_Y+1) = {r_next:+.4f}  ← 予測妥当性")
print(f"  WW比較  r(Whiff_Y, Whiff_Y+1)= {r_ww:+.4f}  ← Whiff%の安定性")

print(f"\n{'年ペア':>12}  {'n':>5}  {'r(CI→次年Whiff)':>18}")
print("-" * 42)
for yr, grp2 in pro.groupby("season_x"):
    if len(grp2) < 10: continue
    r, _ = stats.pearsonr(grp2["CI_new"], grp2["whiff_next"])
    print(f"  {yr}→{yr+1}  {len(grp2):>5}  {r:>+18.4f}")

# ── 5. K% / BB% との相関（参考） ─────────────────────────────
print(f"\n{SEP}")
print("5. K% / BB% との相関（参考）")
print(SEP)
try:
    raw = pd.read_parquet(DATA / "pitches_with_difficulty.parquet",
                          columns=["batter_code","season_x","pitch_call"])
    K_CALLS  = {"StrikeSwinging","StrikeCalled","DropThird"}
    BB_CALLS = {"BallCalled","HitByPitch","BallIntentional"}
    PA_ENDS  = K_CALLS | BB_CALLS | {
        "InPlay","BuntFoul","FoulBall","FoulBallFieldable",
        "FoulBallNotFieldable","FoulTip"}
    pa = raw[raw["pitch_call"].isin(PA_ENDS)].copy()
    pa["is_k"]   = pa["pitch_call"].isin(K_CALLS).astype(int)
    pa["is_swK"] = pa["pitch_call"].isin({"StrikeSwinging","DropThird"}).astype(int)
    pa["is_lkK"] = pa["pitch_call"].isin({"StrikeCalled"}).astype(int)
    pa["is_bb"]  = pa["pitch_call"].isin(BB_CALLS).astype(int)
    out = pa.groupby(["batter_code","season_x"]).agg(
        n_pa=("pitch_call","count"),
        n_k=("is_k","sum"),
        n_swK=("is_swK","sum"),
        n_lkK=("is_lkK","sum"),
        n_bb=("is_bb","sum")
    ).reset_index()
    out = out[out["n_pa"] >= 80].copy()
    out["k_pct"]    = out["n_k"]   / out["n_pa"]
    out["swK_pct"]  = out["n_swK"] / out["n_pa"]
    out["lkK_pct"]  = out["n_lkK"] / out["n_pa"]
    out["bb_pct"]   = out["n_bb"]  / out["n_pa"]

    merged = ci100.merge(out, on=["batter_code","season_x"])
    ichi_m = merged[merged["top_farm"] == "一軍"]
    print(f"\n  {'指標':<20} {'全体 r':>10} {'一軍 r':>10}")
    print(f"  {'-'*44}")
    for col, lbl in [("k_pct","K%"), ("swK_pct","空振り三振率"),
                     ("lkK_pct","見逃し三振率"), ("bb_pct","BB%")]:
        ra, _ = stats.pearsonr(merged["CI_new"],  merged[col]*100)
        ri, _ = stats.pearsonr(ichi_m["CI_new"], ichi_m[col]*100)
        print(f"  {lbl:<20} {ra:>+10.4f} {ri:>+10.4f}")
except Exception as e:
    print(f"  エラー: {e}")
