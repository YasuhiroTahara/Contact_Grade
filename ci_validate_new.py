"""
新 CI モデル（XGBoost 16変数）の検証 — 旧モデルとの比較
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import percentileofscore

DATA_DIR = Path("C:/Users/yasuhiro_tahara/Desktop/Project/3. Player(Feilder)/##. Player Development/R/data")

ci = pd.read_parquet(DATA_DIR / "ci_new.parquet")
ci = ci[ci["n_swings"] >= 100].copy()

ichi = ci[ci["top_farm"] == "一軍"]
farm = ci[ci["top_farm"] == "ファーム"]
ichi_vals = ichi["CI_new"].values
ci["CI_pct"] = ci["CI_new"].apply(lambda v: percentileofscore(ichi_vals, v, kind="rank"))

SEP = "=" * 55

print(SEP)
print("CI モデル更新後 検証（新: XGBoost 16変数）")
print(SEP)

# ── 分布 ───────────────────────────────────────
print(f"\n一軍分布 (n={len(ichi)}):")
for p, lbl in [(10,"p10"),(25,"p25"),(50,"p50"),(75,"p75"),(90,"p90")]:
    print(f"  {lbl}: {np.percentile(ichi['CI_new'], p):+.5f}")

print(f"\nファーム (n={len(farm)}):")
print(f"  mean: {farm['CI_new'].mean():+.5f}  std: {farm['CI_new'].std():.5f}")
f_pct = ci[ci["top_farm"] == "ファーム"]["CI_pct"].mean()
print(f"  ファーム平均パーセンタイル: {f_pct:.1f}")

# ── ファームバイアス ────────────────────────────
dual = ci.pivot_table(
    index=["batter_code","season_x"], columns="top_farm", values="CI_new"
).dropna()
dual.columns.name = None
if "一軍" in dual.columns and "ファーム" in dual.columns:
    diff = dual["ファーム"] - dual["一軍"]
    print(f"\nファームバイアス (n={len(dual)}):")
    print(f"  Farm>Ichi: {(diff>0).mean():.1%}")
    print(f"  効果量: {diff.mean()/diff.std():+.2f} SD")

# ── 相関検証 ────────────────────────────────────
print(f"\n相関検証:")
print(f"  {'指標':<12} {'全体 r':>9} {'一軍 r':>9} {'ファーム r':>9}")
print(f"  {'-'*45}")

# whiff_pct の単位確認・変換
whiff_col = "whiff_pct"
sample_max = ci[whiff_col].max()
scale = 100 if sample_max <= 1.5 else 1

for col, label in [("whiff_pct","Whiff%")]:
    vals = ci[col] * scale
    row = []
    for tier in ["全体","一軍","ファーム"]:
        sub_ci = ci if tier == "全体" else ci[ci["top_farm"] == tier]
        sub_v  = vals if tier == "全体" else vals[ci["top_farm"] == tier]
        r, _ = stats.pearsonr(sub_ci["CI_new"], sub_v)
        row.append(f"{r:+.4f}")
    print(f"  {label:<12} {row[0]:>9} {row[1]:>9} {row[2]:>9}")

print(f"\n旧モデル（GBC 3変数）との比較:")
print(f"  Whiff% 旧: r=-0.8904")
print(f"  ファームバイアス 旧: 64.8% / +0.38 SD")
print(f"  ファーム平均%ile 旧: 42.4")
