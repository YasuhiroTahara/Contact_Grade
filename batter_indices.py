"""
batter_indices.py  — 詳細版
コンタクト実行指数 (KSI) / 選球指数 (PDI) 算出

KSI_raw = B_z  (難易度調整コンタクト率)
PDI_raw = E_z  (SOV ベース統合スイング判断スコア)
スケーリング: 平均=100, SD=15
"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path("C:/Users/yasuhiro_tahara/Desktop/Project/3. Player(Feilder)/##. Player Development/R/data")
RAW_DIR  = Path("C:/Users/yasuhiro_tahara/Desktop/Project/98. Input-Data")
MIN_PITCHES = 500
SEP = "="*70

# ── Step 1: データ読み込み ───────────────────────────────
print(SEP)
print("Step 1: 偏差データ読み込み")
print(SEP)
devs = pd.read_parquet(DATA_DIR / "batter_deviations.parquet")
print(f"  全行数     : {len(devs):,}")
print(f"  打者数     : {devs['batter_code'].nunique():,}")
print(f"  シーズン   : {sorted(devs['season_x'].unique())}")
print(f"  投球数分布 :")
print(f"    中央値={devs['n_pitches'].median():.0f}  "
      f"p25={devs['n_pitches'].quantile(.25):.0f}  "
      f"p75={devs['n_pitches'].quantile(.75):.0f}  "
      f"max={devs['n_pitches'].max():.0f}")

devs_f = devs[devs["n_pitches"] >= MIN_PITCHES].copy()
print(f"\n  {MIN_PITCHES}球以上フィルタ後: {len(devs_f):,}行 / {devs_f['batter_code'].nunique():,}選手")

for col in ["comp_B","comp_E","comp_D","contact_diff_mean"]:
    devs_f[col] = devs_f[col].fillna(devs_f[col].median())

devs_f["o_swing_pct"] = (
    devs_f["chase_swing"] * 27.6 + devs_f["waste_swing"] * 9.3
) / (27.6 + 9.3)

# ── Step 1b: モデルベース KSI / PDI 算出 ────────────────────
print(f"\n{SEP}")
print("Step 1b: PitchGrade 型モデルベース KSI / PDI 算出")
print(SEP)

pitches_all = pd.read_parquet(DATA_DIR / "pitches_with_difficulty.parquet")
pitches_all["batter_code"] = pitches_all["batter_code"].astype(str)

if "exp_contact" not in pitches_all.columns or "exp_swing" not in pitches_all.columns:
    raise RuntimeError("exp_contact / exp_swing 列がありません。pitch_skill_model.py を先に実行してください。")

# diff_k 列 (C+モデル) を使用; なければ後方互換で difficulty
dk_col = "diff_k" if "diff_k" in pitches_all.columns else "difficulty"
dc_col = "diff_c" if "diff_c" in pitches_all.columns else "difficulty"
print(f"  KSI/PDI 難易度列: {dk_col}  (C+ モデル)")
print(f"  CDI   難易度列  : {dc_col}  (純スタッフ C モデル)")

# KSI_raw: diff_k²-weighted コンタクト残差 (スイング投球のみ)
sw = pitches_all[pitches_all["is_swing"]].copy()
sw["is_contact"]       = (~sw["is_whiff"]).astype(float)
sw["contact_residual"] = sw["is_contact"] - sw["exp_contact"]
sw["ksi_contrib"]      = sw[dk_col] ** 2 * sw["contact_residual"]

ksi_agg = (sw.groupby(["batter_code", "season_x"])
             .agg(KSI_raw=("ksi_contrib", "mean"))
             .reset_index())

# PDI_raw (ハイブリッド): SOV sign重み × モデルベーススイング残差
# SOV (Swing Opportunity Value) = RV_zone × (1 − difficulty)
#   高SOV → 振るべき球（ストライクゾーン × 打ちやすい）
#   低SOV → 取るべき球（ボールゾーン or 難しい）
# swing_residual = actual_swing − exp_swing  ← モデル基準（普遍的）
# PDI_raw = mean( swing_residual × sign(SOV − threshold) )

rv_zone_df = pd.read_parquet(DATA_DIR / "zone_runvalue.parquet")
rv_type_map = (rv_zone_df.groupby("zone_type")
               .apply(lambda g: (g["rv_mean"] * g["rv_n"]).sum() / g["rv_n"].sum())
               .to_dict())
zone_int_to_type = {0: "Heart", 1: "Shadow", 2: "Chase", 3: "Waste"}
pitches_all["rv_zone"] = pitches_all["zone_int"].map(
    lambda z: rv_type_map.get(zone_int_to_type.get(z, "Shadow"), 0.0))
pitches_all["SOV"] = pitches_all["rv_zone"] * (1.0 - pitches_all[dk_col])

SOV_threshold = pitches_all["SOV"].median()
print(f"  SOV threshold (中央値): {SOV_threshold:.5f}")
print(f"  SOV > threshold: {(pitches_all['SOV'] > SOV_threshold).mean()*100:.1f}%")

pitches_all["swing_residual"] = pitches_all["is_swing"].astype(float) - pitches_all["exp_swing"]
pitches_all["pdi_contrib"]    = pitches_all["swing_residual"] * np.sign(
    pitches_all["SOV"] - SOV_threshold)

pdi_agg = (pitches_all.groupby(["batter_code", "season_x"])
           .agg(PDI_raw=("pdi_contrib", "mean"))
           .reset_index())

# CDI_raw: コンタクトした投球の diff_c (純スタッフ難易度) 平均
# コンタクト = スイング & 非空振り (foul含む)
sw_contact = sw[sw["is_contact"] == 1.0].copy()
cdi_agg = (sw_contact.groupby(["batter_code", "season_x"])
           .agg(CDI_raw=(dc_col, "mean"))
           .reset_index())

devs_f = devs_f.merge(ksi_agg, on=["batter_code", "season_x"], how="left")
devs_f = devs_f.merge(pdi_agg, on=["batter_code", "season_x"], how="left")
devs_f = devs_f.merge(cdi_agg, on=["batter_code", "season_x"], how="left")

for col in ["KSI_raw", "PDI_raw", "CDI_raw"]:
    n_miss = devs_f[col].isna().sum()
    if col in devs_f.columns:
        print(f"  {col}: NaN={n_miss}  mean={devs_f[col].mean():.5f}  std={devs_f[col].std():.5f}")

# ── Step 2: 標準化 ───────────────────────────────────────
print(f"\n{SEP}")
print("Step 2: コンポーネント標準化 (Z スコア)")
print(SEP)
scaler = StandardScaler()
devs_f[["B_z","E_z","D_z"]] = scaler.fit_transform(devs_f[["comp_B","comp_E","comp_D"]])

print(f"\n  標準化後 統計 (mean≈0, std≈1 が正常):")
for col in ["B_z","E_z","D_z"]:
    print(f"    {col}: mean={devs_f[col].mean():+.4f}  std={devs_f[col].std():.4f}  "
          f"min={devs_f[col].min():.3f}  max={devs_f[col].max():.3f}")

# ── Step 3: KSI / PDI 生スコア確認 ─────────────────────
print(f"\n{SEP}")
print("Step 3: KSI / PDI 生スコア確認 (モデルベース)")
print(SEP)
# KSI_raw / PDI_raw は Step 1b で算出済み

r_kp, p_kp = stats.pearsonr(devs_f["KSI_raw"].fillna(0), devs_f["PDI_raw"].fillna(0))
r_kd, _    = stats.pearsonr(devs_f["KSI_raw"].fillna(0), devs_f["D_z"])
r_pd, _    = stats.pearsonr(devs_f["PDI_raw"].fillna(0), devs_f["D_z"])
print(f"\n  KSI vs PDI 相関: r={r_kp:+.4f}  p={p_kp:.2e}  (|r|<0.2 が理想)")
print(f"  KSI vs D_z  相関: r={r_kd:+.4f}")
print(f"  PDI vs D_z  相関: r={r_pd:+.4f}  (D_zはPDI補助 → r>0 が理想)")

# ── Step 4: スケーリング ─────────────────────────────────
print(f"\n{SEP}")
print("Step 4: スケーリング (mean=100, SD=15)")
print(SEP)

def scale_index(raw, mean=100, sd=15):
    z = (raw - raw.mean()) / (raw.std() + 1e-9)
    return mean + sd * z

devs_f["KSI"] = scale_index(devs_f["KSI_raw"])
devs_f["PDI"] = scale_index(devs_f["PDI_raw"])
# CDI = diff_c (純スタッフ難易度) のコンタクト投球平均 → 難しい球に当てられるか
devs_f["CDI"] = scale_index(devs_f["CDI_raw"].fillna(devs_f["CDI_raw"].median()))

for name in ["KSI","PDI","CDI"]:
    s = devs_f[name]
    print(f"\n  {name}: mean={s.mean():.2f}  SD={s.std():.2f}  "
          f"min={s.min():.1f}  max={s.max():.1f}  "
          f"p25={s.quantile(.25):.1f}  p75={s.quantile(.75):.1f}")
    bins = [0,70,85,100,115,130,200]
    labels = ["<70","70-85","85-100","100-115","115-130","130+"]
    cut = pd.cut(s, bins=bins, labels=labels, right=False)
    dist = cut.value_counts().sort_index()
    print(f"  分布: " + "  ".join(f"{l}:{v}" for l, v in dist.items()))

# シーズン別推移
print(f"\n  シーズン別 KSI / PDI 平均・SD:")
yr_stats = devs_f.groupby("season_x")[["KSI","PDI"]].agg(["mean","std"]).round(2)
print(yr_stats.to_string())

# ── Step 5: 実績スタッツ取得 ─────────────────────────────
print(f"\n{SEP}")
print("Step 5: 検証 — 実績指標収集")
print(SEP)

KEEP_V = ["batter_code","season_x","pitch_call","balls_x","strikes_x"]
parts = []
for yr in range(17, 27):
    p = RAW_DIR / f"ALL_F_combined{yr}.parquet"
    if p.exists():
        parts.append(pd.read_parquet(p, columns=KEEP_V))
val_pitches = pd.concat(parts, ignore_index=True)
val_pitches["batter_code"] = val_pitches["batter_code"].astype(str)

SWING_V = {"StrikeSwinging","InPlay","FoulBall","FoulBallNotFieldable","FoulBallFieldable","FoulTip"}
val_pitches["is_swing"] = val_pitches["pitch_call"].isin(SWING_V).astype(int)
val_pitches["is_whiff"] = (val_pitches["pitch_call"] == "StrikeSwinging").astype(int)

# 空振率 = 空振り / スイング
swing_agg = val_pitches.groupby(["batter_code","season_x"]).agg(
    n_pitches = ("is_swing","count"),
    n_swing   = ("is_swing","sum"),
    n_whiff   = ("is_whiff","sum"),
    swing_rate= ("is_swing","mean"),
).reset_index()
swing_agg["whiff_rate"] = swing_agg["n_whiff"] / swing_agg["n_swing"]

# K% / BB% = 三振|四球 / 打席 （PA終端投球のみで集計）
TERM_K  = {"StrikeSwinging","StrikeLooking"}
TERM_BB = {"BallCalled","BallInDirt","BallinDirt","BallIntentional"}
terminal = val_pitches.copy()
terminal["is_K"]  = ((terminal["strikes_x"]==2) & terminal["pitch_call"].isin(TERM_K)).astype(int)
terminal["is_BB"] = ((terminal["balls_x"]==3)   & terminal["pitch_call"].isin(TERM_BB)).astype(int)
terminal["is_pa_end"] = (
    terminal["pitch_call"].isin({"InPlay","HitByPitch"}) |
    (terminal["is_K"] == 1) |
    (terminal["is_BB"] == 1)
)
pa_only = terminal[terminal["is_pa_end"]]
pa_agg = pa_only.groupby(["batter_code","season_x"]).agg(
    n_pa      = ("is_K","count"),
    actual_K  = ("is_K","mean"),
    actual_BB = ("is_BB","mean"),
).reset_index()

val_agg = swing_agg.merge(pa_agg, on=["batter_code","season_x"], how="left")
merged  = devs_f.merge(val_agg, on=["batter_code","season_x"], how="inner")
merged  = merged[merged["n_pa"] >= 150].copy()

# O-Swing% = Chase+Waste 投球のスイング率（ゾーン頻度で加重平均）
merged["o_swing_pct"] = (
    merged["chase_swing"] * 27.6 + merged["waste_swing"] * 9.3
) / (27.6 + 9.3)
print(f"\n  検証サンプル (150PA以上): {len(merged):,}行 / {merged['batter_code'].nunique():,}選手")
print(f"  O-Swing% 統計: mean={merged['o_swing_pct'].mean():.4f}  "
      f"std={merged['o_swing_pct'].std():.4f}  "
      f"min={merged['o_swing_pct'].min():.4f}  max={merged['o_swing_pct'].max():.4f}")

# ── Step 6: 妥当性検証 ──────────────────────────────────
print(f"\n{SEP}")
print("Step 6: 妥当性検証 — 実績指標との相関")
print(SEP)

VAL_METRICS = [
    ("actual_K",   "実績K%",      "KSI と負相関が望ましい (r<-0.3)"),
    ("whiff_rate", "空振率",      "KSI と負相関が望ましい (r<-0.3)"),
    ("swing_rate", "スイング率",   "KSI とほぼ無相関が望ましい"),
    ("actual_BB",  "実績BB%",     "PDI と正相関が望ましい (r>0.3)"),
    ("swing_rate", "スイング率",   "PDI と適度な負相関 (r<-0.2)"),
    ("whiff_rate", "空振率",      "PDI とほぼ無相関が望ましい"),
]

print(f"\n  KSI (コンタクト実行指数) との相関:")
print(f"  {'指標':<12} {'r':>8} {'p値':>12}  解釈")
print(f"  {'-'*60}")
for col, label, note in [
    ("actual_K",   "実績K%",      "KSI と負相関が望ましい (r<-0.3)"),
    ("whiff_rate", "空振率",      "KSI と負相関が望ましい (r<-0.3)"),
    ("swing_rate", "スイング率",  "KSI とほぼ無相関が望ましい"),
    ("o_swing_pct","O-Swing%",   "KSI とほぼ無相関が望ましい"),
    ("actual_BB",  "実績BB%",    "KSI とほぼ無相関が望ましい"),
    ("PDI",        "PDI",        "KSI と独立 (|r|<0.2 が理想)"),
]:
    if col in merged.columns:
        r, p = stats.pearsonr(merged["KSI"], merged[col])
        print(f"  {label:<12} {r:>+8.4f} {p:>12.2e}  {note}")

print(f"\n  PDI (選球指数) との相関:")
print(f"  {'指標':<12} {'r':>8} {'p値':>12}  解釈")
print(f"  {'-'*60}")
for col, label, note in [
    ("actual_BB",  "実績BB%",   "PDI と正相関が望ましい (r>0.3)"),
    ("o_swing_pct","O-Swing%",  "PDI と負相関が望ましい (r<-0.3)"),
    ("swing_rate", "スイング率", "PDI とほぼ無相関が望ましい"),
    ("whiff_rate", "空振率",    "PDI とほぼ無相関が望ましい"),
    ("actual_K",   "実績K%",    "PDI とほぼ無相関が望ましい"),
    ("KSI",        "KSI",       "PDI と独立 (|r|<0.2 が理想)"),
]:
    if col in merged.columns:
        r, p = stats.pearsonr(merged["PDI"], merged[col])
        print(f"  {label:<12} {r:>+8.4f} {p:>12.2e}  {note}")

# D_z の補助確認
r_de, p_de = stats.pearsonr(merged["D_z"], merged["actual_BB"])
r_de2, _   = stats.pearsonr(merged["D_z"], merged["actual_K"])
print(f"\n  D_z (2スト時判断) vs 実績BB%: r={r_de:+.4f}  (p={p_de:.2e})")
print(f"  D_z (2スト時判断) vs 実績K%:  r={r_de2:+.4f}")

# シーズン別相関推移
print(f"\n  シーズン別 検証相関 (KSI vs K%, PDI vs BB%):")
print(f"  {'season':>8} {'n':>6} {'KSI_vs_K':>10} {'PDI_vs_BB':>11}")
print(f"  {'-'*40}")
for yr, g in merged.groupby("season_x"):
    if len(g) < 30:
        continue
    r1, _ = stats.pearsonr(g["KSI"], g["actual_K"])
    r2, _ = stats.pearsonr(g["PDI"], g["actual_BB"])
    print(f"  {yr:>8}  {len(g):>6}  {r1:>+10.4f}  {r2:>+11.4f}")

# ── Step 7: 2025 ランキング（詳細）────────────────────────
print(f"\n{SEP}")
print("2025年 KSI / PDI ランキング（詳細）")
print(SEP)

parts_name = []
for yr in range(17, 27):
    p = RAW_DIR / f"ALL_F_combined{yr}.parquet"
    if p.exists():
        parts_name.append(pd.read_parquet(p, columns=["batter_code","batter"]).drop_duplicates("batter_code"))
smap = pd.concat(parts_name).drop_duplicates("batter_code")
smap["batter_ps"]   = smap["batter"].str.split(", ").apply(
    lambda x: " ".join(reversed(x)) if isinstance(x,list) and len(x)==2
    else (x[0] if isinstance(x,list) else ""))
smap["batter_code"] = smap["batter_code"].astype(str)

res25 = (devs_f[devs_f["season_x"]==2025]
         .merge(smap[["batter_code","batter_ps"]].drop_duplicates("batter_code"),
                on="batter_code", how="left")
         .merge(val_agg[val_agg["season_x"]==2025][
                    ["batter_code","actual_K","actual_BB","whiff_rate","swing_rate"]],
                on="batter_code", how="left"))
res25["o_swing_pct"] = (
    res25["chase_swing"] * 27.6 + res25["waste_swing"] * 9.3
) / (27.6 + 9.3)
print(f"\n  2025年 対象選手数: {len(res25):,}名")

fmt = lambda x: f"{x:.3f}"

print(f"\n── KSI 上位20 (コンタクト実行力が高い打者) ──")
cols_k = ["batter_ps","n_pitches","KSI","CDI","PDI","comp_B","actual_K","whiff_rate","swing_rate","o_swing_pct"]
print(res25.nlargest(20,"KSI")[cols_k].to_string(index=False, float_format=fmt))

print(f"\n── KSI 下位15 (コンタクト実行力が低い打者) ──")
print(res25.nsmallest(15,"KSI")[cols_k].to_string(index=False, float_format=fmt))

print(f"\n── PDI 上位20 (選球眼が高い打者) ──")
cols_p = ["batter_ps","n_pitches","PDI","KSI","CDI","comp_E","actual_BB","o_swing_pct","swing_rate","whiff_rate"]
print(res25.nlargest(20,"PDI")[cols_p].to_string(index=False, float_format=fmt))

print(f"\n── PDI 下位15 (ボール球を振る打者) ──")
print(res25.nsmallest(15,"PDI")[cols_p].to_string(index=False, float_format=fmt))

print(f"\n── CDI 上位20 (純スタッフ難度の高い球でコンタクトできる打者) ──")
cdi_extra = "CDI_raw" if "CDI_raw" in res25.columns else "CDI"
cols_c = ["batter_ps","n_pitches","CDI","KSI","PDI",cdi_extra,"actual_K","whiff_rate"]
cols_c = [c for c in cols_c if c in res25.columns]
print(res25.nlargest(20,"CDI")[cols_c].to_string(index=False, float_format=fmt))

print(f"\n── CDI 下位15 (甘い球限定でコンタクトしている打者) ──")
print(res25.nsmallest(15,"CDI")[cols_c].to_string(index=False, float_format=fmt))

# 4象限分析
print(f"\n── 4象限分析 (KSI/PDI とも平均=100 基準) ──")
hi_ksi = res25["KSI"] >= 100
hi_pdi = res25["PDI"] >= 100
quadrants = {
    "高KSI×高PDI (コンタクト◎ 選球◎)": res25[ hi_ksi &  hi_pdi],
    "高KSI×低PDI (コンタクト◎ 選球✗)": res25[ hi_ksi & ~hi_pdi],
    "低KSI×高PDI (コンタクト✗ 選球◎)": res25[~hi_ksi &  hi_pdi],
    "低KSI×低PDI (コンタクト✗ 選球✗)": res25[~hi_ksi & ~hi_pdi],
}
for label, g in quadrants.items():
    if len(g) == 0:
        continue
    print(f"\n  {label}  (n={len(g)})")
    cols_q = ["batter_ps","KSI","PDI","actual_K","actual_BB"]
    top5 = g.sort_values("KSI", ascending=False).head(5)
    print(top5[cols_q].to_string(index=False, float_format=fmt))

# ── Step 8: 保存 ─────────────────────────────────────────
out_cols = ["batter_code","season_x","n_pitches","n_swing","n_two_strike",
            "comp_B","comp_E","comp_D",
            "B_z","E_z","D_z",
            "KSI_raw","PDI_raw","CDI_raw","KSI","PDI","CDI",
            "contact_diff_mean",
            "heart_swing","shadow_swing","chase_swing","waste_swing","o_swing_pct",
            ]
# contact_diff_mean が存在しない場合は除外
out_cols = [c for c in out_cols if c in devs_f.columns]
devs_f[out_cols].to_parquet(DATA_DIR / "batter_indices.parquet", index=False)
print(f"\n[OK] 保存: data/batter_indices.parquet")
