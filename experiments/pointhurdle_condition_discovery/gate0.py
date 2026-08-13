"""Gate 0: is there any selection headroom between Point and Hurdle on real data?
Reads the draft paired panel only. No model is fitted."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = r"E:\CODING\proj\covariate-trust-pilot\results\pointhurdle_condition_discovery"
os.makedirs(OUT, exist_ok=True)

panel = pd.read_parquet(os.path.join(HERE, "paired_panel_draft.parquet"))
TAU = 2.0
SEED = 20260813
rng = np.random.default_rng(SEED)

panel["winner"] = np.where(panel["G"] > TAU, "hurdle",
                           np.where(panel["G"] < -TAU, "point", "neutral"))

print("=" * 74)
print("C1  static models and oracles   (loss = RMSE on realized y, mean over rows)")
print("=" * 74)


def agg(df, col):
    return df[col].mean()


rows = []
for ds, d in list(panel.groupby("dataset_id")) + [("MACRO", panel)]:
    Lp, Lh = agg(d, "loss_point"), agg(d, "loss_hurdle")
    L_static_best = min(Lp, Lh)
    L_origin_oracle = d[["loss_point", "loss_hurdle"]].min(axis=1).mean()
    # series oracle: pick one model per series using that series' own outer outcomes
    s = d.groupby(["dataset_id", "series_id"])[["loss_point", "loss_hurdle"]].mean()
    pick_h = (s["loss_hurdle"] < s["loss_point"])
    sm = d.merge(pick_h.rename("pick_h"), left_on=["dataset_id", "series_id"],
                 right_index=True, how="left")
    L_series_oracle = np.where(sm["pick_h"], sm["loss_hurdle"], sm["loss_point"]).mean()
    rows.append(dict(dataset=ds, n_rows=len(d),
                     always_point=Lp, always_hurdle=Lh,
                     test_static_best=L_static_best,
                     series_oracle=L_series_oracle,
                     origin_oracle=L_origin_oracle,
                     hr_vs_static=100 * (L_static_best - L_origin_oracle) / L_static_best,
                     hr_series_vs_static=100 * (L_static_best - L_series_oracle) / L_static_best))

# train-domain global choice: the model chosen on the OTHER dataset, applied here
gl = {}
for ds in panel["dataset_id"].unique():
    other = panel[panel["dataset_id"] != ds]
    gl[ds] = "hurdle" if agg(other, "loss_hurdle") < agg(other, "loss_point") else "point"
for r in rows:
    if r["dataset"] == "MACRO":
        r["train_domain_global"] = np.mean([x["train_domain_global"] for x in rows
                                            if x["dataset"] != "MACRO"])
        r["global_choice"] = "per-dataset"
    else:
        c = gl[r["dataset"]]
        r["train_domain_global"] = r["always_hurdle"] if c == "hurdle" else r["always_point"]
        r["global_choice"] = c
for r in rows:
    r["hr_vs_global"] = 100 * (r["train_domain_global"] - r["origin_oracle"]) / r["train_domain_global"]

hd = pd.DataFrame(rows)[["dataset", "n_rows", "always_point", "always_hurdle",
                         "global_choice", "train_domain_global", "test_static_best",
                         "series_oracle", "origin_oracle",
                         "hr_vs_static", "hr_series_vs_static", "hr_vs_global"]]
print(hd.to_string(index=False, float_format=lambda x: "%.4f" % x))
hd.to_csv(os.path.join(OUT, "oracle_headroom_by_dataset.csv"), index=False)

print("\n" + "=" * 74)
print("C3  winner distribution  (tau = %.0f%%)" % TAU)
print("=" * 74)
wr = []
for ds, d in list(panel.groupby("dataset_id")) + [("MACRO", panel)]:
    n = len(d)
    vc = d["winner"].value_counts()
    ser = d.groupby("series_id")["winner"].agg(lambda s: s.nunique() == 1)
    wr.append(dict(dataset=ds, rows=n, series=d["series_id"].nunique(),
                   point_pct=100 * vc.get("point", 0) / n,
                   neutral_pct=100 * vc.get("neutral", 0) / n,
                   hurdle_pct=100 * vc.get("hurdle", 0) / n,
                   point_rows=int(vc.get("point", 0)),
                   hurdle_rows=int(vc.get("hurdle", 0)),
                   series_consistent_pct=100 * ser.mean()))
wd = pd.DataFrame(wr)
print(wd.to_string(index=False, float_format=lambda x: "%.2f" % x))
wd.to_csv(os.path.join(OUT, "winner_distribution.csv"), index=False)

print("\nsensitivity of practical classes to tau:")
sens = []
for t in (1.0, 2.0, 5.0):
    w = np.where(panel["G"] > t, "hurdle", np.where(panel["G"] < -t, "point", "neutral"))
    vc = pd.Series(w).value_counts()
    sens.append(dict(tau=t, point_pct=100 * vc.get("point", 0) / len(w),
                     neutral_pct=100 * vc.get("neutral", 0) / len(w),
                     hurdle_pct=100 * vc.get("hurdle", 0) / len(w),
                     point_rows=int(vc.get("point", 0)), hurdle_rows=int(vc.get("hurdle", 0))))
print(pd.DataFrame(sens).to_string(index=False, float_format=lambda x: "%.2f" % x))

print("\n" + "=" * 74)
print("C3  winner stability across origins within a series")
print("=" * 74)
st = []
for ds, d in panel.groupby("dataset_id"):
    p = d.pivot_table(index="series_id", columns="outer_origin_id", values="G")
    cols = list(p.columns)
    cor = p.corr(method="spearman")
    st.append(dict(dataset=ds,
                   rho_o1_o2=cor.iloc[0, 1], rho_o1_o3=cor.iloc[0, 2], rho_o2_o3=cor.iloc[1, 2],
                   sign_all_same_pct=100 * (np.sign(p).abs().sum(axis=1).eq(3)
                                            & (np.sign(p).nunique(axis=1) == 1)).mean()))
sd = pd.DataFrame(st)
print(sd.to_string(index=False, float_format=lambda x: "%.3f" % x))
sd.to_csv(os.path.join(OUT, "winner_stability.csv"), index=False)

print("\n" + "=" * 74)
print("C4  cluster bootstrap on headroom (series-level resample, 2000 draws)")
print("=" * 74)
B = 2000
for ds in ["m5", "favorita"]:
    d = panel[panel["dataset_id"] == ds]
    sids = d["series_id"].unique()
    idx = {s: g.index.values for s, g in d.groupby("series_id")}
    hrs = []
    for _ in range(B):
        pick = rng.choice(sids, size=len(sids), replace=True)
        take = np.concatenate([idx[s] for s in pick])
        dd = d.loc[take]
        Ls = min(dd["loss_point"].mean(), dd["loss_hurdle"].mean())
        Lo = dd[["loss_point", "loss_hurdle"]].min(axis=1).mean()
        hrs.append(100 * (Ls - Lo) / Ls)
    lo, hi = np.percentile(hrs, [2.5, 97.5])
    print("  %-9s origin-oracle headroom vs test-static-best: %.3f%%  95%% CI [%.3f, %.3f]"
          % (ds, np.mean(hrs), lo, hi))

print("\nseed = %d" % SEED)
panel.to_parquet(os.path.join(OUT, "paired_panel.parquet"), index=False)
print("panel written: %s" % os.path.join(OUT, "paired_panel.parquet"))
