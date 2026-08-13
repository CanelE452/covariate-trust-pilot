"""Recoverability analysis: Step 1-4.
A4 reproduction, seed audit, oracle ladders (hard + soft), stability, policy benchmark.
Read-only on existing results.  No model is trained."""
import json, os
import numpy as np
import pandas as pd

R = r"E:\CODING\proj\covariate-trust-pilot\results"
EVS = os.path.join(R, "external_validity_screen")
PRIOR = os.path.join(R, "pointhurdle_condition_discovery")
OUT = os.path.join(R, "pointhurdle_recoverability")
os.makedirs(OUT, exist_ok=True)
SEED = 20260813
TAU = 2.0
rng = np.random.default_rng(SEED)

RAW = {"m5": os.path.join(EVS, "rule_replication", "independent_raw_predictions.parquet"),
       "favorita": os.path.join(EVS, "favorita_independent", "independent_raw_predictions.parquet")}

# ============================================================ A4 reproduction
print("=" * 78)
print("A4  independent reproduction of the 3-origin panel")
print("=" * 78)
rows = []
steps = []
for ds, p in RAW.items():
    df = pd.read_parquet(p)
    m = df[df["target_mask"]].copy()
    m["dataset_id"] = ds
    steps.append(m)
    g = m.groupby(["dataset_id", "series_id", "origin"], as_index=False).apply(
        lambda d: pd.Series({
            "loss_point": np.sqrt(((d["y_observed"] - d["point_mean_prediction"]) ** 2).mean()),
            "loss_hurdle": np.sqrt(((d["y_observed"] - d["hurdle_mean_prediction"]) ** 2).mean()),
            "n_steps": len(d)}), include_groups=False)
    rows.append(g)
panel = pd.concat(rows, ignore_index=True).rename(columns={"origin": "origin_id"})
allsteps = pd.concat(steps, ignore_index=True)
panel["G"] = 100 * (1 - panel["loss_hurdle"] / panel["loss_point"])
panel["winner"] = np.where(panel["G"] > TAU, "hurdle",
                           np.where(panel["G"] < -TAU, "point", "neutral"))

prior = pd.read_parquet(os.path.join(PRIOR, "paired_panel.parquet"))
chk = panel.merge(prior[["dataset_id", "series_id", "outer_origin_id", "G"]],
                  left_on=["dataset_id", "series_id", "origin_id"],
                  right_on=["dataset_id", "series_id", "outer_origin_id"],
                  suffixes=("", "_prior"))
maxdiff = float((chk["G"] - chk["G_prior"]).abs().max())
print("  rows %d   matched to prior panel %d   max |G - G_prior| = %.3e"
      % (len(panel), len(chk), maxdiff))
repro = []
for name, new, old in [
    ("point win %", 100 * (panel["winner"] == "point").mean(), 33.85),
    ("hurdle win %", 100 * (panel["winner"] == "hurdle").mean(), 44.82),
    ("neutral %", 100 * (panel["winner"] == "neutral").mean(), 21.33),
]:
    repro.append(dict(quantity=name, recomputed=new, reported=old, abs_diff=abs(new - old)))

# ============================================================ D2/D3 oracle ladder
print("\n" + "=" * 78)
print("D2/D3  hard-selection oracle ladder and headroom decomposition")
print("=" * 78)
lad, dec = [], []
for ds, d in list(panel.groupby("dataset_id")) + [("MACRO", panel)]:
    Lp, Lh = d["loss_point"].mean(), d["loss_hurdle"].mean()
    if ds == "MACRO":
        gl = np.mean([x["global"] for x in lad])
    else:
        other = panel[panel["dataset_id"] != ds]
        pick = "hurdle" if other["loss_hurdle"].mean() < other["loss_point"].mean() else "point"
        gl = Lh if pick == "hurdle" else Lp
    s = d.groupby(["dataset_id", "series_id"])[["loss_point", "loss_hurdle"]].sum()
    ph = (s["loss_hurdle"] < s["loss_point"]).rename("ph")
    dd = d.merge(ph, left_on=["dataset_id", "series_id"], right_index=True)
    Lser = np.where(dd["ph"], dd["loss_hurdle"], dd["loss_point"]).mean()
    Lori = d[["loss_point", "loss_hurdle"]].min(axis=1).mean()
    lad.append(dict(dataset=ds, always_point=Lp, always_hurdle=Lh, global_=gl,
                    heldout_static_best=min(Lp, Lh), series_oracle=Lser, origin_oracle=Lori))
    lad[-1]["global"] = gl
    Ht = (gl - Lori) / gl
    Hs = (gl - Lser) / gl
    Hd = (Lser - Lori) / gl
    dec.append(dict(dataset=ds, global_loss=gl, series_oracle=Lser, origin_oracle=Lori,
                    H_total_pct=100 * Ht, H_static_pct=100 * Hs, H_dynamic_pct=100 * Hd,
                    R_static=Hs / Ht if Ht > 1e-9 else np.nan,
                    R_dynamic=1 - (Hs / Ht) if Ht > 1e-9 else np.nan))
ladder = pd.DataFrame(lad).drop(columns=["global_"])
decomp = pd.DataFrame(dec)
print(ladder.to_string(index=False, float_format=lambda x: "%.4f" % x))
print()
print(decomp.to_string(index=False, float_format=lambda x: "%.4f" % x))
ladder.to_csv(os.path.join(OUT, "oracle_ladder_by_dataset.csv"), index=False)
decomp.to_csv(os.path.join(OUT, "recoverability_decomposition.csv"), index=False)

# ============================================================ D4 soft ladder
print("\n" + "=" * 78)
print("D4  soft-combination oracle ladder (forecasts combined, then metric recomputed)")
print("=" * 78)
WGRID = np.round(np.arange(0, 1.0001, 0.05), 2)


def comb_loss(sub, w):
    yhat = w * sub["hurdle_mean_prediction"] + (1 - w) * sub["point_mean_prediction"]
    se = (sub["y_observed"] - yhat) ** 2
    per = se.groupby([sub["dataset_id"], sub["series_id"], sub["origin"]]).mean()
    return np.sqrt(per)


soft = {}
for ds, d in allsteps.groupby("dataset_id"):
    curves = {w: comb_loss(d, w) for w in WGRID}
    mean_by_w = {w: float(v.mean()) for w, v in curves.items()}
    w_star_global = min(mean_by_w, key=mean_by_w.get)
    L50 = mean_by_w[0.5]
    # series-fixed convex oracle
    stack = pd.DataFrame(curves)
    ser_mean = stack.groupby(level=[0, 1]).mean()
    w_ser = ser_mean.idxmin(axis=1)
    L_ser = np.array([stack.loc[i, w_ser.loc[(i[0], i[1])]] for i in stack.index]).mean()
    L_ori = stack.min(axis=1).mean()
    soft[ds] = dict(dataset=ds, L_50_50=L50, w_global_heldout=w_star_global,
                    L_global_convex_heldout=mean_by_w[w_star_global],
                    L_series_convex_oracle=L_ser, L_origin_convex_oracle=L_ori,
                    w_curve=mean_by_w)
    print("  %-9s 50:50 %.4f | heldout-best w=%.2f -> %.4f | series-convex %.4f | origin-convex %.4f"
          % (ds, L50, w_star_global, mean_by_w[w_star_global], L_ser, L_ori))
sc = pd.DataFrame([{k: v for k, v in s.items() if k != "w_curve"} for s in soft.values()])
sc.to_csv(os.path.join(OUT, "soft_combination_ladder.csv"), index=False)

# ============================================================ D5 stability
print("\n" + "=" * 78)
print("D5  winner stability and transitions")
print("=" * 78)
stab, trans = [], []
for ds, d in panel.groupby("dataset_id"):
    p = d.pivot_table(index="series_id", columns="origin_id", values="G")
    w = d.pivot_table(index="series_id", columns="origin_id", values="winner", aggfunc="first")
    cols = list(w.columns)
    same = (w.nunique(axis=1) == 1)
    agree = np.mean([(w[a] == w[b]).mean() for a, b in zip(cols, cols[1:])])
    lag1 = np.mean([p[a].corr(p[b], method="spearman") for a, b in zip(cols, cols[1:])])
    sw = (w[cols].values[:, :-1] != w[cols].values[:, 1:]).sum(axis=1)
    stab.append(dict(dataset=ds, series=len(w), same_winner_all_origins_pct=100 * same.mean(),
                     pairwise_agreement_pct=100 * agree, gain_lag1_spearman=lag1,
                     mean_switches=sw.mean(), zero_switch_pct=100 * (sw == 0).mean()))
    for a, b in zip(cols, cols[1:]):
        t = pd.crosstab(w[a], w[b], normalize="index")
        for i in t.index:
            for j in t.columns:
                trans.append(dict(dataset=ds, from_origin=a, to_origin=b,
                                  from_state=i, to_state=j, prob=t.loc[i, j]))
sd = pd.DataFrame(stab)
print(sd.to_string(index=False, float_format=lambda x: "%.3f" % x))
sd.to_csv(os.path.join(OUT, "winner_stability.csv"), index=False)
pd.DataFrame(trans).to_csv(os.path.join(OUT, "winner_transition_matrix.csv"), index=False)

# ============================================================ C seed audit
print("\n" + "=" * 78)
print("C  seed audit (M5, 3 model seeds, series level; no retraining)")
print("=" * 78)
sb = pd.read_csv(os.path.join(EVS, "seed_robustness", "per_series_by_seed.csv"))
cols = ["delta_relative_seed0", "delta_relative_seed1", "delta_relative_seed2"]
X = sb[cols].to_numpy(float)
grand = np.nanmean(X)
series_mean = np.nanmean(X, axis=1)
seed_mean = np.nanmean(X, axis=0)
v_series = np.nanvar(series_mean, ddof=1)
v_seed = np.nanvar(seed_mean, ddof=1)
v_resid = np.nanmean((X - series_mean[:, None] - seed_mean[None, :] + grand) ** 2)
tot = v_series + v_seed + v_resid
# practical winner flip across seeds (delta_relative sign vs +-2%)
W = np.where(X * 100 > TAU, "hurdle", np.where(X * 100 < -TAU, "point", "neutral"))
flip = (pd.DataFrame(W).nunique(axis=1) > 1).mean()
vc = pd.DataFrame([dict(component="series (stable)", variance=v_series, share=v_series / tot),
                   dict(component="seed", variance=v_seed, share=v_seed / tot),
                   dict(component="residual series x seed", variance=v_resid, share=v_resid / tot)])
print(vc.to_string(index=False, float_format=lambda x: "%.6f" % x))
print("  practical winner changes across seeds: %.2f%% of series" % (100 * flip))
print("  seed variance / (series x seed residual) = %.4f" % (v_seed / v_resid))
vc.to_csv(os.path.join(OUT, "variance_components.csv"), index=False)

json.dump(dict(seed_variance_share=float(v_seed / tot),
               winner_flip_across_seeds_pct=float(100 * flip),
               seed_over_residual=float(v_seed / v_resid),
               gate_c4_seed_variance_lt_20pct_of_interaction=bool(v_seed / v_resid < 0.20),
               gate_c4_winner_flip_lt_20pct=bool(flip < 0.20)),
          open(os.path.join(OUT, "seed_gate.json"), "w"), indent=2)

panel.to_parquet(os.path.join(OUT, "multi_origin_paired_panel.parquet"), index=False)
pd.DataFrame(repro).to_csv(os.path.join(OUT, "reproduction_check.csv"), index=False)
print("\nreproduction check:")
print(pd.DataFrame(repro).to_string(index=False, float_format=lambda x: "%.4f" % x))
allsteps.to_parquet(os.path.join(OUT, "_steps_cache.parquet"), index=False)
print("\nwritten to", OUT)
