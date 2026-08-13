"""Oracle family separation, per-origin policy results, cumulative regret,
label-margin diagnosis, and exp-weight normalization sensitivity.
Read-only on existing results.  No model is trained."""
import json, os
import numpy as np
import pandas as pd

R = r"E:\CODING\proj\covariate-trust-pilot\results"
EVS = os.path.join(R, "external_validity_screen")
OUT = os.path.join(R, "pointhurdle_recoverability")
SEED = 20260813
TAU = 2.0
rng = np.random.default_rng(SEED)
W = np.round(np.arange(0, 1.0001, 0.05), 2)

RAW = {"m5": os.path.join(EVS, "rule_replication", "independent_raw_predictions.parquet"),
       "favorita": os.path.join(EVS, "favorita_independent", "independent_raw_predictions.parquet")}
steps = []
for ds, p in RAW.items():
    d = pd.read_parquet(p)
    d = d[d["target_mask"]].copy()
    d["dataset_id"] = ds
    steps.append(d)
steps = pd.concat(steps, ignore_index=True)

key = [steps["dataset_id"], steps["series_id"], steps["origin"]]
rmse = lambda e: np.sqrt(e.groupby(key).mean())
Lp = rmse((steps["y_observed"] - steps["point_mean_prediction"]) ** 2)
Lh = rmse((steps["y_observed"] - steps["hurdle_mean_prediction"]) ** 2)
comb = pd.DataFrame({w: rmse((steps["y_observed"] -
                              (w * steps["hurdle_mean_prediction"]
                               + (1 - w) * steps["point_mean_prediction"])) ** 2)
                     for w in W})
comb.index.names = Lp.index.names = Lh.index.names = ["dataset_id", "series_id", "origin_id"]
P = pd.DataFrame({"loss_point": Lp, "loss_hurdle": Lh}).sort_index()
comb = comb.reindex(P.index)
ORIG = {ds: sorted(P.loc[ds].index.get_level_values("origin_id").unique())
        for ds in P.index.get_level_values(0).unique()}

# ================================================== 1/2  oracle families, separated
print("=" * 86)
print("FAMILY-SEPARATED ORACLE LADDERS")
print("=" * 86)
rows = []
for ds in ORIG:
    p, c = P.loc[ds], comb.loc[ds]
    other = [d for d in ORIG if d != ds][0]
    po = P.loc[other]
    gl_pick = "hurdle" if po["loss_hurdle"].mean() < po["loss_point"].mean() else "point"
    L_global = (p["loss_hurdle"] if gl_pick == "hurdle" else p["loss_point"]).mean()
    L_heldout_static_best = min(p["loss_point"].mean(), p["loss_hurdle"].mean())
    # HARD family
    s = p.groupby(level="series_id")[["loss_point", "loss_hurdle"]].sum()
    ph = (s["loss_hurdle"] < s["loss_point"]).reindex(
        p.index.get_level_values("series_id")).values
    L_hard_series = np.where(ph, p["loss_hurdle"], p["loss_point"]).mean()
    L_hard_origin = p.min(axis=1).mean()
    # CONVEX family
    w_tr = float(comb.loc[other].mean().idxmin())          # train-domain, deployable
    L_convex_global_train = c[w_tr].mean()
    w_ho = float(c.mean().idxmin())                        # held-out best, diagnostic
    L_convex_global_heldout = c[w_ho].mean()
    ser = c.groupby(level="series_id").mean().idxmin(axis=1)
    _cv = c.to_numpy(); _cp = {float(w_): j for j, w_ in enumerate(c.columns)}
    _sid = c.index.get_level_values("series_id")
    L_convex_series = np.array(
        [_cv[i, _cp[float(ser.loc[_sid[i]])]] for i in range(len(c))]).mean()
    L_convex_origin = c.min(axis=1).mean()
    rows.append(dict(dataset=ds, family="hard_selection",
                     train_domain_global=L_global,
                     heldout_static_best=L_heldout_static_best,
                     series_oracle=L_hard_series, origin_oracle=L_hard_origin,
                     w_train_domain=np.nan, w_heldout_best=np.nan))
    rows.append(dict(dataset=ds, family="convex_combination",
                     train_domain_global=L_convex_global_train,
                     heldout_static_best=L_convex_global_heldout,
                     series_oracle=L_convex_series, origin_oracle=L_convex_origin,
                     w_train_domain=w_tr, w_heldout_best=w_ho))
lad = pd.DataFrame(rows)
lad["H_total_pct"] = 100 * (lad["train_domain_global"] - lad["origin_oracle"]) / lad["train_domain_global"]
lad["H_static_pct"] = 100 * (lad["train_domain_global"] - lad["series_oracle"]) / lad["train_domain_global"]
lad["H_dynamic_pct"] = 100 * (lad["series_oracle"] - lad["origin_oracle"]) / lad["train_domain_global"]
lad["static_recoverability_ratio"] = lad["H_static_pct"] / lad["H_total_pct"]
print(lad.to_string(index=False, float_format=lambda x: "%.4f" % x))
lad.to_csv(os.path.join(OUT, "oracle_family_ladders.csv"), index=False)

HARD_ORACLE = {r["dataset"]: r["origin_oracle"] for _, r in lad.iterrows()
               if r["family"] == "hard_selection"}
CONV_ORACLE = {r["dataset"]: r["origin_oracle"] for _, r in lad.iterrows()
               if r["family"] == "convex_combination"}
GLOBAL = {r["dataset"]: r["train_domain_global"] for _, r in lad.iterrows()
          if r["family"] == "hard_selection"}


# ================================================== 3  policy by dataset x origin
def run_policies(test_ds, train_ds):
    p, c = P.loc[test_ds], comb.loc[test_ds]
    w_tr = float(comb.loc[train_ds].mean().idxmin())
    po = P.loc[train_ds]
    gl_pick = "hurdle" if po["loss_hurdle"].mean() < po["loss_point"].mean() else "point"
    origins = ORIG[test_ds]
    res = {}
    res["P0 always Point"] = ("hard", p["loss_point"])
    res["P1 always Hurdle"] = ("hard", p["loss_hurdle"])
    res["P2 train-domain global choice"] = ("hard", p["loss_hurdle"] if gl_pick == "hurdle"
                                            else p["loss_point"])
    res["P3 50:50 average"] = ("convex", c[0.5])
    res["P4 train-domain convex w=%.2f" % w_tr] = ("convex", c[w_tr])
    # D1 discounted hard selector
    for gamma in (0.5, 0.9):
        out = []
        for k, o in enumerate(origins):
            sl = p.xs(o, level="origin_id")
            if k == 0:
                ch = pd.Series(gl_pick, index=sl.index)
            else:
                cp = np.zeros(len(sl)); chh = np.zeros(len(sl))
                for j, oo in enumerate(origins[:k]):
                    wt = gamma ** (k - j - 1)
                    hh = p.xs(oo, level="origin_id").reindex(sl.index)
                    cp += wt * hh["loss_point"].values; chh += wt * hh["loss_hurdle"].values
                ch = pd.Series(np.where(chh < cp, "hurdle", "point"), index=sl.index)
            out.append(pd.Series(np.where(ch.values == "hurdle", sl["loss_hurdle"].values,
                                          sl["loss_point"].values),
                                 index=pd.MultiIndex.from_product([sl.index, [o]],
                                                                  names=["series_id", "origin_id"])))
        res["D1 discounted selector g=%.1f" % gamma] = ("hard", pd.concat(out).reindex(p.index))
    # D3 exp-weighted, several normalizations
    for norm in ("raw", "per_series_scale", "per_origin_scale"):
        for eta in (2.0, 8.0):
            out = []
            for k, o in enumerate(origins):
                sl = p.xs(o, level="origin_id")
                if k == 0:
                    wv = np.full(len(sl), w_tr)
                else:
                    h = p[p.index.get_level_values("origin_id").isin(origins[:k])]
                    cp = h["loss_point"].groupby(level="series_id").mean().reindex(sl.index).values
                    ch = h["loss_hurdle"].groupby(level="series_id").mean().reindex(sl.index).values
                    if norm == "per_series_scale":
                        sc = np.maximum((cp + ch) / 2, 1e-9); cp, ch = cp / sc, ch / sc
                    elif norm == "per_origin_scale":
                        sc = max(np.nanmean(np.concatenate([cp, ch])), 1e-9); cp, ch = cp / sc, ch / sc
                    m = np.minimum(cp, ch)
                    a = np.exp(-eta * (ch - m)); b = np.exp(-eta * (cp - m))
                    wv = np.where(np.isfinite(a / (a + b)), a / (a + b), 0.5)
                wq = np.clip(np.round(np.asarray(wv, dtype=np.float64) / 0.05) * 0.05,
                             0, 1).round(2)
                cc = c.xs(o, level="origin_id").reindex(sl.index)
                cv = cc.to_numpy()
                colpos = {float(w_): j for j, w_ in enumerate(cc.columns)}
                out.append(pd.Series([cv[i, colpos[float(wq[i])]] for i in range(len(sl))],
                                     index=pd.MultiIndex.from_product([sl.index, [o]],
                                                                      names=["series_id", "origin_id"])))
            res["D3 expw eta=%.1f norm=%s" % (eta, norm)] = ("convex", pd.concat(out).reindex(p.index))
    res["ORACLE hard origin"] = ("hard", p.min(axis=1))
    res["ORACLE convex origin"] = ("convex", c.min(axis=1))
    return res, gl_pick


print("\n" + "=" * 86)
print("POLICY BY DATASET x ORIGIN, capture measured against the MATCHING family oracle")
print("=" * 86)
byo, cum = [], []
for train_ds, test_ds in (("m5", "favorita"), ("favorita", "m5")):
    res, gl_pick = run_policies(test_ds, train_ds)
    p = P.loc[test_ds]
    glv = (p["loss_hurdle"] if gl_pick == "hurdle" else p["loss_point"])
    for name, (fam, v) in res.items():
        orc = (p.min(axis=1) if fam == "hard" else comb.loc[test_ds].min(axis=1))
        for o in ORIG[test_ds]:
            m = v.index.get_level_values("origin_id") == o
            g, l, oc = glv[m].mean(), v[m].mean(), orc[m].mean()
            byo.append(dict(train=train_ds, test=test_ds, origin=o, policy=name, family=fam,
                            loss=l, global_loss=g, family_oracle=oc,
                            rel_improve_pct=100 * (g - l) / g,
                            capture_pct=100 * (g - l) / (g - oc) if g - oc > 1e-9 else np.nan))
        # cumulative regret vs family oracle, in loss units, over origins
        run = 0.0
        for o in ORIG[test_ds]:
            m = v.index.get_level_values("origin_id") == o
            run += float(v[m].mean() - orc[m].mean())
            cum.append(dict(train=train_ds, test=test_ds, origin=o, policy=name, family=fam,
                            cumulative_regret=run))
bo = pd.DataFrame(byo)
bo.to_csv(os.path.join(OUT, "policy_by_dataset_origin.csv"), index=False)
pd.DataFrame(cum).to_csv(os.path.join(OUT, "cumulative_regret.csv"), index=False)

sel = bo[bo["policy"].isin(["P2 train-domain global choice", "P3 50:50 average",
                            "D1 discounted selector g=0.9",
                            "D3 expw eta=8.0 norm=raw", "ORACLE hard origin",
                            "ORACLE convex origin"])]
print(sel.pivot_table(index=["test", "policy"], columns="origin",
                      values="rel_improve_pct").to_string(float_format=lambda x: "%.4f" % x))
print("\ncapture against the MATCHING family oracle (macro over origins):")
mac = (bo.groupby(["policy", "family"])
       .agg(macro_improve=("rel_improve_pct", "mean"),
            worst_improve=("rel_improve_pct", "min"),
            macro_capture=("capture_pct", "mean")).reset_index()
       .sort_values("macro_improve", ascending=False))
print(mac.to_string(index=False, float_format=lambda x: "%.4f" % x))
mac.to_csv(os.path.join(OUT, "policy_macro_family_corrected.csv"), index=False)

# ================================================== 6  normalization sensitivity
ns = bo[bo["policy"].str.startswith("D3")].groupby("policy").agg(
    macro_improve=("rel_improve_pct", "mean"), worst=("rel_improve_pct", "min"),
    capture=("capture_pct", "mean")).reset_index()
ns.to_csv(os.path.join(OUT, "exp_weight_normalization_sensitivity.csv"), index=False)
print("\n" + "=" * 86)
print("EXP-WEIGHT NORMALIZATION SENSITIVITY")
print("=" * 86)
print(ns.to_string(index=False, float_format=lambda x: "%.4f" % x))

# ================================================== 4  label margin
print("\n" + "=" * 86)
print("PRACTICAL-LABEL MARGIN: how close are rows to the +-2% boundary?")
print("=" * 86)
G = 100 * (1 - P["loss_hurdle"] / P["loss_point"])
mar = []
for ds in ORIG:
    g = G.loc[ds]
    for band in (0.5, 1.0, 2.0):
        near = ((g.abs() - TAU).abs() <= band).mean()
        mar.append(dict(dataset=ds, band_pp=band, frac_within_band_of_tau=100 * near))
    mar.append(dict(dataset=ds, band_pp="abs(G)<=1", frac_within_band_of_tau=100 * (g.abs() <= 1).mean()))
mr = pd.DataFrame(mar)
print(mr.to_string(index=False, float_format=lambda x: "%.2f" % x))
mr.to_csv(os.path.join(OUT, "seed_margin_flip.csv"), index=False)
print("\nwritten to", OUT)
