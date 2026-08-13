"""Phase E: feasible strategy benchmark, sequential on the held-out dataset.
Only origins < o of the held-out dataset may inform the decision at origin o.
No model is trained."""
import json, os
import numpy as np
import pandas as pd

R = r"E:\CODING\proj\covariate-trust-pilot\results"
OUT = os.path.join(R, "pointhurdle_recoverability")
SEED = 20260813
TAU = 2.0
rng = np.random.default_rng(SEED)

panel = pd.read_parquet(os.path.join(OUT, "multi_origin_paired_panel.parquet"))
steps = pd.read_parquet(os.path.join(OUT, "_steps_cache.parquet"))
WGRID = np.round(np.arange(0, 1.0001, 0.05), 2)

# per (dataset, series, origin) combined loss for every w on the grid
cl = {}
for w in WGRID:
    yhat = w * steps["hurdle_mean_prediction"] + (1 - w) * steps["point_mean_prediction"]
    se = (steps["y_observed"] - yhat) ** 2
    cl[w] = np.sqrt(se.groupby([steps["dataset_id"], steps["series_id"],
                                steps["origin"]]).mean())
comb = pd.DataFrame(cl)
comb.index.names = ["dataset_id", "series_id", "origin_id"]
P = panel.set_index(["dataset_id", "series_id", "origin_id"]).sort_index()
comb = comb.reindex(P.index)

ORIG = {ds: sorted(d["origin_id"].unique()) for ds, d in panel.groupby("dataset_id")}
res = []


def report(train_ds, test_ds, name, loss_vec, switch=np.nan):
    te = P.loc[test_ds]
    gl_pick = ("hurdle" if P.loc[train_ds, "loss_hurdle"].mean()
               < P.loc[train_ds, "loss_point"].mean() else "point")
    gl = te["loss_hurdle"].mean() if gl_pick == "hurdle" else te["loss_point"].mean()
    orc = te[["loss_point", "loss_hurdle"]].min(axis=1).mean()
    L = float(np.mean(loss_vec))
    res.append(dict(train=train_ds, test=test_ds, policy=name, loss=L, global_loss=gl,
                    origin_oracle=orc, rel_improve_pct=100 * (gl - L) / gl,
                    oracle_gap_capture_pct=(100 * (gl - L) / (gl - orc)
                                            if gl - orc > 1e-9 else np.nan),
                    switch_rate=switch))


for train_ds, test_ds in (("m5", "favorita"), ("favorita", "m5")):
    tr, te = P.loc[train_ds], P.loc[test_ds]
    ctr, cte = comb.loc[train_ds], comb.loc[test_ds]
    gl_pick = "hurdle" if tr["loss_hurdle"].mean() < tr["loss_point"].mean() else "point"

    # ---- E0 baselines
    report(train_ds, test_ds, "P0 always Point", te["loss_point"].values)
    report(train_ds, test_ds, "P1 always Hurdle", te["loss_hurdle"].values)
    report(train_ds, test_ds, "P2 train-domain global choice",
           (te["loss_hurdle"] if gl_pick == "hurdle" else te["loss_point"]).values)
    report(train_ds, test_ds, "P3 50:50 forecast average", cte[0.5].values)
    w_tr = float(ctr.mean().idxmin())
    report(train_ds, test_ds, "P4 train-domain global convex w=%.2f" % w_tr, cte[w_tr].values)

    # ---- E3 dynamic selection, sequential over origins
    origins = ORIG[test_ds]
    for gamma in (0.0, 0.5, 0.9, 1.0):
        pick_loss, sw = [], []
        prev = None
        for k, o in enumerate(origins):
            sl = te.xs(o, level="origin_id")
            if k == 0:
                choice = pd.Series(gl_pick, index=sl.index)
            else:
                hist = te[te.index.get_level_values("origin_id").isin(origins[:k])]
                cp = np.zeros(len(sl)); ch = np.zeros(len(sl))
                for j, oo in enumerate(origins[:k]):
                    wt = gamma ** (k - j - 1) if gamma > 0 else (1.0 if j == k - 1 else 0.0)
                    hh = hist.xs(oo, level="origin_id").reindex(sl.index)
                    cp += wt * hh["loss_point"].values
                    ch += wt * hh["loss_hurdle"].values
                choice = pd.Series(np.where(ch < cp, "hurdle", "point"), index=sl.index)
            if prev is not None:
                sw.append(float((choice.values != prev.values).mean()))
            prev = choice
            pick_loss.append(np.where(choice.values == "hurdle",
                                      sl["loss_hurdle"].values, sl["loss_point"].values))
        report(train_ds, test_ds, "D1 discounted-loss selector gamma=%.2f" % gamma,
               np.concatenate(pick_loss), np.mean(sw) if sw else np.nan)

    # ---- E4 dynamic combination: rolling convex weight over previous K origins
    for K in (1, 2, 3):
        out, sw, prevw = [], [], None
        for k, o in enumerate(origins):
            sl_idx = te.xs(o, level="origin_id").index
            if k == 0:
                wsel = pd.Series(w_tr, index=sl_idx)
            else:
                use = origins[max(0, k - K):k]
                acc = None
                for oo in use:
                    cc = cte.xs(oo, level="origin_id").reindex(sl_idx)
                    acc = cc if acc is None else acc + cc
                wsel = acc.idxmin(axis=1).astype(float)
            if prevw is not None:
                sw.append(float((wsel.values != prevw.values).mean()))
            prevw = wsel
            cc = cte.xs(o, level="origin_id").reindex(sl_idx)
            out.append(np.array([cc.loc[i, wsel.loc[i]] for i in sl_idx]))
        report(train_ds, test_ds, "D2 rolling convex weight K=%d" % K,
               np.concatenate(out), np.mean(sw) if sw else np.nan)

    # ---- E4b exponentially weighted average over cumulative loss
    for eta in (0.5, 2.0, 8.0):
        out = []
        for k, o in enumerate(origins):
            sl = te.xs(o, level="origin_id")
            if k == 0:
                wv = np.full(len(sl), w_tr)
            else:
                hist = te[te.index.get_level_values("origin_id").isin(origins[:k])]
                cp = hist["loss_point"].groupby(level="series_id").mean().reindex(sl.index).values
                ch = hist["loss_hurdle"].groupby(level="series_id").mean().reindex(sl.index).values
                # numerically stable 2-way softmax: subtract the row min before exp,
                # otherwise exp(-eta*L) underflows to 0 for both arms and gives 0/0
                mlo = np.minimum(cp, ch)
                a = np.exp(-eta * (ch - mlo)); b = np.exp(-eta * (cp - mlo))
                wv = a / (a + b)
                wv = np.where(np.isfinite(wv), wv, 0.5)
            wq = np.clip(np.round(wv / 0.05) * 0.05, 0, 1).round(2)
            cc = cte.xs(o, level="origin_id").reindex(sl.index)
            out.append(np.array([cc.iloc[i][wq[i]] for i in range(len(sl))]))
        report(train_ds, test_ds, "D3 exp-weighted average eta=%.1f" % eta,
               np.concatenate(out))

    # ---- diagnostic oracles
    report(train_ds, test_ds, "ORACLE origin (hard) - diagnostic",
           te[["loss_point", "loss_hurdle"]].min(axis=1).values)
    report(train_ds, test_ds, "ORACLE origin convex - diagnostic", cte.min(axis=1).values)

df = pd.DataFrame(res)
df.to_csv(os.path.join(OUT, "policy_results_by_dataset.csv"), index=False)
macro = (df.groupby("policy").agg(macro_rel_improve=("rel_improve_pct", "mean"),
                                  worst_rel_improve=("rel_improve_pct", "min"),
                                  macro_capture=("oracle_gap_capture_pct", "mean"),
                                  mean_switch=("switch_rate", "mean")).reset_index()
         .sort_values("macro_rel_improve", ascending=False))
macro.to_csv(os.path.join(OUT, "policy_macro_results.csv"), index=False)
pd.set_option("display.width", 190)
print(df.to_string(index=False, float_format=lambda x: "%.4f" % x))
print("\n" + "=" * 78)
print("MACRO over the two transfer directions")
print("=" * 78)
print(macro.to_string(index=False, float_format=lambda x: "%.4f" % x))

# ---- cluster bootstrap on the two headline comparisons
print("\ncluster bootstrap (series, 2000 draws) on 50:50 and best dynamic vs global")
B = 2000
boot = []
for test_ds in ("m5", "favorita"):
    te = P.loc[test_ds]; cte = comb.loc[test_ds]
    gl_pick = "hurdle"
    gl_v = te["loss_hurdle"].values
    sids = te.index.get_level_values("series_id").unique().to_numpy()
    pos = {s: np.where(te.index.get_level_values("series_id") == s)[0]
           for s in sids}
    d50 = cte[0.5].values
    diffs = []
    for _ in range(B):
        pick = rng.choice(sids, size=len(sids), replace=True)
        ix = np.concatenate([pos[s] for s in pick])
        diffs.append(100 * (gl_v[ix].mean() - d50[ix].mean()) / gl_v[ix].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    boot.append(dict(dataset=test_ds, comparison="50:50 vs global", mean=np.mean(diffs),
                     ci_low=lo, ci_high=hi, excludes_zero=bool(lo > 0 or hi < 0)))
bt = pd.DataFrame(boot)
print(bt.to_string(index=False, float_format=lambda x: "%.4f" % x))
bt.to_csv(os.path.join(OUT, "policy_bootstrap.csv"), index=False)
