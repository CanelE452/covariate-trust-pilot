"""Gate 1 pilot: M5 <-> Favorita transfer, two tracks.

Track 1  STRICT TRAIN-ONLY   features from train-split descriptors only; target = G at all
                             three outer origins.
Track 2  ORIGIN-1-CONDITIONED  adds S_occ, S_mag and inner gain computed from the FIRST
                             outer origin; target = G at origins 2 and 3 only.
                             Causally valid (origin 1 is observed before the origin-2
                             decision) but NOT train-only.  Labelled everywhere.

No model is retrained.  Everything is read from cached artifacts.
"""
import json, os, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
EVS = r"E:\CODING\proj\covariate-trust-pilot\results\external_validity_screen"
OUT = r"E:\CODING\proj\covariate-trust-pilot\results\pointhurdle_condition_discovery"
SEED = 20260813
EPS_Q = 0.01

panel = pd.read_parquet(os.path.join(OUT, "paired_panel.parquet"))
POP = {"m5": os.path.join(EVS, "rule_replication", "independent_population.csv"),
       "favorita": os.path.join(EVS, "favorita_independent", "independent_population.csv")}
RAW = {"m5": os.path.join(EVS, "rule_replication", "independent_raw_predictions.parquet"),
       "favorita": os.path.join(EVS, "favorita_independent", "independent_raw_predictions.parquet")}

# ---------------------------------------------------------------- train-only features
pops = []
for ds, p in POP.items():
    d = pd.read_csv(p)
    d["dataset_id"] = ds
    pops.append(d)
pop = pd.concat(pops, ignore_index=True)
pop["log_n_positive_train"] = np.log1p(pop["n_positive_train"])
pop["D_gap"] = pop["rho_interval_abs_train"]
pop["mean_positive"] = pop["train_scale"]
STRUCT = ["zero_ratio_train", "ADI_train", "CV2_positive_train", "log_n_positive_train",
          "rho_magnitude_train", "occurrence_binary_acf1_train", "mean_positive"]

# ------------------------------------------------- origin-1 component forecastability
def origin1_features(ds):
    df = pd.read_parquet(RAW[ds])
    o1 = int(sorted(df["origin"].unique())[0])
    d = df[(df["origin"] == o1) & df["target_mask"]].copy()
    out = []
    for sid, g in d.groupby("series_id"):
        z = g["occurrence"].to_numpy(float)
        p = g["hurdle_p_prediction"].to_numpy(float)
        y = g["y_observed"].to_numpy(float)
        mu = g["hurdle_mu_prediction"].to_numpy(float)
        zb = z.mean()
        den_o = ((z - zb) ** 2).sum()
        s_occ = 1 - ((z - p) ** 2).sum() / den_o if den_o > 1e-12 else np.nan
        pos = y > 0
        npos = int(pos.sum())
        if npos >= 10:
            yb = y[pos].mean()
            den_m = ((y[pos] - yb) ** 2).sum()
            s_mag = 1 - ((y[pos] - mu[pos]) ** 2).sum() / den_m if den_m > 1e-12 else np.nan
        else:
            s_mag = np.nan
        lp = np.sqrt(((y - g["point_mean_prediction"].to_numpy(float)) ** 2).mean())
        lh = np.sqrt(((y - g["hurdle_mean_prediction"].to_numpy(float)) ** 2).mean())
        out.append(dict(dataset_id=ds, series_id=sid, S_occ=s_occ, S_mag=s_mag,
                        s_mag_missing=int(np.isnan(s_mag)), n_pos_o1=npos,
                        inner_loss_point=lp, inner_loss_hurdle=lh))
    return pd.DataFrame(out), o1


o1f, O1 = [], {}
for ds in ("m5", "favorita"):
    f, o = origin1_features(ds)
    o1f.append(f)
    O1[ds] = o
o1 = pd.concat(o1f, ignore_index=True)

# epsilon from the pooled positive loss distribution (both tracks use the same rule)
eps = float(np.quantile(panel["loss_point"][panel["loss_point"] > 0], EPS_Q))
o1["inner_gain_log"] = np.log((o1["inner_loss_point"] + eps) / (o1["inner_loss_hurdle"] + eps))
panel["gain_log"] = np.log((panel["loss_point"] + eps) / (panel["loss_hurdle"] + eps))
print("epsilon (1st percentile of positive loss_point) = %.6f" % eps)
print("S_occ  non-null %d / %d ;  S_mag non-null %d / %d"
      % (o1["S_occ"].notna().sum(), len(o1), o1["S_mag"].notna().sum(), len(o1)))

base = panel.merge(pop, on=["dataset_id", "series_id"], how="left") \
            .merge(o1, on=["dataset_id", "series_id"], how="left")
base["is_origin1"] = base.groupby("dataset_id")["outer_origin_id"].transform("min") \
    == base["outer_origin_id"]

TRACKS = {
    "T1_strict_train_only": dict(df=base, target="gain_log",
                                 sets={"A0": [], "A1": ["ADI_train", "CV2_positive_train"],
                                       "A2": ["D_gap"], "A3": STRUCT + ["D_gap"]}),
    "T2_origin1_conditioned": dict(df=base[~base["is_origin1"]].copy(), target="gain_log",
                                   sets={"A0": [], "A2": ["D_gap"],
                                         "A4": ["S_occ", "S_mag", "zero_ratio_train",
                                                "log_n_positive_train"],
                                         "A5": ["inner_gain_log"],
                                         "A6": ["S_occ", "S_mag"] + STRUCT + ["D_gap"],
                                         "A7": ["S_occ", "S_mag", "inner_gain_log",
                                                "zero_ratio_train", "log_n_positive_train"]}),
}


def add_inter(df, cols):
    df = df.copy()
    # Phase D3: missingness is informative and must be an explicit feature, never
    # smuggled in through the imputer.
    for c in list(cols):
        df[c + "_miss"] = df[c].isna().astype(float)   # always created, so train and
                                                       # test share an identical schema
    if {"S_occ", "S_mag"} <= set(cols):
        df["S_occ_x_S_mag"] = df["S_occ"] * df["S_mag"]
    if {"S_occ", "zero_ratio_train"} <= set(cols):
        df["S_occ_x_zero"] = df["S_occ"] * df["zero_ratio_train"]
    if {"S_mag", "zero_ratio_train"} <= set(cols):
        df["S_mag_x_zero"] = df["S_mag"] * df["zero_ratio_train"]
    extra = [c for c in ("S_occ_x_S_mag", "S_occ_x_zero", "S_mag_x_zero") if c in df.columns]
    miss = [c + "_miss" for c in cols]
    return df, cols + extra + miss


def fit_eval(tr, te, cols, target, rng, shuffle=False, randfeat=False):
    if randfeat:
        cols = ["_rand1", "_rand2"]
        tr, te = tr.copy(), te.copy()
        for c in cols:
            tr[c] = rng.normal(size=len(tr)); te[c] = rng.normal(size=len(te))
    if not cols:                                     # A0: training-domain mean
        pred = np.full(len(te), tr[target].mean())
        return pred, None
    ytr = tr[target].to_numpy()
    if shuffle:
        ytr = rng.permutation(ytr)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("en", ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], cv=5,
                                         random_state=SEED, max_iter=5000))])
    pipe.fit(tr[cols], ytr)
    return pipe.predict(te[cols]), pipe


def policy(te, pred, tau_sel, global_choice):
    lp, lh = te["loss_point"].to_numpy(), te["loss_hurdle"].to_numpy()
    g_hat_pct = 100 * (1 - np.exp(-pred))            # gain_log -> approx G%
    pick_h = g_hat_pct > tau_sel
    pick_p = g_hat_pct < -tau_sel
    base_loss = lh if global_choice == "hurdle" else lp
    sel = np.where(pick_h, lh, np.where(pick_p, lp, base_loss))
    return sel.mean(), base_loss.mean(), np.minimum(lp, lh).mean()


rng = np.random.default_rng(SEED)
rows = []
for tname, T in TRACKS.items():
    df = T["df"]
    for train_ds, test_ds in (("m5", "favorita"), ("favorita", "m5")):
        tr_all = df[df["dataset_id"] == train_ds]
        te_all = df[df["dataset_id"] == test_ds]
        gchoice = "hurdle" if tr_all["loss_hurdle"].mean() < tr_all["loss_point"].mean() else "point"
        for aid, cols in T["sets"].items():
            tr, c = add_inter(tr_all, cols)
            te, _ = add_inter(te_all, cols)
            for ctl in ([""] if aid != "A0" else [""]) + ([] if aid == "A0" else ["C0", "C1"]):
                pred, model = fit_eval(tr, te, c, T["target"], rng,
                                       shuffle=(ctl == "C0"), randfeat=(ctl == "C1"))
                # degeneracy guard: a shrunk-to-constant ElasticNet leaves ~1e-16 float
                # jitter, and spearman would rank that dust.  Treat as no prediction.
                spread = float(np.std(pred))
                degenerate = spread < 1e-8 * max(1.0, float(np.std(te[T["target"]])))
                rho = np.nan if degenerate else spearmanr(pred, te[T["target"]]).statistic
                sl, bl, orc = policy(te, pred, 2.0, gchoice)
                rows.append(dict(track=tname, train=train_ds, test=test_ds,
                                 ablation=aid + ("_" + ctl if ctl else ""),
                                 n_test=len(te), spearman=rho,
                                 pred_sd=spread, degenerate=bool(degenerate),
                                 selector_loss=sl, global_loss=bl, oracle_loss=orc,
                                 rel_improve_pct=100 * (bl - sl) / bl,
                                 oracle_gap_capture_pct=(100 * (bl - sl) / (bl - orc)
                                                         if bl - orc > 1e-9 else np.nan)))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "pilot_ablation.csv"), index=False)
pd.set_option("display.width", 200)
for t in TRACKS:
    print("\n" + "=" * 90)
    print(t)
    print("=" * 90)
    sub = res[res["track"] == t]
    print(sub[["train", "test", "ablation", "n_test", "spearman", "rel_improve_pct",
               "oracle_gap_capture_pct"]]
          .to_string(index=False, float_format=lambda x: "%.4f" % x))

macro = (res.groupby(["track", "ablation"])
         .agg(macro_spearman=("spearman", "mean"),
              macro_rel_improve=("rel_improve_pct", "mean"),
              worst_rel_improve=("rel_improve_pct", "min"),
              both_dirs_positive=("spearman", lambda s: bool((s > 0).all())))
         .reset_index())
macro.to_csv(os.path.join(OUT, "pilot_fold_metrics.csv"), index=False)
print("\n" + "=" * 90)
print("MACRO over the two transfer directions")
print("=" * 90)
print(macro.to_string(index=False, float_format=lambda x: "%.4f" % x))
print("\nseed=%d  origins used as 'inner' in T2: %s" % (SEED, O1))
