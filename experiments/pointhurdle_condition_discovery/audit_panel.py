"""Phase A/B: audit cached artifacts and build the series x outer_origin paired panel.
READ-ONLY on results/. Writes nothing here; the caller decides where output goes."""
import json, os
import numpy as np
import pandas as pd

R = r"E:\CODING\proj\covariate-trust-pilot\results\external_validity_screen"

SRC = {
    "m5": os.path.join(R, "rule_replication", "independent_raw_predictions.parquet"),
    "favorita": os.path.join(R, "favorita_independent", "independent_raw_predictions.parquet"),
}
POP = {
    "m5": os.path.join(R, "rule_replication", "independent_population.csv"),
    "favorita": os.path.join(R, "favorita_independent", "independent_population.csv"),
}

print("=" * 72)
print("A2  raw prediction artifacts")
print("=" * 72)
panels = []
for ds, p in SRC.items():
    df = pd.read_parquet(p)
    origins = sorted(df["origin"].unique())
    steps = sorted(df["step"].unique())
    print("\n%s  %s" % (ds, os.path.basename(p)))
    print("   rows           %d" % len(df))
    print("   series         %d" % df["series_id"].nunique())
    print("   origins        %s" % origins)
    print("   steps          %d..%d (%d)" % (steps[0], steps[-1], len(steps)))
    print("   groups         %s" % dict(df["group"].value_counts()))
    print("   target_mask    %s" % dict(df["target_mask"].value_counts()))
    print("   NaN point/hurdle  %d / %d"
          % (df["point_mean_prediction"].isna().sum(),
             df["hurdle_mean_prediction"].isna().sum()))

    m = df[df["target_mask"]].copy()
    m["se_point"] = (m["y_observed"] - m["point_mean_prediction"]) ** 2
    m["se_hurdle"] = (m["y_observed"] - m["hurdle_mean_prediction"]) ** 2
    g = m.groupby(["series_id", "group", "origin"], as_index=False).agg(
        n_steps=("se_point", "size"),
        mse_point=("se_point", "mean"),
        mse_hurdle=("se_hurdle", "mean"),
        n_positive_test=("occurrence", "sum"),
        mean_y=("y_observed", "mean"),
    )
    g["loss_point"] = np.sqrt(g["mse_point"])
    g["loss_hurdle"] = np.sqrt(g["mse_hurdle"])
    g["dataset_id"] = ds
    g = g.rename(columns={"origin": "outer_origin_id"})
    panels.append(g)

panel = pd.concat(panels, ignore_index=True)
panel["paired_valid"] = (panel["loss_point"].notna() & panel["loss_hurdle"].notna()
                         & (panel["n_steps"] > 0))
panel["delta_rmse"] = panel["loss_point"] - panel["loss_hurdle"]
panel["G"] = 100.0 * (1.0 - panel["loss_hurdle"] / panel["loss_point"])

print("\n" + "=" * 72)
print("B1  paired panel")
print("=" * 72)
print("rows                     %d" % len(panel))
print("paired_valid             %d (%.2f%%)"
      % (panel["paired_valid"].sum(), 100 * panel["paired_valid"].mean()))
print("unique series            %d" % panel.groupby(["dataset_id", "series_id"]).ngroups)
print("rows per series          %s"
      % dict(panel.groupby(["dataset_id", "series_id"]).size().value_counts()))
print("loss_point == 0          %d" % (panel["loss_point"] == 0).sum())
print("loss_hurdle == 0         %d" % (panel["loss_hurdle"] == 0).sum())
print("\nby dataset x origin:")
print(panel.groupby(["dataset_id", "outer_origin_id"])
      .agg(rows=("series_id", "size"), series=("series_id", "nunique"),
           mean_G=("G", "mean")).to_string())

print("\n" + "=" * 72)
print("A2  population / natural-distribution check")
print("=" * 72)
for ds, p in POP.items():
    if os.path.exists(p):
        pop = pd.read_csv(p)
        print("\n%s  %s   shape %s" % (ds, os.path.basename(p), pop.shape))
        print("   cols: %s" % list(pop.columns)[:14])
        for c in ("group", "regime", "sbc_regime"):
            if c in pop.columns:
                print("   %-8s %s" % (c, dict(pop[c].value_counts())))

panel.to_parquet(os.path.join(os.path.dirname(__file__), "paired_panel_draft.parquet"),
                 index=False)
print("\ndraft panel written to scratchpad")
