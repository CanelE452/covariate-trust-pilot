"""Write Gate 0 / Gate 1 artifacts and the audit tables. No model is fitted here."""
import json, os
import numpy as np
import pandas as pd

OUT = r"E:\CODING\proj\covariate-trust-pilot\results\pointhurdle_condition_discovery"
EVS = r"E:\CODING\proj\covariate-trust-pilot\results\external_validity_screen"
os.makedirs(OUT, exist_ok=True)

panel = pd.read_parquet(os.path.join(OUT, "paired_panel.parquet"))
abl = pd.read_csv(os.path.join(OUT, "pilot_ablation.csv"))

# ---- eligibility / sample summary -------------------------------------------------
es = (panel.groupby("dataset_id")
      .agg(rows=("series_id", "size"), series=("series_id", "nunique"),
           origins=("outer_origin_id", "nunique"),
           paired_valid_pct=("paired_valid", lambda s: 100 * s.mean()),
           mean_loss_point=("loss_point", "mean"),
           mean_loss_hurdle=("loss_hurdle", "mean"))
      .reset_index())
es.to_csv(os.path.join(OUT, "eligibility_summary.csv"), index=False)

# ---- candidate region sample sizes (Phase H3 screening only) ----------------------
pop = []
for ds, f in (("m5", "rule_replication"), ("favorita", "favorita_independent")):
    d = pd.read_csv(os.path.join(EVS, f, "independent_population.csv"))
    d["dataset_id"] = ds
    pop.append(d)
pop = pd.concat(pop, ignore_index=True)
m = panel.merge(pop, on=["dataset_id", "series_id"], how="left")
TAU = 2.0
m["winner"] = np.where(m["G"] > TAU, "hurdle", np.where(m["G"] < -TAU, "point", "neutral"))
reg = []
for ds, d in m.groupby("dataset_id"):
    zmed = d["zero_ratio_train"].median()
    for lab, sel in (("high_sparsity", d["zero_ratio_train"] >= zmed),
                     ("low_sparsity", d["zero_ratio_train"] < zmed)):
        s = d[sel]
        reg.append(dict(dataset=ds, region=lab, rows=len(s), series=s["series_id"].nunique(),
                        mean_G=s["G"].mean(),
                        point_pct=100 * (s["winner"] == "point").mean(),
                        hurdle_pct=100 * (s["winner"] == "hurdle").mean()))
rg = pd.DataFrame(reg)
rg.to_csv(os.path.join(OUT, "pilot_candidate_regions.csv"), index=False)

# ---- policy metrics table ---------------------------------------------------------
pol = abl[~abl["ablation"].str.contains("_C")][
    ["track", "train", "test", "ablation", "n_test", "spearman", "pred_sd",
     "selector_loss", "global_loss", "oracle_loss", "rel_improve_pct",
     "oracle_gap_capture_pct"]]
pol.to_csv(os.path.join(OUT, "pilot_policy_metrics.csv"), index=False)

# ---- artifact inventory -----------------------------------------------------------
inv = [
 dict(dataset="m5", artifact="rule_replication/independent_raw_predictions.parquet",
      paired_series=5693, origins=3, horizon=28, primary_metric="RMSE on realized y",
      raw_predictions="yes", head_predictions="yes (p_hat, mu_hat)",
      train_only_features="yes (independent_population.csv)",
      natural_sample="independent eligible population, not regime-balanced"),
 dict(dataset="favorita", artifact="favorita_independent/independent_raw_predictions.parquet",
      paired_series=5405, origins=3, horizon=28, primary_metric="RMSE on realized y",
      raw_predictions="yes", head_predictions="yes (p_hat, mu_hat)",
      train_only_features="yes (independent_population.csv)",
      natural_sample="independent eligible population, not regime-balanced"),
 dict(dataset="m5+favorita (Stage A)", artifact="per_series_metrics.csv",
      paired_series=2400, origins=0, horizon=28, primary_metric="RMSE on realized y",
      raw_predictions="no", head_predictions="brier / magnitude rmse only, aggregated",
      train_only_features="yes", natural_sample="NO - SBC regime-balanced, 300 per regime"),
 dict(dataset="favorita (transfer)", artifact="favorita_transfer/transfer_raw_predictions.parquet",
      paired_series=135, origins=3, horizon=28, primary_metric="RMSE on realized y",
      raw_predictions="yes", head_predictions="yes", train_only_features="yes",
      natural_sample="candidate/control subset only"),
 dict(dataset="FreshRetailNet-LT", artifact="results/multi_benchmark (routing only)",
      paired_series=0, origins=0, horizon=0, primary_metric="routing gain, not paired RMSE",
      raw_predictions="not in this schema", head_predictions="no",
      train_only_features="no", natural_sample="n/a - stress test"),
 dict(dataset="UCI Online Retail II", artifact="results/temporal_routing_encoder (routing only)",
      paired_series=0, origins=0, horizon=0, primary_metric="routing gain, not paired RMSE",
      raw_predictions="not in this schema", head_predictions="no",
      train_only_features="no", natural_sample="n/a - stress test"),
]
pd.DataFrame(inv).to_csv(os.path.join(OUT, "artifact_inventory.csv"), index=False)

# ---- protocol comparison (A3) -----------------------------------------------------
proto = [
 ("target", "realized y on the test window", "realized y on the test window", "match"),
 ("input variables", "same lookback window", "same lookback window", "match"),
 ("lookback", "96", "96", "match"),
 ("horizon", "28", "28", "match"),
 ("split m5", "train_end 1829 / val_end 1857", "train_end 1829 / val_end 1857", "match"),
 ("split favorita", "train_end 1576 / val_end 1604", "train_end 1576 / val_end 1604", "match"),
 ("backbone", "DLinear (M0PM_point_mse_param_matched)", "DLinear (M1_factorized_mean)", "match"),
 ("training budget", "om_factorization_killtest.train.train_one, imported unchanged",
  "same trainer, imported unchanged", "match"),
 ("seed", "canonical_model_seed 0", "canonical_model_seed 0", "match"),
 ("primary loss", "MSE on y", "factorized objective (p, mu heads)", "DIFFERS BY DESIGN"),
 ("evaluation metric", "RMSE vs realized y", "RMSE vs realized y", "match"),
 ("test origins m5", "1857 / 1885 / 1913", "1857 / 1885 / 1913", "match"),
 ("test origins favorita", "1604 / 1632 / 1660", "1604 / 1632 / 1660", "match"),
]
pd.DataFrame(proto, columns=["item", "point", "hurdle", "agreement"]).to_csv(
    os.path.join(OUT, "protocol_comparison.csv"), index=False)

print("eligibility_summary")
print(es.to_string(index=False, float_format=lambda x: "%.4f" % x))
print("\ncandidate regions (sparsity split at within-dataset median zero_ratio_train)")
print(rg.to_string(index=False, float_format=lambda x: "%.2f" % x))
print("\nwrote:", sorted(os.listdir(OUT)))
