import csv, os, re

R = r"E:\CODING\proj\covariate-trust-pilot\results"
MO = os.path.join(R, "paper_manuscript_verified")
SV = "results/synthetic_source_verification"
RD = "results/paper_rendering_verified/drafts"
PS = "results/paper_synthesis_verified"
PO = "results/paper_outline_verified"
S1 = SV + "/stage1_verified_contrasts.csv"
S1C = SV + "/stage1_verified_cells.csv"
S2F = SV + "/stage2_verified_factor_effects.csv"
S2C = SV + "/stage2_verified_cells.csv"
T4 = RD + "/table4_draft.md"
BOOT = "2000-draw paired series bootstrap"

NUM = [
 ("4.6", "+7.83 pp [+6.09,+9.45]", "interval dependence contrast", S1, "effect_pp/ci, M1_HURDLE_MEAN", "per series then bootstrap", BOOT),
 ("4.6", "-4.58 pp [-6.28,-2.92]", "magnitude dependence contrast", S1, "effect_pp/ci", "same", BOOT),
 ("4.6", "-6.26 pp [-8.01,-4.57]", "sparsity contrast", S1, "effect_pp/ci", "same", BOOT),
 ("4.6", "+3.35 pp [+1.62,+5.09]", "sparsity x interval", S1, "effect_pp/ci", "same", BOOT),
 ("4.6", "-0.01 pp [-1.70,+1.69]", "sparsity x magnitude, spans zero", S1, "effect_pp/ci", "same", BOOT),
 ("4.6", "-16.74 pp [-18.45,-15.05]", "interval x magnitude", S1, "effect_pp/ci", "same", BOOT),
 ("4.6", "-1.96 pp [-3.64,-0.31]", "three-way", S1, "effect_pp/ci", "same", BOOT),
 ("4.6", "G = -3.01 [-6.79,+0.23]", "C08, most direct-favourable Stage 1 cell", S1C, "rmse_mean_truth M0 vs M1", "G=100(1-H/P)", PS + "/claim_ledger_frozen.md sec 2"),
 ("4.6", "0.0035 / 40 of 40 / [-0.2036,-0.0005]", "control-arm integrity checks", SV + "/stage1_validity.json", "long-motif, distinct seqs, Markov gain", "as recorded", "n/a"),
 ("4.7", "0.9652 / 0.2900 / 0.9030", "C03 M1 error and the two hybrid columns", S1C, "rmse_mean_truth, p_true_x_mu_hat, p_hat_x_mu_true", "per cell", "none quoted"),
 ("4.8", "+0.1904 [+0.1699,+0.2119]", "abs_rho_I coefficient", S2F, "estimate/ci, HURDLE_MEAN", "factor model over 18 cells", BOOT),
 ("4.8", "+0.0667 [+0.0533,+0.0807]", "rho_I coefficient", S2F, "estimate/ci", "same", BOOT),
 ("4.8", "-0.0711 [-0.0851,-0.0570]", "rho_M coefficient", S2F, "estimate/ci", "same", BOOT),
 ("4.8", "-0.0228 [-0.0441,-0.0020]", "abs_rho_M coefficient", S2F, "estimate/ci", "same", BOOT),
 ("4.8", "-0.0239 [-0.0322,-0.0153]", "d coefficient", S2F, "estimate/ci", "same", BOOT),
 ("4.8", "+0.0332 [+0.0194,+0.0471]", "d x rho_I", S2F, "estimate/ci", "same", BOOT),
 ("4.8", "-0.0124 [-0.0262,+0.0012]", "d x rho_M, spans zero", S2F, "estimate/ci", "same", BOOT),
 ("4.8", "-0.0886 [-0.1115,-0.0654]", "rho_I x rho_M", S2F, "estimate/ci", "same", BOOT),
 ("4.8", "+12.10 / +3.57 / +16.47 pp", "Figure 2B marginal means over rho_I", S2C, "gain*100", "mean of 6 cells per level", "NONE - spread only, explicitly not a CI"),
 ("4.8", "+14.10 / +11.94 / +6.11 pp", "Figure 2C marginal means over rho_M", S2C, "gain*100", "mean of 6 cells per level", "NONE - spread only, explicitly not a CI"),
 ("4.8", "G = -19.76 [-26.00,-14.53]", "d=8, rho_I=0, rho_M=+0.8", S2C, "gain, gain_ci_low/high", "per cell", BOOT),
 ("4.8", "G = +2.36 [-1.20,+5.73]", "d=8, rho_I=0, rho_M=0, spans zero", S2C, "gain, ci", "per cell", BOOT),
 ("4.8", "-0.0650 [-0.0743,-0.0562]", "no-signal control Markov gain", SV + "/stage2_scientific_classification.json", "G1_NO_SIGNAL", "as recorded", "as recorded"),
 ("5.4", "3.152 / 3.260 / 3.411 / 3.483 / 4.202 / 4.220", "M5 classical mean ranks", PO + "/detailed_outline.md sec 5.2", "classical_benchmark/benchmark.json", "mean rank", "none"),
 ("5.5", "+0.1064 [+0.0437,+0.1652]", "H1 M5", T4, "estimate/ci", "per series then bootstrap", BOOT),
 ("5.5", "+0.0789 [+0.0205,+0.1405]", "H1 Favorita", T4, "estimate/ci", "same", BOOT),
 ("5.5", "+0.1529 [+0.0519,+0.2613]", "H1 within the intermittent regime, M5", T4, "estimate/ci", "same", BOOT),
 ("5.6", "-0.0230 [-0.0294,-0.0163]", "H2 frozen selector, M5 675 vs 5018", T4, "estimate/ci", "independent population", BOOT),
 ("5.6", "+11.87 pp", "H2 direct-prediction win-rate difference", T4, "reading column", "win rate", "3 seeds reported separately"),
 ("5.7", "+0.0032 [-0.0033,+0.0094]", "H2 isolated mechanism, overlap-weighted, n=5693", T4, "estimate/ci", "overlap weighting", BOOT),
 ("5.7", "1.32 / 0.614", "unweighted |SMD| on log scale; best matching balance", PS + "/claim_ledger_frozen.md", "sec 4", "as recorded", "none"),
 ("5.7", "-0.0084 / -0.0908 [-0.1401,-0.0438]", "occurrence-head Brier skill, M5 / Favorita", PO + "/reviewer_risk_map.md", "R8 evidence", "as recorded", "as recorded"),
 ("5.8", "-0.0305 [-0.1418,+0.0912]", "H3 M5", T4, "estimate/ci", "ADI-median split", BOOT),
 ("5.8", "-0.0428 [-0.1587,+0.0704]", "H3 Favorita", T4, "estimate/ci", "same", BOOT),
 ("5.8", "1.304 / 1.317", "ADI median split points, M5 / Favorita", PS + "/paper_readiness_verified.md", "H3 block", "as recorded", "none"),
 ("5.9", "4.11%", "convex oracle over the best static mixture, M5", PO + "/claim_to_evidence_map.md", "convex_oracle.json", "as recorded", "none quoted"),
 ("5.9", "2.15x", "expert-diversity ceiling multiplier", PO + "/claim_to_evidence_map.md", "expert_set_spec.json", "as recorded", "none quoted"),
 ("5.9", "-2.43% [-2.74,-2.13]", "frozen gate vs static mixture, first external dataset", PO + "/claim_to_evidence_map.md", "external_benchmark.json", "as recorded", "as recorded"),
 ("5.9", "+2.648% [+2.068,+3.287]", "sequence gate, FreshRetailNet-LT", PO + "/claim_to_evidence_map.md", "temporal_routing_encoder", "as recorded", "as recorded"),
 ("5.9", "-193.9%", "sequence gate, UCI Online Retail II", PO + "/claim_to_evidence_map.md", "temporal_routing_encoder", "as recorded", "none"),
 ("Abstract", "about 20%", "rounded from G = -19.76 at the direct-favourable cell", S2C, "gain", "rounding of a mapped value", "see the 4.8 row"),
 ("Abstract", "11.87 percentage points", "H2 win-rate difference", T4, "reading column", "see the 5.6 row", "see the 5.6 row"),
 ("Introduction", "about -19.8%", "rounded from G = -19.76, direct-favourable cell", S2C, "gain", "rounding of a mapped value", "see the 4.8 row"),
 ("Methods 4.1", "0.0015", "empirical mean-interval mode difference at d=4 (marginal control check)", SV + "/dgp_verification.md", "marginal_control_pass block", "as recorded", "n/a"),
 ("Methods", "0.95 / +-0.8", "bootstrap level; swept dependence levels", SV + "/metric_sign.md; " + SV + "/stage1_stage2_verified.md", "bootstrap level; rho grid", "setup constants", "n/a"),
 ("Methods", "5,856 / 5,857 / 0.017%", "parameter counts and the match margin", SV + "/point_hurdle_fairness.md", "parameters (recorded)", "setup constant", "n/a"),
 ("Methods", "96 / 24 / 576 / 384 / 480 / 80 / 18 / 2,000 / 30 / 256 / 1e-3", "protocol constants", SV + "/point_hurdle_fairness.md; " + SV + "/dgp_verification.md; " + SV + "/metric_sign.md", "geometry, replication, bootstrap", "setup constants", "n/a"),
 ("Methods", "1,200 / 1,941 / 1,829 / 1,688 / 1,576 / 300 / 20", "empirical protocol constants", RD + "/table3_draft.md", "Table 3", "setup constants", "n/a"),
]

p = os.path.join(MO, "manuscript_number_map.csv")
with open(p, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["section", "displayed_value", "quantity", "source_file",
                "source_field", "aggregation", "uncertainty_source"])
    w.writerows(NUM)
print("manuscript_number_map.csv:", len(NUM), "rows")

M = open(os.path.join(MO, "manuscript_v1.md"), encoding="utf-8").read()
keys = sorted(set(k.strip() for g in re.findall(r"\[([A-Za-z0-9;\s]+?)\]", M)
                  for k in g.split(";") if re.fullmatch(r"[A-Za-z]+\d{2}", k.strip())))
p2 = os.path.join(MO, "manuscript_reference_map.csv")
with open(p2, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["citation_key", "occurrences_in_manuscript", "peer_reviewed", "verified_source"])
    for k in keys:
        n = len(re.findall(r"\b%s\b" % k, M))
        w.writerow([k, n, "no - arXiv preprint, labelled in a footnote" if k == "MC26"
                    else "yes - Crossref verified",
                    "literature_boundary_verified/core_reference_list.md"])
print("manuscript_reference_map.csv:", len(keys), "keys", keys)

CL = [
 ("C1", "Controlled characterization of the finite-sample relative inductive bias of direct and factorized forecasting under temporal occurrence and magnitude dependence", "CONFIRMED", "Sec 4.6, 4.8; Table 2; Figure 2", PS + "/claim_ledger_frozen.md sec 1", "one backbone family; no scale axis; Stage 1 is alternation-only"),
 ("C2", "Empirical transfer with an explicit boundary: an analogue and a frozen selector transfer, while the isolated mechanism does not survive overlap adjustment", "SUPPORTED", "Sec 5.5, 5.6, 5.7; Table 4; Figure 3", PS + "/claim_ledger_frozen.md sec 4", "selector is not mechanism; no causal attribution to scale"),
 ("C3", "Adaptive-use boundary: a measurable oracle opportunity does not imply a learnable routing function", "SUPPORTED", "Sec 5.9; appendix", PS + "/claim_ledger_frozen.md sec 4", "reported as a boundary; stop rule triggered"),
 ("H1", "the occurrence-dependence relationship appears in observed demand", "SUPPORTED_WITH_BOUNDARY / EMPIRICAL_ANALOGUE", "Sec 5.5", T4, "never the word replicates"),
 ("H2a", "the frozen selector transfers predictively", "CONFIRMED", "Sec 5.6", T4, "independent population, three seeds"),
 ("H2b", "the isolated mechanism", "NOT_REPLICATED", "Sec 5.7", T4, "overlap-weighted interval crosses zero"),
 ("H3", "the sparsity interaction", "NOT_REPLICATED / CONSTRUCT_MISMATCH", "Sec 5.8", T4, "ADI-median split is not d=4 vs d=8"),
 ("MECH", "occurrence-head attribution", "COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT, synthetic only; not supported on real data", "Sec 4.7, 5.7", PS + "/claim_ledger_frozen.md sec 0 F5", "attribution, not causation"),
 ("ROUTE", "stable learned routing", "NOT_REPLICATED", "Sec 5.9", PS + "/paper_readiness_verified.md", "UCI -193.9% reported at full strength"),
]
p3 = os.path.join(MO, "manuscript_claim_ledger.csv")
with open(p3, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "claim", "status", "manuscript_location", "frozen_source", "wording_guard"])
    w.writerows(CL)
print("manuscript_claim_ledger.csv:", len(CL), "rows")
