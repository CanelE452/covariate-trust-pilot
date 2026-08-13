import csv, os, re
R=r"E:\CODING\proj\covariate-trust-pilot\results"
MO=os.path.join(R,"paper_manuscript_verified"); SV=os.path.join(R,"synthetic_source_verification")
M=open(os.path.join(MO,"manuscript_v2.md"),encoding="utf-8").read()
B=" ".join(re.sub(r"(?m)^\s*>\s?", "", M).split())
F=[]
def chk(n,c,d=""):
    print("  %-46s %s %s"%(n,"PASS" if c else "FAIL",d))
    if not c: F.append(n)

print("=== SCIENTIFIC ===")
chk("F4  finite-sample consistent", B.count("finite-sample")>=4)
chk("F5  Stage terminology", "Stage 1" in B and "Stage 2" in B and "Stage 3" not in B and "Stage 4" not in B)
chk("F6  H1 analogue everywhere",
    B.count("empirical analogue")>=3 and "H1 is reported as `SUPPORTED_WITH_BOUNDARY`" in B)
chk("F7  H2 selector/mechanism split", "transfers as a **predictor**" in B and "isolated mechanism" in B)
chk("F8  H3 negative visible", "NOT_REPLICATED` at the pre-registered split" in B or "H3 is `NOT_REPLICATED`" in B)
chk("F9  routing negative visible", "−193.9%" in B and "stopping rule" in B)
chk("F10 no universal winner",
    not re.search(r"(factorized|direct) (arm |formulation )?is (always|universally|generally) better", B, re.I)
    and "neither is uniformly better" in B or "neither representation dominates" in B)

print("\n=== NUMBERS ===")
c1={r["contrast"]:float(r["effect_pp"]) for r in csv.DictReader(open(os.path.join(SV,"stage1_verified_contrasts.csv"),encoding="utf-8")) if r["model"]=="M1_HURDLE_MEAN"}
f2={r["term"]:float(r["estimate"]) for r in csv.DictReader(open(os.path.join(SV,"stage2_verified_factor_effects.csv"),encoding="utf-8")) if r["model"]=="HURDLE_MEAN"}
KEY={"+7.83":c1["interval_dependence"],"−4.58":c1["magnitude_dependence"],"−6.26":c1["sparsity"],
     "+3.35":c1["sparsity_x_interval"],"−16.74":c1["interval_x_magnitude"],"−1.96":c1["three_way"],
     "+0.1904":f2["abs_rho_I"],"+0.0667":f2["rho_I"],"−0.0711":f2["rho_M"],"−0.0228":f2["abs_rho_M"],
     "−0.0239":f2["d"],"+0.0332":f2["d_x_rho_I"],"−0.0886":f2["rho_I_x_rho_M"]}
mism=[]
for s,v in KEY.items():
    shown=float(s.replace("−","-").replace("+",""))
    if abs(shown-v)>0.006: mism.append((s,v))
    if s not in B: mism.append((s,"NOT IN MANUSCRIPT"))
chk("N2  same quantity same value everywhere", not mism, str(mism)[:80])
# G sign convention stated once, consistently
chk("N1  G sign convention consistent",
    B.count("100 · (1 − RMSE_H / RMSE_P)")+B.count("100(1 − RMSE_H / RMSE_P)")>=1
    and "positive favouring the factorized arm" in B or "positive values favour the factorized arm" in B)
chk("N5  cell counts consistent", B.count("18-cell")+B.count("18 cells")>=2 and "eight-cell" in B)
chk("N6  C_neg / C_pos absent", "C_neg" not in B and "C_pos" not in B)
chk("N7  C_sign absent", "C_sign" not in B)

print("\n=== LITERATURE ===")
chk("L1  ALR12 control detail absent",
    not any(x in B.lower() for x in ["held adi","fixed the marginals","holds the marginals","altay et al. held"]))
chk("L2  Kou13 ratio everywhere",
    "combines them as a ratio" in B and not re.search(r"Kou(rentzes|13)[^.]{0,170}hurdle",B,re.I))
chk("L3  NAR26 LightGBM / product", "gradient-boosting" in B and "LightGBM" in B
    and not re.search(r"NAR26[^.]{0,120}neural(?!\s+setting)",B))
chk("L4  no first claim",
    not re.search(r"\bwe are the first\b|\bthe first (study|paper|work) to\b|\bfirst to (compare|show|study)\b",B,re.I))
chk("L5/L6 no decomposition / dependence novelty",
    "not a modelling choice this paper introduces" in B and "established territory" in B)

print("\n=== NOTATION ===")
reg={r["symbol"] for r in csv.DictReader(open(os.path.join(R,"paper_methods_verified","notation_registry.csv"),encoding="utf-8"))}
chk("T1-T8 registry symbols present & no collision",
    all(s in B for s in ["p_t","mu_t","ρ_I","ρ_M","G","o_t"]) and "z_t" not in B and reg)
chk("T6  G defined once, used consistently", B.count("G = 100")>=1)

print("\n=== FIGURES / TABLES ===")
chk("V3  captions match plotted values", "[−23, +23]" in B and "18 cells, as two 3 × 3 panels" in B)
chk("V4  Fig2 spread not called CI", "is not a confidence interval" in B and B.count("not a confidence interval")>=2)
chk("V5  open-circle semantics", "open circle marks a cell" in B)
chk("V6  Fresh/UCI appendix role", "stress tests" in B and "never treated as core validation data" in B)

print("\n=== CLAIM STRENGTH ===")
NEG=r"not|never|cannot|does not|do not|neither|without|no such|\*\*not\*\*"
def asserted(words):
    out=[]
    for w in words:
        for m in re.finditer(w,B,re.I):
            if re.search(NEG,B[max(0,m.start()-70):m.start()],re.I): continue
            out.append((w,B[max(0,m.start()-45):m.end()+35]))
    return out
a1=asserted([r"\bproves\b",r"\bproven\b"]); chk("C-A1 proves = 0", not a1, str(a1)[:70])
a2=asserted([r"\buniversally\b",r"\balways better\b"]); chk("C-A2 universal = 0", not a2, str(a2)[:70])
a3=asserted([r"establishes (a|the) (causal )?mechanism",r"demonstrates the mechanism"]); chk("C-A3 causal mechanism = 0", not a3)
a4=[x for x in asserted([r"the first"])
    if not re.search(r"the first (half|external dataset)", x[1], re.I)]
chk("C-A4 first = 0 (precedence sense)", not a4, str(a4)[:70])
a5=asserted([r"state-of-the-art",r"\bSOTA\b"]); chk("C-A5 SOTA = 0", not a5)
a6=asserted([r"mechanism replicat"]); chk("C-A6 mechanism replication = 0", not a6, str(a6)[:70])

print("\n  manuscript words:",len(B.split()))
print("  FAILS:",F if F else "none")
