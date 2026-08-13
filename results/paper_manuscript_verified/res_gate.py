import csv, os, re
PW=r"E:\CODING\proj\covariate-trust-pilot\results\paper_writing_verified"
SV=r"E:\CODING\proj\covariate-trust-pilot\results\synthetic_source_verification"
t=open(os.path.join(PW,"results_v1.md"),encoding="utf-8").read()
body=" ".join(t.split("---\n",1)[1].split())
F=[]
def chk(n,c,d=""):
    print("  %-48s %s %s"%(n,"PASS" if c else "FAIL",d)); F.append(n) if not c else None

# --- numeric cross-check against artifacts ---
c1={r["contrast"]:r for r in csv.DictReader(open(os.path.join(SV,"stage1_verified_contrasts.csv"),encoding="utf-8")) if r["model"]=="M1_HURDLE_MEAN"}
f2={r["term"]:r for r in csv.DictReader(open(os.path.join(SV,"stage2_verified_factor_effects.csv"),encoding="utf-8")) if r["model"]=="HURDLE_MEAN"}
s2=[r for r in csv.DictReader(open(os.path.join(SV,"stage2_verified_cells.csv"),encoding="utf-8")) if r["model"]=="HURDLE_MEAN"]
bad=[]
def near(txt,val,tol=0.006):
    return abs(float(txt)-val)<=tol
for k,shown in [("interval_dependence",7.83),("magnitude_dependence",-4.58),("sparsity",-6.26),
                ("sparsity_x_interval",3.35),("sparsity_x_magnitude",-0.01),
                ("interval_x_magnitude",-16.74),("three_way",-1.96)]:
    v=float(c1[k]["effect_pp"])
    if abs(v-shown)>0.006: bad.append((k,v,shown))
for k,shown in [("abs_rho_I",0.1904),("rho_I",0.0667),("rho_M",-0.0711),("abs_rho_M",-0.0228),
                ("d",-0.0239),("d_x_rho_I",0.0332),("d_x_rho_M",-0.0124),("rho_I_x_rho_M",-0.0886)]:
    v=float(f2[k]["estimate"])
    if abs(v-shown)>0.00006: bad.append((k,v,shown))
cell=[r for r in s2 if r["d"]=="8" and float(r["rho_interval"])==0.0 and float(r["rho_magnitude"])==0.8][0]
if abs(float(cell["gain"])*100+19.76)>0.01: bad.append(("point-fav cell",cell["gain"],-19.76))
null=[r for r in s2 if r["d"]=="8" and float(r["rho_interval"])==0.0 and float(r["rho_magnitude"])==0.0][0]
if abs(float(null["gain"])*100-2.36)>0.01: bad.append(("null cell",null["gain"],2.36))
nz=sum(1 for r in s2 if r["delta_ci_excludes_zero"]=="False")
chk("RES1 numbers match artifacts", not bad, str(bad))
chk("RES1b n cells / non-significant count", len(s2)==18 and nz==1, "18 cells, %d spanning zero"%nz)

# marginal means recomputed
def mean_g(key,val):
    xs=[float(r["gain"])*100 for r in s2 if abs(float(r[key])-val)<1e-9]
    return sum(xs)/len(xs), len(xs)
mb=[mean_g("rho_interval",v)[0] for v in (-0.8,0.0,0.8)]
mc=[mean_g("rho_magnitude",v)[0] for v in (-0.8,0.0,0.8)]
shownB=[12.10,3.57,16.47]; shownC=[14.10,11.94,6.11]
okB=all(abs(a-b)<0.02 for a,b in zip(mb,shownB)); okC=all(abs(a-b)<0.02 for a,b in zip(mc,shownC))
chk("RES1c Fig2B/2C marginals recomputed", okB and okC,
    "B=%s C=%s"%([round(x,2) for x in mb],[round(x,2) for x in mc]))

chk("RES2 Stage1 limitation visible",
    "no cell in which the direct arm wins with an interval clear of" in body
    and "CONDITIONALLY_VALID" in body and "deterministic period-2 alternation" in body)
chk("RES3 Stage2 scope visible", "Within the 18-cell Stage 2 grid" in body and "tested grid" in body)
chk("RES4 H1 analogue wording", "empirical analogue" in body and "not\na replication" in t.replace("not a replication","not\na replication"))
chk("RES5 H2 selector/mechanism split",
    "transfers as a **predictor**" in body and "NOT_REPLICATED" in body)
chk("RES6 H3 negative visible", "wrong sign" in body and "construct mismatch" in body.lower())
chk("RES7 routing failure visible", "−193.9%" in body and "stopping rule was triggered" in body
    and "reported at full strength" in body)
chk("RES8 no invented CI (Fig2B/C spread)", "is not a confidence interval" in body)
# assertion-based: a causal/absolute verb only counts when NOT inside a negation
NEG = r"not |never |cannot |does not |do not |neither |without "
c9 = []
for w in ["proves","proven","establishes the mechanism","causes the","universally","always better"]:
    for m in re.finditer(w, body, re.I):
        if re.search(NEG, body[max(0, m.start()-70):m.start()], re.I):
            continue
        c9.append((w, body[max(0,m.start()-50):m.end()+40]))
chk("RES9 no causal overclaim (asserted)", not c9 and
    "does not demonstrate that the occurrence process causes" in body, str(c9)[:90])
chk("RES10 UNIT-W1 / FLAG-W2 absent", not any(w in body for w in ["C_neg","C_pos","C_sign"]))
chk("RES11 no universal winner", "neither representation dominates" in body.lower())
chk("RES12 classical context present", "3.152" in body and "4.220" in body)
chk("RES13 occurrence-head real-data negative visible", "−0.0084" in body and "−0.0908" in body)
print("\n  words:",len(body.split()))
print("  FAILS:",F if F else "none")
