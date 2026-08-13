import csv, os, re
PW=r"E:\CODING\proj\covariate-trust-pilot\results\paper_writing_verified"
PM=r"E:\CODING\proj\covariate-trust-pilot\results\paper_methods_verified"
t=open(os.path.join(PW,"methods_v1.md"),encoding="utf-8").read()
body=" ".join(t.split("---\n",1)[1].split())
F=[]
def chk(n,c,d=""):
    print("  %-46s %s %s"%(n,"PASS" if c else "FAIL",d))
    if not c: F.append(n)

# METH-P1 result leakage
leak_words=["we find","we show that","as expected","outperform","wins","winner",
            "better than the direct","better than the factorized","improves by","favours the factorized arm by"]
hits=[w for w in leak_words if w in body.lower()]
# numeric outcome scan: any G value or CI
# performance outcomes only: a % or pp figure NOT inside the parameter-match sentence
gvals=[m.group(0) for m in re.finditer(r"[−-]?\d+\.\d+\s*(?:pp|percentage points|%)", body)
       if not re.search(r"parameter|scalar|match rule", body[max(0,m.start()-140):m.end()+60], re.I)]
cis=re.findall(r"\[[−+-]?\d+\.\d+,\s*[−+-]?\d+\.\d+\]", body)
chk("METH-P1 result leakage = 0", not hits and not gvals and not cis,
    "words=%s gvals=%s cis=%s"%(hits,gvals,cis))

# METH-P2 setup constants source-linked
rows=list(csv.DictReader(open(os.path.join(PM,"methods_claim_source_map.csv"),encoding="utf-8")))
consts=["5,856","5,857","96","24","576","384","480","80","18","2,000","30","256","1e-3","0.017"]
present=[c for c in consts if c in body]
chk("METH-P2 setup constants present & mapped", len(present)>=12 and len(rows)>=30,
    "%d/%d constants, %d map rows"%(len(present),len(consts),len(rows)))

# METH-P3 notation registry consistency
reg={r["symbol"] for r in csv.DictReader(open(os.path.join(PM,"notation_registry.csv"),encoding="utf-8"))}
used=["y_t","o_t","y_t^+","h_t","p_t","mu_t","G","d"]
chk("METH-P3 notation matches registry", all(u in reg for u in used) and "z_t" not in body,
    "z_t absent=%s"%("z_t" not in body))
chk("METH-P3b horizon written H_f not bare H", "H_f" in body and " H " not in body.replace("H_f",""))
chk("METH-P4 Stage naming correct",
    "Stage 1" in body and "Stage 2" in body and "Stage 3" not in body and "Stage 4" not in body)
chk("METH-P5 fairness accurately stated",
    "5,856 parameters each" in body and "validation realized-`y`" in body
    and "prohibited by" in body and "5,857" in body)
chk("METH-P6 H1/H2/H3 operationalization",
    "absolute** first-order autocorrelation" in body.replace("**absolute**","absolute**")
    or "absolute" in body and "frozen before it is applied" in body
    and "different\nconstructs" in t or "different constructs" in body)
chk("METH-P7 no novelty overclaim",
    not any(w in body.lower() for w in ["first to","no prior work","novel","unexplored","state-of-the-art"]))
chk("METH-P8 finite-sample framing present", "finite-sample" in body.lower())
chk("METH-P9 control framed as control, not finding",
    "an experimental control, not a finding" in body)
chk("METH-P10 rho only in Stage 2 context",
    "only in Stage 2" in body and "ρ = −1" in body)
chk("METH-P11 UNIT-W1 / FLAG-W2 quantities absent",
    "C_neg" not in body and "C_pos" not in body and "C_sign" not in body)
print("\n  words:",len(body.split()))
print("\n  FAILS:",F if F else "none")
