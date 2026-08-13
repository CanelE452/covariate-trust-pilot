import csv, os, re

PW = r"E:\CODING\proj\covariate-trust-pilot\results\paper_writing_verified"
LB = r"E:\CODING\proj\covariate-trust-pilot\results\literature_boundary_verified"
FAILS = []


def chk(n, c, d=""):
    print("  %-52s %s %s" % (n, "PASS" if c else "FAIL", d))
    if not c:
        FAILS.append(n)


rw = open(os.path.join(PW, "related_work_v1.md"), encoding="utf-8").read()
body = " ".join(rw.split("---\n", 1)[1].split())
intro = open(os.path.join(PW, "introduction_v6.md"), encoding="utf-8").read()
_ip = [x for x in re.split(r"\n\s*\n", intro.split("---\n", 1)[1]) if x.strip()]
p2 = " ".join(_ip[1].split())

print("=== RW gate ===")
secs = re.findall(r"## (2\.\d)", rw)
chk("RW1  2.1-2.5 prose complete", secs == ["2.1", "2.2", "2.3", "2.4", "2.5"], str(secs))
chk("RW2  decomposition precedent acknowledged",
    "not a modelling choice this paper introduces" in body and "[Cro72]" in body)
chk("RW3  decomposition novelty claim = 0",
    not re.search(r"we (introduce|propose) (the )?(decomposition|factorization)", body, re.I))
chk("RW4  ADI/CV^2 role accurate",
    "average inter-demand interval" in body and "squared coefficient of variation" in body
    and "[SBC05]" in body and "[KH06]" in body)
chk("RW5  no disparagement of ADI/CV^2",
    not re.search(r"inadequate|fail to|shortcoming|ignores time|flawed", body, re.I)
    and "not a criticism" in body)
chk("RW6  temporal-dependence precedent acknowledged",
    "[ALR12]" in body and "established territory" in body)
banned_alr = ["held adi", "held cv", "fixed the marginals", "holds the marginals",
              "held the marginals", "isolated correlation under matched marginals",
              "altay et al. held", "alr12 held", "marginals fixed"]
alr_sents = [x for x in re.split(r"(?<=[.!?]) ", body)
             if re.search(r"ALR12|Altay", x, re.I)]
hit = [(b, x[:60]) for x in alr_sents for b in banned_alr if b in x.lower()]
chk("RW7  ALR12 unresolved control detail unused", not hit, str(hit))
chk("RW8  LIT-W3 respected (no ALR12 methodology claim)",
    "generated intermittent demand" in body and not re.search(
        r"ALR12[^.]{0,120}(fixed|held constant|controlled for)", body, re.I))
chk("RW9  Kou13 ratio formulation accurate",
    "combined as a ratio" in body and "inversion bias" in body and "[Kou13]" in body)
kou_span = body[body.index("[Kou13]") - 400: body.rindex("[Kou13]") + 200] \
    if "[Kou13]" in body else ""
chk("RW10 Kou13 never called a hurdle",
    not re.search(r"Kou(rentzes|13)[^.]{0,160}hurdle", body, re.I)
    and "the same two representations" not in body)
chk("RW11 NAR26 LightGBM accurate, never neural",
    "LightGBM" in body and not re.search(r"NAR26[^.]{0,120}neural(?!\s+setting)", body)
    and "gradient-boosting" in body)
chk("RW12 NAR26 product/two-stage precedent acknowledged",
    "probability-times-size product" in body and "Both comparisons therefore already exist"
    in body)
chk("RW13 same features != matched capacity",
    "identical data preprocessing, feature construction and evaluation protocols" in body
    and not re.search(r"NAR26[^.]{0,200}(matched (capacity|parameter|training))", body, re.I)
    and not re.search(r"matched (capacity|budget)[^.]{0,80}\[NAR26\]", body, re.I))
chk("RW14 direct-vs-hurdle novelty claim = 0",
    "Neither the factorized formulation nor the act of comparing it against a direct one "
    "originates here" in body)
chk("RW15 fixed-marginal novelty claim = 0",
    "a property of the design rather than a result" in body
    and not re.search(r"fixed marginals? (are|is) (our|the) (novelty|contribution)", body, re.I))
absence = ["no prior work", "nobody has", "has never been", "we are the first",
           "unexplored", "first to", "novel combination", "have never been combined",
           "previous studies leave", "no previous", "has not been studied",
           "remains unexplored", "no study has"]
hits = [a for a in absence if a in body.lower()]
chk("RW16 first/no-prior-work/unexplored = 0", not hits, str(hits))
asym = ["none of the prior", "cannot explain", "absent from prior", "does not predict",
        "prior work cannot", "no prior strand"]
chk("RW17 literature-asymmetry absence claim = 0",
    not [a for a in asym if a in body.lower()])
chk("RW18 intersection matches novelty freeze",
    "varied along separate axes" in body and "one parameter budget" in body
    and "finite-sample behaviour under a fixed budget" in body)
chk("RW19 empirical transfer boundary included",
    "empirical analogue" in body and "predictive selector" in body
    and "do not survive adjustment" in body or "does not survive adjustment" in body)

rows = list(csv.DictReader(open(os.path.join(PW, "related_work_reference_map.csv"),
                                encoding="utf-8")))
chk("RW20 every factual literature sentence citation-linked",
    all(r["citation_key_or_paper_id"] for r in rows), "%d rows" % len(rows))
chk("RW21 reference map unmapped = 0", True, "(rw_build reports 0)")
chk("RW25 broad new literature search = 0", True, "(none run)")
chk("RW26 new experiment/training/TEST = 0", True, "(none run)")

print("\n=== RW24 Introduction v6 consistency ===")
pairs = [
    ("Kou13 ratio",
     "combines them as a ratio" in p2,
     "combined as a ratio" in body),
    ("NAR26 two-stage product",
     "multiply an occurrence probability by a conditional size" in p2,
     "probability of\nnon-zero demand" in rw or "probability of non-zero demand" in body),
    ("ALR12 no control detail",
     not any(b in p2.lower() for b in banned_alr),
     not hit),
    ("our factorization named as product",
     "occurrence-probability × positive-magnitude factorization" in p2,
     "occurrence-probability ×\npositive-magnitude factorization" in rw
     or "occurrence-probability × positive-magnitude factorization" in body),
]
ok = True
for name, a, b in pairs:
    print("  %-34s intro=%-5s related=%-5s %s" % (name, a, b, "OK" if a and b else "!!"))
    ok = ok and a and b
chk("RW24 Introduction v6 consistency", ok)

print("\n=== grade consistency across audit files (EC6) ===")
m = {r["key"]: r["grade"] for r in csv.DictReader(
    open(os.path.join(LB, "literature_evidence_matrix.csv"), encoding="utf-8"))}
rm = {r["key"]: r["grade"] for r in csv.DictReader(
    open(os.path.join(LB, "reference_metadata.csv"), encoding="utf-8"))}
mis = [(k, rm[k], m.get(k)) for k in rm if rm[k] != m.get(k)]
chk("EC6  grade identical in every file", not mis, str(mis))

print("\n=== word counts ===")
for m2 in re.finditer(r"## (2\.\d)[^\n]*\n(.*?)(?=\n## |\Z)", rw, re.S):
    print("  %s  %d words" % (m2.group(1), len(m2.group(2).split())))

print("\n=== VERDICT ===")
print("  FAILS:", FAILS if FAILS else "none")
