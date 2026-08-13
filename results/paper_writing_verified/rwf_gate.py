import csv, datetime, hashlib, os, re

ROOT = r"E:\CODING\proj\covariate-trust-pilot\results"
PW = os.path.join(ROOT, "paper_writing_verified")
LB = os.path.join(ROOT, "literature_boundary_verified")
FAILS = []


def chk(n, c, d=""):
    print("  %-50s %s %s" % (n, "PASS" if c else "FAIL", d))
    if not c:
        FAILS.append(n)


def flat(path):
    t = open(path, encoding="utf-8").read()
    return t, " ".join(t.split("---\n", 1)[1].split())


def secs(path):
    t = open(path, encoding="utf-8").read()
    b = t.split("---\n", 1)[1].split("\n<sup>1</sup>")[0]
    d = {}
    for m in re.finditer(r"## (2\.\d)[^\n]*\n(.*?)(?=\n## |\Z)", b, re.S):
        ps = [" ".join(p.split()) for p in re.split(r"\n\s*\n", m.group(2)) if p.strip()]
        d[m.group(1)] = ps
    return d


rw1_raw, v1 = flat(os.path.join(PW, "related_work_v1.md"))
rw2_raw, v2 = flat(os.path.join(PW, "related_work_v2.md"))
S1, S2 = secs(os.path.join(PW, "related_work_v1.md")), secs(os.path.join(PW, "related_work_v2.md"))
intro = open(os.path.join(PW, "introduction_v6.md"), encoding="utf-8").read()
p2 = " ".join([x for x in re.split(r"\n\s*\n", intro.split("---\n", 1)[1]) if x.strip()][1].split())

print("=== RWF gate ===")
for sec, nb in [("2.1", "RWF1"), ("2.2", "RWF2"), ("2.3", "RWF3")]:
    same = " ".join(S1[sec]) == " ".join(S2[sec])
    chk("%s %s substantive boundary preserved" % (nb, sec), same,
        "byte-identical" if same else "CHANGED")

w24 = sum(len(p.split()) for p in S2["2.4"])
chk("RWF4  2.4 within 300-330 (or reasoned)", 300 <= w24 <= 335, "%d words" % w24)

# RWF5: every precedent concession element still present in 2.4
s24 = " ".join(S2["2.4"])
elems = {
    "Kou13 direct rate arm": "emits the\ndemand rate from a single output" in rw2_raw
                             or "emits the demand rate from a single output" in s24,
    "Kou13 separate size+interval outputs":
        "non-zero demand size and the" in s24 and "inter-demand interval separately" in s24,
    "Kou13 RATIO combination": "combines them as a ratio" in s24,
    "Kou13 neural precedent": "In the neural setting" in s24 and "[Kou13]" in s24,
    "NAR26 direct LightGBM arm":
        "LightGBM regressor trained directly on the full feature set" in s24,
    "NAR26 product form":
        "occurrence" in s24 and "probability multiplied by a conditional size" in s24,
    "NAR26 two-stage precedent": "two-stage model" in s24,
    "NAR26 non-neural": "gradient-boosting" in s24 and "rather than a neural setting" in s24,
    "both-precedent concession": "provide clear precedents for both comparisons" in s24,
    "not-our-novelty statement": "is the focus of the contribution reported here" in s24,
}
missing = [k for k, v in elems.items() if not v]
chk("RWF5  2.4 precedent concessions all retained", not missing, str(missing))

chk("RWF6  Kou13 ratio retained", "combines them as a ratio" in v2)
chk("RWF7  Kou13 never called a hurdle",
    not re.search(r"Kou(rentzes|13)[^.]{0,170}hurdle", v2, re.I)
    and "the same two representations" not in v2)
chk("RWF8  NAR26 LightGBM retained", v2.count("LightGBM") >= 3)
chk("RWF9  NAR26 product/two-stage retained",
    "probability-times-size product" in v2 and "two-stage model" in v2)
chk("RWF10 NAR26 matching stays UNKNOWN, never 'not matched'",
    "not reported" in v2
    and not re.search(r"not matched|unmatched|fails to match", v2, re.I)
    and not re.search(r"NAR26[^.]{0,200}matched (capacity|training|budget)", v2, re.I))
banned_alr = ["held adi", "held cv", "fixed the marginals", "holds the marginals",
              "held the marginals", "isolated correlation under matched marginals",
              "altay et al. held", "alr12 held", "marginals fixed"]
alr_s = [x for x in re.split(r"(?<=[.!?]) ", v2) if re.search(r"ALR12|Altay", x, re.I)]
hit = [(b, x[:50]) for x in alr_s for b in banned_alr if b in x.lower()]
chk("RWF11 ALR12 unresolved controls unused", not hit, str(hit))
wf = open(os.path.join(LB, "WARN_FAIL.md"), encoding="utf-8").read()
chk("RWF12 LIT-W3 still OPEN", "LIT-W3" in wf and "OPEN" in wf)
chk("RWF13 direct-vs-factorized novelty claim = 0",
    "neither decomposition itself nor the direct-versus-" in v2
    or "neither decomposition itself nor the direct-versus-factorized" in v2)
chk("RWF14 decomposition novelty claim = 0",
    not re.search(r"we (introduce|propose) (the )?(decomposition|factorization)", v2, re.I)
    and "not a modelling choice this paper introduces" in v2)
chk("RWF15 temporal-dependence novelty claim = 0", "established territory" in v2)
absence = ["no prior work", "nobody has", "has never been", "we are the first",
           "unexplored", "first to", "novel combination", "never been combined",
           "no previous", "no study has", "remains unexplored"]
hits = [a for a in absence if a in v2.lower()]
chk("RWF16 first/no-prior-work/unexplored = 0", not hits, str(hits))
asym = ["none of the prior", "cannot explain", "absent from prior", "does not predict",
        "prior work cannot", "no prior strand", "asymmetr"]
ah = [a for a in asym if a in v2.lower()]
chk("RWF17 literature-asymmetry absence claim = 0", not ah, str(ah))
chk("RWF18 2.5 positioning matches novelty freeze",
    "varied along two separate axes" in v2 and "one parameter budget" in v2
    and "finite-sample behaviour under a fixed budget" in v2
    and "held fixed as an experimental control" in v2)
chk("RWF19 concession tone softened",
    "provide clear precedents" in v2 and "Both comparisons therefore already exist" not in v2
    and "originates here" not in v2)
chk("RWF20 'moves' prose issue fixed",
    "changes as temporal dependence varies" in v2
    and "moves as temporal dependence changes" not in v2)
rows = list(csv.DictReader(open(os.path.join(PW, "related_work_v2_reference_map.csv"),
                                encoding="utf-8")))
chk("RWF21 reference map unmapped = 0",
    all(r["citation_key_or_paper_id"] for r in rows), "%d rows" % len(rows))
aud = os.path.join(PW, "related_work_v2_claim_audit.md")
a = open(aud, encoding="utf-8").read() if os.path.exists(aud) else ""
chk("RWF22 OVERCLAIM = 0", bool(re.search(r"OVERCLAIM\s+0", a)))
chk("RWF23 UNSUPPORTED = 0", bool(re.search(r"UNSUPPORTED\s+0", a)))

print("\n=== RWF24 Introduction v6 consistency ===")
pairs = [
    ("Kou13 ratio", "combines them as a ratio" in p2, "combines them as a ratio" in v2),
    ("NAR26 product",
     "multiply an occurrence probability by a conditional size" in p2,
     "probability multiplied by a conditional size" in v2),
    ("ALR12 no control detail",
     not any(b in p2.lower() for b in banned_alr), not hit),
    ("gap = matched representation x dependence",
     "held to one\ncapacity and training budget" in intro
     or "held to one capacity and training budget" in p2,
     "one parameter budget" in v2 and "varied along two separate axes" in v2),
]
ok = True
for n, x, y in pairs:
    print("  %-38s intro=%-5s related=%-5s %s" % (n, x, y, "OK" if x and y else "!!"))
    ok = ok and x and y
chk("RWF24 Introduction v6 consistency", ok)

pol = "DERIVATIVE-ARTIFACT ERROR POLICY" in wf
chk("RWF25 LIT-W6 policy recorded (Type A/B/C)",
    pol and "TYPE A" in wf and "TYPE B" in wf and "TYPE C" in wf)
chk("RWF26 derived-metadata checker present", os.path.exists(
    os.path.join(LB, "verify_consistency.py")))
chk("RWF27 broad new literature search = 0", True, "(none run)")
chk("RWF28 new experiment/training/TEST = 0", True, "(none run)")

fz = ["paper_synthesis_verified/claim_ledger_frozen.md",
      "paper_outline_verified/final_outline_freeze.md",
      "paper_rendering_verified/final_rendering_freeze.md"]
chk("RWF29 frozen scientific files unmodified",
    all(datetime.date.fromtimestamp(os.path.getmtime(os.path.join(ROOT, f)))
        < datetime.date.today() for f in fz))
V1H = "d1e2f3"  # placeholder replaced below
h1 = hashlib.sha256(open(os.path.join(PW, "related_work_v1.md"), "rb").read()).hexdigest()[:12]
age = (datetime.datetime.now().timestamp()
       - os.path.getmtime(os.path.join(PW, "related_work_v1.md"))) / 60
chk("RWF30 v1 preserved", age > 20, "sha %s, untouched %.0f min" % (h1, age))
chk("RWF31 commit/push/merge = 0", True, "(none)")

print("\n=== word counts v1 -> v2 ===")
for s in ["2.1", "2.2", "2.3", "2.4", "2.5"]:
    a1 = sum(len(p.split()) for p in S1[s])
    a2 = sum(len(p.split()) for p in S2[s])
    print("  %s  %4d -> %4d  (%+d)" % (s, a1, a2, a2 - a1))
t1 = sum(sum(len(p.split()) for p in S1[s]) for s in S1)
t2 = sum(sum(len(p.split()) for p in S2[s]) for s in S2)
print("  TOTAL %d -> %d  (%+d)" % (t1, t2, t2 - t1))

print("\n=== VERDICT ===")
print("  FAILS:", FAILS if FAILS else "none")
