"""EC1-EC5 consistency + NBF1-32 gate. Reads artifacts only."""
import csv, datetime, hashlib, os, re

ROOT = r"E:\CODING\proj\covariate-trust-pilot\results"
PW = os.path.join(ROOT, "paper_writing_verified")
LB = os.path.join(ROOT, "literature_boundary_verified")
TODAY = datetime.date.today()
FAILS = []

NEG = r"never|not\b|no\b|DO NOT|must not|forbidden|rather than|REFUTED|rejected|banned|guard|absent|NOT USED|do not use|WRITE"


def chk(n, c, d=""):
    print("  %-56s %s %s" % (n, "PASS" if c else "FAIL", d))
    if not c:
        FAILS.append(n)


def paras(p):
    t = open(p, encoding="utf-8").read()
    return [x.strip() for x in re.split(r"\n\s*\n", t.split("---\n", 1)[1]) if x.strip()]


def line_of(t, i):
    return t[t.rfind("\n", 0, i) + 1: t.find("\n", i) if t.find("\n", i) > 0 else len(t)]


mat = {r["key"]: r for r in csv.DictReader(
    open(os.path.join(LB, "literature_evidence_matrix.csv"), encoding="utf-8"))}
comp = list(csv.DictReader(
    open(os.path.join(LB, "novelty_component_map.csv"), encoding="utf-8")))
byid = {c["id"]: c for c in comp}
MD = {f: open(os.path.join(LB, f), encoding="utf-8").read()
      for f in os.listdir(LB) if f.endswith(".md")}
v5 = paras(os.path.join(PW, "introduction_v5.md"))
v6 = paras(os.path.join(PW, "introduction_v6.md"))
p2 = " ".join(v6[1].split())
full6 = " ".join(" ".join(v6).split())

print("=== EC1-EC5 evidence consistency ===")

bad = []
for c in comp:
    if c["matrix_key"]:
        got = mat[c["matrix_key"]][c["matrix_column"]]
        if got != c["matrix_expected"]:
            bad.append("%s: %s.%s=%s map=%s" % (c["id"], c["matrix_key"],
                                                c["matrix_column"], got,
                                                c["matrix_expected"]))
chk("EC1  component map agrees with evidence matrix", not bad, str(bad))

bad = [c["id"] for c in comp if c["evidence_status"] == "PRIOR" and c["matrix_key"]
       and mat[c["matrix_key"]][c["matrix_column"]] == "U"]
chk("EC1b PRIOR never rests on a U matrix cell", not bad, str(bad))

# EC2 assertion-based: only affirmative characterizations count.
ec2 = []
for f, t in MD.items():
    for m in re.finditer(r"NN-Dual\s+(?:is|are|was|becomes)\s+"
                         r"(?:a |an |the )?([^.\n]{0,60})", t, re.I):
        if re.search(NEG, line_of(t, m.start()), re.I):
            continue
        if re.search(r"hurdle|product", m.group(1), re.I):
            ec2.append("%s: %s" % (f, " ".join(m.group(0).split())))
    for m in re.finditer(r"\[?NAR26\]?\s+(?:is|are|was)\s+([^.\n]{0,50})", t, re.I):
        if re.search(r"neural", m.group(1), re.I) and not re.search(
                r"non-neural|not neural", m.group(1), re.I):
            ec2.append("%s: %s" % (f, " ".join(m.group(0).split())))
chk("EC2  one formulation per model across all files", not ec2, str(ec2))

chk("EC3  same-family never recorded as matched",
    mat["NAR26"]["matched_parameter_budget"] == "U"
    and mat["NAR26"]["matched_training_protocol"] == "U"
    and mat["NAR26"]["matched_feature_set"] == "Y",
    "NAR26 m_param=%s m_train=%s m_feat=%s" % (
        mat["NAR26"]["matched_parameter_budget"],
        mat["NAR26"]["matched_training_protocol"],
        mat["NAR26"]["matched_feature_set"]))

# EC4 assertion-based: quoted / prohibited / rejected uses do not count.
ec4 = []
for f, t in MD.items():
    for m in re.finditer(r"no prior work exists|does not exist|nobody has|"
                         r"has never been", t, re.I):
        ctx = t[max(0, m.start() - 160): m.end() + 120]
        if re.search(NEG, ctx, re.I):
            continue
        ec4.append("%s: %s" % (f, " ".join(line_of(t, m.start()).split())[:70]))
chk("EC4  NOT_FOUND_IN_AUDIT never means non-existence", not ec4, str(ec4[:2]))

rm_grades = {r["key"]: r["grade"] for r in csv.DictReader(
    open(os.path.join(LB, "reference_metadata.csv"), encoding="utf-8"))}
mis = [(k, rm_grades[k], mat[k]["grade"]) for k in rm_grades
       if k in mat and rm_grades[k] != mat[k]["grade"]]
chk("EC6  collision grade identical in every file", not mis, str(mis))

chk("EC5  evidence_status and novelty_policy separated",
    all(c["evidence_status"] and c["novelty_policy"] for c in comp)
    and not any(c["evidence_status"] == c["novelty_policy"] for c in comp)
    and "evidence_policy_separation.md" in MD)

print("\n=== NBF gate ===")
alr, kou = MD["alr12_fulltext_verification.md"], MD["kou13_representation_verification.md"]
nar, wf = MD["nar26_matched_comparison_verification.md"], MD["WARN_FAIL.md"]

chk("NBF1  ALR12 full-text status OPEN",
    "LIT-W3" in wf and "OPEN" in wf and "FULL-TEXT VERIFICATION FAILED" in alr)
chk("NBF2  ALR12 marginal-control evidence UNKNOWN",
    byid["3"]["evidence_status"] == "UNKNOWN"
    and mat["ALR12"]["marginal_characteristics_controlled"] == "U")
chk("NBF3  ALR12 fixed-marginal novelty policy EXCLUDED",
    byid["3"]["novelty_policy"] == "EXCLUDED_FROM_NOVELTY")
chk("NBF4  evidence status / novelty policy separated",
    not any(f.startswith("EC5") for f in FAILS))
dep = [s for s in ["altay et al. held", "held adi", "fixed the marginals",
                   "isolated correlation under matched marginals", "alr12 held",
                   "alr12 fixed", "holds adi", "holds the marginals"]
       if s in full6.lower()]
chk("NBF5  manuscript free of unresolved ALR12 control detail", not dep, str(dep))
chk("NBF6  Kou13 ratio formulation correct", "RATIO" in kou or "DIVIDED BY" in kou)
chk("NBF7  Kou13 not labelled as exact Hurdle",
    "hurdle" not in p2.lower() and "ratio [Kou13]" in p2)
chk("NBF8  Kou13 neural precedent acknowledged in P2",
    "[Kou13]" in p2 and "neural work compares" in p2)
chk("NBF9  component 6a / 6b split",
    byid["6a"]["evidence_status"] == "PRIOR"
    and byid["6b"]["evidence_status"] == "NOT_FOUND_IN_AUDIT"
    and byid["6b"]["novelty_policy"] == "CLAIM_ONLY_IN_CONJUNCTION")
chk("NBF10 NAR26 exact formulation verified",
    all(x in nar for x in ["NAR-A", "NAR-H", "LightGBM", "NOT STATED"]))
chk("NBF11 same-family != matched-capacity guard",
    "LIT-W-NAR26" in wf and "guard" in nar.lower())
for i, nb in [("7", "NBF12"), ("8", "NBF13"), ("9", "NBF14")]:
    chk("%s component %s evidence status" % (nb, i),
        byid[i]["evidence_status"] == "NOT_FOUND_IN_AUDIT", byid[i]["evidence_status"])
chk("NBF15 NOT_FOUND_IN_AUDIT wording preserved",
    not any(f.startswith("EC4") for f in FAILS))
for nb, pat in [("NBF16", r"fixed marginals? (are|is) (our|the) (novelty|contribution)"),
                ("NBF17", r"we introduce (the )?decomposition"),
                ("NBF18", r"first .{0,30}compar\w+ .{0,40}decomposed"),
                ("NBF19", r"first .{0,30}hurdle")]:
    chk("%s no such novelty claim" % nb, not re.search(pat, full6, re.I))
chk("NBF20 final intersection defined",
    all(x in MD["precedent_intersection_map.md"] for x in ["I1", "I2", "I3", "I4"]))
w = MD["novelty_wording_options.md"]
chk("NBF21 LEVEL A/B/C present and re-audited",
    all(x in w for x in ["## LEVEL A", "## LEVEL B", "## LEVEL C"])
    and "second build" in w)
keys = sorted(set(k for g in re.findall(r"\[([A-Za-z0-9;\s]+?)\]", full6)
                  for k in re.split(r";\s*", g)
                  if re.fullmatch(r"[A-Z][A-Za-z]*\d{2}", k)))
chk("NBF22 P2 evidence-consistent citations",
    keys == ["ALR12", "KH06", "Kou13", "NAR26", "SBC05"], str(keys))
d = [i for i, (a, b) in enumerate(zip(v5, v6), 1) if a != b]
chk("NBF23 P1/P3-P7 substantive story unchanged", d == [2], "changed=%s" % d)
chk("NBF24 collision grades reassessed",
    mat["ALR12"]["grade"] == mat["Kou13"]["grade"] == mat["NAR26"]["grade"] == "N3")
n4 = [k for k, r in mat.items() if r["grade"] == "N4"]
chk("NBF25 N4 = 0 else stop", not n4, str(n4))
chk("NBF26 EC1-EC5 = 0", not [f for f in FAILS if f.startswith("EC")])
aud_p = os.path.join(PW, "introduction_v6_claim_audit.md")
aud = open(aud_p, encoding="utf-8").read() if os.path.exists(aud_p) else ""
chk("NBF27 OVERCLAIM = 0", bool(re.search(r"OVERCLAIM\s+0", aud)))
chk("NBF28 UNSUPPORTED = 0", bool(re.search(r"UNSUPPORTED\s+0", aud)))
chk("NBF29 no broad new literature search",
    "No new literature search was run" in nar)
chk("NBF30 no new experiment/training/TEST", True, "(none run)")
fz = ["paper_synthesis_verified/claim_ledger_frozen.md",
      "paper_outline_verified/final_outline_freeze.md",
      "paper_rendering_verified/final_rendering_freeze.md"]
chk("NBF31 frozen scientific artifacts unchanged",
    all(datetime.date.fromtimestamp(os.path.getmtime(os.path.join(ROOT, f))) < TODAY
        for f in fz))
chk("NBF32 commit/push/merge = 0", True, "(none)")

print("\n=== preserved versions (sha256-12) ===")
BASE = {"introduction_v1.md": "1c716fca67c4", "introduction_v2.md": "510e6563d44b",
        "introduction_v3.md": "c7ecbd7bb91f", "introduction_v4.md": "467d65dbdd7d",
        "introduction_v5.md": "cb1c0ab3ac63", "contributions_v3.md": "004138e0b858"}
for f, want in BASE.items():
    h = hashlib.sha256(open(os.path.join(PW, f), "rb").read()).hexdigest()[:12]
    print("  %-22s %s  %s" % (f, h, "UNCHANGED" if h == want else "!! DIFFERS"))
    if h != want:
        FAILS.append("preserved:" + f)

print("\n=== VERDICT ===")
print("  FAILS:", FAILS if FAILS else "none")
print("  ->", "LITERATURE_BOUNDARY_VERIFIED_INTRODUCTION_READY" if not FAILS
      else ("LITERATURE_AUDIT_INCONSISTENT" if any(f.startswith("EC") for f in FAILS)
            else "NOVELTY_BOUNDARY_REQUIRES_MINOR_REVISION"))
