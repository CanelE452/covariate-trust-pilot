"""Split related_work_v1 into sentences, build the reference map, run RW gate."""
import csv, os, re

PW = r"E:\CODING\proj\covariate-trust-pilot\results\paper_writing_verified"
LB = r"E:\CODING\proj\covariate-trust-pilot\results\literature_boundary_verified"

raw = open(os.path.join(PW, "related_work_v2.md"), encoding="utf-8").read()
body = raw.split("---\n", 1)[1]
body = body.split("\n<sup>1</sup>")[0]

SEC = {}
cur = None
for block in re.split(r"\n(?=## )", body):
    m = re.match(r"## (2\.\d)\s+(.+)", block)
    if not m:
        continue
    cur = m.group(1)
    text = block.split("\n", 1)[1]
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text) if p.strip()]
    SEC[cur] = (m.group(2).strip(), paras)


def sentences(p):
    p = p.replace("<sup>1</sup>", "")
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(“\"])", p)
    return [s.strip() for s in parts if s.strip()]


rows = []
for sec, (title, paras) in SEC.items():
    for pi, p in enumerate(paras, 1):
        for si, s in enumerate(sentences(p), 1):
            keys = sorted(set(k.strip() for g in re.findall(r"\[([A-Za-z0-9;\s]+?)\]", s)
                             for k in g.split(";")
                             if re.fullmatch(r"[A-Za-z]+\d{2}", k.strip())))
            rows.append(dict(section=sec, pid="P%d" % pi, sid="S%d" % si,
                             text=s, keys=keys))

print("=== structure ===")
tot = 0
for sec, (title, paras) in SEC.items():
    w = sum(len(p.split()) for p in paras)
    tot += w
    print("  %-4s %-52s %4dw  %d para  %d sent" %
          (sec, title[:52], w, len(paras),
           sum(1 for r in rows if r["section"] == sec)))
print("  TOTAL %d words, %d sentences" % (tot, len(rows)))

allkeys = [k for r in rows for k in r["keys"]]
print("  citations in prose: %d occurrences, %d unique %s" %
      (len(allkeys), len(set(allkeys)), sorted(set(allkeys))))

# literature-factual = any sentence naming a work or describing prior work
LITMARK = re.compile(
    r"\[[A-Za-z]+\d{2}\]|Croston|Syntetos|Kourentzes|Altay|Litteral|Rudisill|"
    r"prior|previous|literature|established|has been|have been|studies|study reports",
    re.I)
lit = [r for r in rows if LITMARK.search(r["text"])]
unmapped = [r for r in lit if not r["keys"] and not re.search(
    r"this paper|present study|the design|here\b|no part of the present", r["text"], re.I)]
print("  literature-factual sentences: %d ; unmapped: %d" % (len(lit), len(unmapped)))
for r in unmapped:
    print("    UNMAPPED %s%s%s: %s" % (r["section"], r["pid"], r["sid"], r["text"][:90]))

SCOPE = {
    "Cro72": "Croston (1972) JORS 23(3) 289-303; size/interval separation, ratio rate",
    "SB05": "Syntetos & Boylan (2005) IJF 21(2) 303-314; SBA (1-alpha/2) bias correction",
    "TSB11": "Teunter, Syntetos & Babai (2011) EJOR 214(3) 606-615; occurrence-probability "
             "updating every period; obsolescence",
    "SBC05": "Syntetos, Boylan & Croston (2005) JORS 56(5) 495-503; ADI/CV^2 categorization, "
             "3000 automotive series",
    "KH06": "Kostenko & Hyndman (2006) JORS 57(10) 1256-1257; boundary refinement",
    "WSS04": "Willemain, Smart & Schwarz (2004) IJF 20(3) 375-387; two-state Markov "
             "occurrence + jittered bootstrap; nine industrial datasets",
    "ALR12": "Altay, Litteral & Rudisill (2012) IJPE 135(1) 275-283; size AC, interval AC, "
             "size-interval cross-corr varied in generated demand; forecast + inventory "
             "outcomes; ABSTRACT/RECORD LEVEL ONLY",
    "Kou13": "Kourentzes (2013) IJPE 143(1) 198-206; FULL TEXT; NN-Rate single output vs "
             "NN-Dual size+interval combined as ratio, de-biased; per-arm best (I,H)",
    "NAR26": "Nathan et al. (2026) Sci Rep 16:4792; FULL TEXT (PMC12873174); LightGBM "
             "direct vs classifier x Tweedie regressor product; identical preprocessing, "
             "feature construction, evaluation protocols",
    "TJWC21": "Turkmen, Januschowski, Wang & Cemgil (2021) PLOS ONE 16(11) e0259764; deep "
              "renewal processes; illustrative synthetic patterns",
    "GDTP25": "Giannopoulos et al. (2025) IJPR online 31 Oct 2025; ML review",
    "MC26": "Musat & Cabuz (2026) arXiv:2602.22685; PREPRINT, NOT peer-reviewed",
}
WARN = {
    "ALR12": "LIT-W3 OPEN: full text unobtained; no statement about what it holds constant",
    "Kou13": "LIT-W-KOU13: ratio form, never 'hurdle', never 'the same two representations'",
    "NAR26": "LIT-W-NAR26: NOT neural; capacity/training/tuning match NOT STATED (U)",
    "MC26": "arXiv preprint; must be labelled as not peer-reviewed",
    "GDTP25": "LIT-W2: volume/pages unassigned (online first)",
    "Cro72": "LIT-W1: published in Operational Research Quarterly; indexed under JORS",
}

out = os.path.join(PW, "related_work_v2_reference_map.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["section", "paragraph_id", "sentence_id", "claim_summary",
                "citation_key_or_paper_id", "verified_source", "evidence_scope",
                "warning", "notes"])
    n = 0
    for r in lit:
        if not r["keys"]:
            continue
        for k in r["keys"]:
            w.writerow([r["section"], r["pid"], r["sid"], r["text"][:170], k,
                        "literature_boundary_verified/core_reference_list.md",
                        SCOPE[k], WARN.get(k, ""), ""])
            n += 1
print("  wrote %s (%d mapped rows)" % (out, n))
