"""Assemble manuscript v2 in strict numeric section order."""
import csv, hashlib, os, re

R = r"E:\CODING\proj\covariate-trust-pilot\results"
PW = os.path.join(R, "paper_writing_verified")
MO = os.path.join(R, "paper_manuscript_verified")


def body(f):
    t = open(os.path.join(PW, f), encoding="utf-8").read()
    b = t.split("---\n", 1)[1].strip()
    return re.sub(r"^#\s+[^\n]*\n+", "", b)


# methods_v1 holds 3.x + 4.1-4.5 + 5.1-5.3; split so Section 5 design lands after 4.8
meth = body("methods_v1.md")
i = meth.index("## 5. Empirical Validation")
meth_A, meth_B = meth[:i].rstrip(), meth[i:].strip()

# results_v1 holds 4.6-4.8 + 5.4-5.9; split at 5.4
res = body("results_v1.md")
j = res.index("## 5.4 Overall comparison")
res_A, res_B = res[:j].rstrip(), res[j:].strip()

PARTS = [
    ("Abstract", "abstract_v2.md", body("abstract_v2.md")),
    ("1 Introduction", "introduction_v6.md", body("introduction_v6.md")),
    ("2 Related Work", "related_work_v3.md", body("related_work_v3.md")),
    ("3 + 4.1-4.5 Setup and synthetic design", "methods_v1.md (part A)", meth_A),
    ("4.6-4.8 Synthetic results", "results_v1.md (part A)", res_A),
    ("5.1-5.3 Empirical design", "methods_v1.md (part B)", meth_B),
    ("5.4-5.9 Empirical results", "results_v1.md (part B)", res_B),
    ("6 Discussion", "discussion_v2.md", body("discussion_v2.md")),
    ("7 Conclusion", "conclusion_v1.md", body("conclusion_v1.md")),
    ("Figure captions", "figure_captions_v1.md", body("figure_captions_v1.md")),
    ("Table captions", "table_captions_v1.md", body("table_captions_v1.md")),
]

out = ["# Manuscript v2 — assembled draft\n",
       "Assembled 2026-08-12 from the section files listed in",
       "`manuscript_v2_section_map.md`. Sections appear in **strict numeric order**; the",
       "v1 assembly presented 5.1-5.3 before 4.6, which is the defect this build fixes.",
       "No sentence was rewritten during assembly.\n", "---\n"]
smap = [["manuscript_section", "source_file", "words"]]
for title, src, b in PARTS:
    smap.append([title, src, len(b.split())])
    out.append("\n\n<!-- ===== %s  (source: %s) ===== -->\n" % (title, src))
    out.append(b)

open(os.path.join(MO, "manuscript_v2.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
with open(os.path.join(MO, "manuscript_v2_section_map.csv"), "w", newline="",
          encoding="utf-8") as f:
    csv.writer(f).writerows(smap)

M = open(os.path.join(MO, "manuscript_v2.md"), encoding="utf-8").read()
nums = [re.match(r"(\d\.\d)", m.group(1)).group(1)
        for m in re.finditer(r"^#{2,3}\s+((?:\d\.\d)[^\n]*)", M, re.M)]
back = [(a, b) for a, b in zip(nums, nums[1:]) if float(b) < float(a)]
print("total words:", sum(r[2] for r in smap[1:]))
print("section order:", " -> ".join(nums))
print("BACKWARD JUMPS:", back if back else "none")
for r in smap[1:]:
    print("  %-40s %-26s %5dw" % (r[0], r[1], r[2]))
