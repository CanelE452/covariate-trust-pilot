import os, re, csv, hashlib
R=r"E:\CODING\proj\covariate-trust-pilot\results"
PW=os.path.join(R,"paper_writing_verified"); MO=os.path.join(R,"paper_manuscript_verified")

def body(f, start="---\n"):
    t=open(os.path.join(PW,f),encoding="utf-8").read()
    return t.split(start,1)[1].strip()

PARTS=[("Abstract","abstract_v1.md"),("1 Introduction","introduction_v6.md"),
       ("2 Related Work","related_work_v3.md"),("3-5 Methods","methods_v1.md"),
       ("4.6-5.9 Results","results_v1.md"),("6 Discussion","discussion_v1.md"),
       ("7 Conclusion","conclusion_v1.md"),("Figure captions","figure_captions_v1.md"),
       ("Table captions","table_captions_v1.md")]
out=["# Manuscript v1 — assembled draft\n",
     "Assembled 2026-08-12 from the frozen section files listed in",
     "`manuscript_section_map.md`. No sentence was rewritten during assembly; each section",
     "is its source file's body verbatim. Section numbering follows",
     "`../paper_outline_verified/final_outline_freeze.md`.\n","---\n"]
smap=[["manuscript_section","source_file","sha256_12","words"]]
for title,f in PARTS:
    b=body(f)
    # strip the per-file H1 heading, keep内部 structure
    b=re.sub(r"^#\s+[^\n]*\n+","",b)
    h=hashlib.sha256(open(os.path.join(PW,f),"rb").read()).hexdigest()[:12]
    smap.append([title,f,h,len(b.split())])
    out.append("\n\n<!-- ===== %s  (source: %s) ===== -->\n"%(title,f))
    out.append(b)
open(os.path.join(MO,"manuscript_v1.md"),"w",encoding="utf-8").write("\n".join(out)+"\n")
with open(os.path.join(MO,"manuscript_section_map.csv"),"w",newline="",encoding="utf-8") as fh:
    csv.writer(fh).writerows(smap)
tot=sum(r[3] for r in smap[1:])
print("assembled manuscript_v1.md ; total %d words"%tot)
for r in smap[1:]: print("  %-18s %-24s %s %5dw"%(r[0],r[1],r[2],r[3]))
