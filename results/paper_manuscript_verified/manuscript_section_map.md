# Manuscript section map

Assembled 2026-08-12. Each section is its source file body verbatim; no sentence was
rewritten during assembly.

```
manuscript section   source file                sha256-12      words
----------------------------------------------------------------------------
Abstract             abstract_v1.md             b714cc8c986e     265
1 Introduction       introduction_v6.md         48c46c8c7a2e     881
2 Related Work       related_work_v3.md         8755d241961a    1412
3-5 Methods          methods_v1.md              ad2c600e5d7e    2634
4.6-5.9 Results      results_v1.md              7f11ba08e511    1931
6 Discussion         discussion_v1.md           1ccc1d7626ed    1430
7 Conclusion         conclusion_v1.md           060d6ee8b0b5     277
Figure captions      figure_captions_v1.md      6b50e2108b0c     547
Table captions       table_captions_v1.md       c969fe2858da     426

TOTAL 9803 words
```

## Numbering, against final_outline_freeze.md

```
1  Introduction                                 introduction_v6.md
2  Related Work                2.1-2.5          related_work_v3.md
3  Problem Setup               3.1-3.4          methods_v1.md
4  Controlled Synthetic Study  4.1-4.5 design   methods_v1.md
                               4.6-4.8 results  results_v1.md
5  Empirical Validation        5.1-5.3 design   methods_v1.md
                               5.4-5.9 results  results_v1.md
6  Discussion                  6.1-6.6          discussion_v1.md
7  Conclusion                                   conclusion_v1.md
   Abstract                                     abstract_v1.md
   Figure / Table captions                      figure_captions_v1.md, table_captions_v1.md
```

The design/result split inside Sections 4 and 5 is the one refinement proposed in
`../paper_methods_verified/methods_structure.md`. It preserves the frozen outline's
section numbers and adds nothing; if the venue prefers the merged form the prose
recombines without editing a sentence.

## Not yet in the manuscript

```
title            five candidates in ../paper_writing_verified/title_candidates.md; not chosen
bibliography     twelve verified records in literature_boundary_verified/core_reference_list.md;
                 no .bib generated
appendices       roles specified in the caption files; prose not written
figures/tables   drafts rendered in ../paper_rendering_verified/drafts/; captions frozen here
```
