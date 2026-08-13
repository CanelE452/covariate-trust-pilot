# Literature boundary audit — search log

Run 2026-08-12. Tool: web search + publisher/DOI verification + Crossref REST API.

Rule applied throughout: **a search snippet is never the evidence.** Every entry below is
graded by how the record was actually confirmed:

```
CROSSREF   metadata confirmed against api.crossref.org/works/{DOI}
FULLTEXT   the paper's own text (PDF or publisher HTML) was read for the claim made about it
PUBLISHER  a publisher / repository record page was read (abstract-level only)
SNIPPET    only a search snippet was seen  -- NOT usable for a collision verdict
```

---

## Axis 1 — ADI / CV², the categorization scheme

```
Q1-1  "Syntetos Boylan Croston 2005 categorization intermittent demand ADI CV2
       classification original paper"
      -> Syntetos, Boylan & Croston (2005), JORS 56(5) 495-503.            PUBLISHER
      -> surfaced a follow-up note by Kostenko & Hyndman.
Q1-2  "On the categorization of demand patterns ... DOI 10.1057 ... vol 56 p495"
      -> DOI 10.1057/palgrave.jors.2601841 confirmed on Springer and T&F.  PUBLISHER
      -> categorization rules expressed in terms of the average inter-demand interval
         and the squared coefficient of variation of demand sizes; validated on 3000
         automotive series.
Q1-3  "Kostenko Hyndman A note on the categorization of demand patterns ... 2006"
      -> JORS 57(10) 1256-1257, DOI 10.1057/palgrave.jors.2602211.         PUBLISHER
Q1-X  both DOIs re-resolved through Crossref.                              CROSSREF
```

Axis 1 closed at three records. The protocol's target ("1~3편이면 충분") is met, and the
two are the origin and its correction, which is the minimum honest pair.

---

## Axis 2 — temporal dependence in intermittent demand *(the decisive axis)*

```
Q2-1  "intermittent demand autocorrelation inter-demand intervals demand sizes
       cross-correlation forecasting stock control simulation study"
      -> Altay, Litteral & Rudisill, "Effects of correlation on intermittent demand
         forecasting and stock control", IJPE.                             SNIPPET
Q2-2  "Altay Litteral Rudisill ... IJPE 2012 volume pages DOI"
      -> 135(1) 275-283.                                                   SNIPPET
Q2-3  RePEc record fetched directly
      -> DOI 10.1016/j.ijpe.2011.08.002; abstract read: three correlation types,
         compound Poisson generated demand, service level and cost outcomes,
         higher intermittency intensifies the effect.                      PUBLISHER
      -> ScienceDirect returned HTTP 403; the RePEc record was used instead.
Q2-4  "Willemain Smart Schwarz 2004 ... bootstrap Markov chain autocorrelation"
      -> IJF 20(3) 375-387.                                                PUBLISHER
Q2-5  "Willemain 2004 ... two-state transition probabilities zero nonzero ...
       jittering"
      -> confirmed the method estimates transition probabilities of a two-state
         (zero / non-zero) Markov model, i.e. it exploits occurrence dependence
         directly, then jitters sampled sizes.                             PUBLISHER
Q2-6  "simulation study intermittent demand generated series varying autocorrelation
       of demand intervals fixed marginal distribution ... factorial experiment"
      -> returned Altay et al. again as the only direct match; no second study
         found that manipulates dependence as an experimental factor.      SNIPPET
```

**Axis 2 verdict.** Prior work manipulating temporal dependence in generated
intermittent demand exists and is singular in the search: Altay et al. (2012). It varies
dependence and measures *forecast and inventory* performance of Croston-family
estimators. It does not vary the representation.

---

## Axis 3 — the Croston / decomposition lineage

```
Q3-1  "Croston 1972 Forecasting and stock control for intermittent demands ORQ 23"
      -> 23(3) 289-303, DOI 10.1057/jors.1972.50.                          PUBLISHER
         (Published in Operational Research Quarterly; Crossref and the publisher
         now index the container as Journal of the Operational Research Society.
         Recorded as a metadata discrepancy, see WARN_FAIL.md LIT-W1.)
Q3-2  "Syntetos Boylan 2005 The accuracy of intermittent demand estimates ... SBA"
      -> IJF 21(2) 303-314, DOI 10.1016/j.ijforecast.2004.10.001;
         SBA = Croston with the (1 - alpha/2) bias correction.             PUBLISHER
Q3-3  "Teunter Syntetos Babai 2011 ... TSB method"
      -> EJOR 214(3) 606-615; updates demand probability every period rather than
         the interval.                                                     PUBLISHER
Q3-4  Crossref lookup for the Teunter DOI
      -> 10.1016/j.ejor.2011.05.018.                                       CROSSREF
```

---

## Axis 4 — direct vs factorized forecasting, and the novelty collision sweep

Synonym sweep actually issued: *direct / single-stage / one-stage / rate*, *factorized /
decomposed / two-stage / hurdle / zero-inflated / two-part / occurrence-size*, plus
*matched budget*, *same backbone*, *inductive bias*, *controlled comparison*, *ablation*.

```
Q4-1  "neural network intermittent demand ... direct prediction versus two-stage hurdle
       zero-inflated decomposition occurrence probability times size"
      -> surfaced Kourentzes (2013), Nathan et al. (2026), Switch-Hurdle (arXiv),
         a dual-CNN paper in MENDEL, and a transformer paper in Ann. Oper. Res.
Q4-2  Switch-Hurdle arXiv:2602.22685 fetched
      -> Muşat & Căbuz, 26 Feb 2026, no peer-reviewed venue stated. Abstract read in
         full. No matched-budget direct-vs-hurdle comparison; no controlled synthetic
         dependence.                                                       FULLTEXT (abstract page)
Q4-3  "Kourentzes 2013 Intermittent demand forecasts with neural networks"
      -> IJPE 143(1) 198-206, DOI 10.1016/j.ijpe.2013.01.009.              PUBLISHER
Q4-4  author's own summary page fetched
      -> two variants named: NN-Dual (size and interval forecast separately,
         Croston-like) and NN-Rate (demand rate forecast directly).        PUBLISHER
Q4-5  the paper PDF was downloaded from Lancaster eprints and text-extracted locally,
      because the collision verdict cannot rest on an abstract.            FULLTEXT
      Read: section 3.1 (dataset), section 3.4 (methods), section 4 (results).
      Findings quoted in novelty_collision_audit.md.
Q4-6  "Primacy of feature engineering over architectural complexity ..." (Sci Rep 2026)
      -> nature.com redirected to an IdP; the PMC mirror PMC12873174 was read
         instead.                                                          FULLTEXT (PMC)
Q4-7  "Türkmen Wang Januschowski Deep Renewal Processes ... peer reviewed"
      -> arXiv:1911.10416 (2019) is a preprint / NeurIPS workshop item; the
         peer-reviewed record is PLOS ONE 16(11) e0259764 (2021).
Q4-8  PLOS ONE article page fetched
      -> author order confirmed; synthetic experiments exist but are illustrative
         (periodic demand, alternating inter-demand times, random arrivals), not a
         factorial over dependence with fixed marginals.                   FULLTEXT (publisher HTML)
Q4-9  "Machine learning algorithms in intermittent demand forecasting: a review"
      -> IJPR, published online 31 Oct 2025.                               PUBLISHER
Q4-10 "controlled experiment inductive bias representation choice sparse count time
       series direct regression versus two-part model matched capacity"
      -> nothing in intermittent demand; only generic deep-forecasting surveys.
Q4-11 "'zero-inflated' OR 'hurdle' deep learning demand forecasting ... single-head
       versus two-head same backbone"
      -> hurdle/zero-inflated heads are common (transit, ridesharing, EV charging,
         vessel traffic), but no controlled single-head vs two-head comparison at a
         matched budget was returned.
```

---

## What was *not* done

```
no citation was written that was not resolved to a DOI
no author, year, venue or page range was inferred from memory
no arXiv-only item is presented as peer-reviewed
no collision verdict rests on an abstract alone -- the two N3 papers were read in full
no Related Work prose, Methods, Results or Discussion text was drafted
no new experiment, training run or TEST scoring
no commit, push or merge
frozen scientific artifacts unmodified
```

---

## Coverage honesty

Search is English-language and web-index-limited. Three specific gaps are declared
rather than papered over:

```
G1  paywalled full texts (IJPE, IJF, EJOR, JORS) were verified at abstract + metadata
    level only, except Kourentzes (2013), whose author-deposited PDF was read in full.
    Altay et al. (2012) therefore carries an abstract-level reading of its DGP.
G2  no systematic database sweep (Scopus / WoS) was run; recall is not quantified.
G3  the 2025-2026 literature is thinly indexed and moves fast; a same-idea preprint
    appearing after this audit cannot be excluded.
```

G1 is the one that matters for a verdict, and it is handled conservatively: Altay et al.
is graded as an overlap (N3) on the strength of its abstract, i.e. graded *against* our
novelty, not for it.
