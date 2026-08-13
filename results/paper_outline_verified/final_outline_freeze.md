# Final outline freeze

One page. Everything below is settled; anything later that contradicts it is a defect.
Source of truth for claims: `../paper_synthesis_verified/claim_ledger_frozen.md`.

---

## Section hierarchy

```
1  Introduction
2  Related Work
3  Problem Setup and Forecasting Formulations
   3.1  Limits of marginal intermittency summaries
   3.2  Direct and factorized forecasting
   3.3  Relative finite-sample comparison
4  Controlled Synthetic Study
   4.1  DGP and marginal control
   4.2  Experimental control and fairness
   4.3  Stage 1: fixed-marginal factorial
   4.4  Component-attribution diagnostic
   4.5  Stage 2: stationary dependence sweep
   4.6  Occurrence-strength versus magnitude-direction asymmetry
   4.7  The direct-favourable synthetic boundary
5  Empirical Validation
   5.1  Datasets and evaluation protocol
   5.2  Overall Point/Hurdle comparison
   5.3  H1 empirical analogue
   5.4  H2 frozen direct-favourable selector
   5.5  Overlap-adjusted mechanism boundary
   5.6  H3 / sparsity-transfer boundary
   5.7  Adaptive-use boundary: routing instability      [0.5-1 page]
6  Discussion
   6.1  Finite-sample interpretation
   6.2  Why occurrence and magnitude structure differ
   6.3  Synthetic-to-real entanglement
   6.4  What the routing failures imply
   6.5  Limitations
7  Conclusion
```

## Figure roles

```
Fig 1   problem and controlled design.  No result leakage: the three rho levels are
        design levels, the two axes are drawn symmetrically, and neither |rho_I| nor
        the sign of rho_M is hinted at.
Fig 2   controlled synthetic discovery.
        2A  18 cells, two 3x3 panels (d=4, d=8).  Zero-centred SYMMETRIC diverging
            scale over [-23, +23], one scale for both panels.  G printed in every
            cell.  A marker, not colour, indicates CI excludes zero.
        2B  occurrence marginal: +12.10 / +3.57 / +16.47 pp  -> U
        2C  magnitude marginal: +14.10 / +11.94 / +6.11 pp   -> monotone
        2B and 2C share one y-axis.
Fig 3   empirical transfer and boundary.
        3A  H1 analogue, M5 and Favorita, with the regime forest inset
        3B  H2 selector on the independent M5 population, three seeds
        3C  covariate balance and the overlap-adjusted association crossing zero
No routing panel appears in any main figure.
```

## Table roles

```
T1  controlled design and fairness            5,856 = 5,856 parameters
T2  Stage 1 factorial contrasts               in G percentage points
T3  core empirical datasets and protocol      M5 and Favorita ONLY
T4  empirical results, classical baselines, hypothesis summary
A0  appendix: FreshRetailNet-LT and UCI protocol, incl. AVAILABILITY_UNKNOWN
```

## Wording decisions

```
metric              G = 100 (1 - RMSE_Hurdle / RMSE_Point);  raw delta never crosses
                    studies; the three artifact conventions live in Appendix K
stages              Stage 1 = fixed-marginal 2x2x2 factorial
                    Stage 2 = stationary rho sweep
                    no Stage 3, no Stage 4; the mechanism material is the
                    "Stage 1 component-attribution diagnostic"
asymmetry           "primarily associated with dependence strength / direction"
                    never causal, never "we prove an asymmetric mechanism"
scope on the        "within the eighteen-cell Stage 2 grid, only one cell shows
direct-favourable    statistically clear superiority for direct prediction"
cell                never "exactly one counterexample" unqualified
H2                  two claims: predictive selector transfer supported;
                    isolated mechanism NOT_REPLICATED.  Never "mechanism replicated".
H1                  EMPIRICAL_ANALOGUE, never "replicates"
Stage 1             never credited with identifying a direct-favourable regime;
                    C08 is -3.01 with CI [-6.79, +0.23]
mechanism           COMPONENT-ATTRIBUTION DIAGNOSTIC SUPPORT, synthetic only
transport audit     PRIMARY SOURCE PAYLOAD INTEGRITY PASS
related work        no novelty claim over the correlation literature
overall             no universal Hurdle claim; the question is when factorization
                    helps or hurts
```

## Contributions

```
C1  Controlled characterization of the finite-sample relative inductive bias of
    direct and factorized forecasting under temporal occurrence and magnitude
    dependence.

C2  Empirical transfer with an explicit boundary: an occurrence-dependence analogue
    and a synthetic-derived direct-favourable selector show transfer, while the
    isolated mechanism does not survive overlap adjustment.

C3  Adaptive-use boundary: expert complementarity exists, but learned routing does
    not transfer robustly across domains and time.  Compressed; C3 is support, not a
    second paper.
```

## Remaining risks

```
OPEN                    R1 single backbone;  R8 occurrence-head skill on real data
MITIGATED, NOT          R3 Stage 1 conditional verdict;  R4 Stage 1 vs Stage 2
  ELIMINATED               parameterization
MITIGATED               R2 synthetic simplicity
ACKNOWLEDGED_BOUNDARY   R5 construct mismatch;  R6 selector vs mechanism;
                        R7 H3 non-replication;  R9 routing instability;
                        R10 absolute performance
```

Nothing is eliminated. R1 is the largest residual exposure and is answered by scope,
not by evidence.

## Work status

```
MUST_HAVE      NONE
NICE_TO_HAVE   second backbone; per-observation occurrence diagnostic on real data;
               H3 re-tested at ADI 3-5 vs >= 8
DO_NOT_RUN     routing development; synthetic rerun; new SOTA expansion
```

## What changed in this freeze

```
1  original Sections 3 and 4 merged into Section 3, now with 3.1/3.2/3.3
2  routing demoted from an independent Section 6 to subsection 5.7, 0.5-1 page
3  Figure 2A colour fixed to a zero-centred SYMMETRIC diverging scale shared by both
   panels; this reverses the earlier draft's asymmetric-scale advice, which was wrong
   -- with a symmetric scale the single negative cell is the only one in the opposite
   hue and separates on its own
4  significance moved out of colour and into a separate marker
5  Table 3 restricted to M5 and Favorita; FreshRetailNet and UCI moved to an appendix
   dataset table so no reader mistakes them for core validation data
6  empirical wording downgraded: "the direction reproduces" -> "an empirical analogue
   appears"; "exactly one counterexample" -> scope-qualified
7  reviewer risks reclassified as OPEN / MITIGATED / ACKNOWLEDGED_BOUNDARY; the earlier
   claim that several attacks were effectively neutral was withdrawn
8  Discussion renumbered to 6 with a new 6.4 on what the routing failures imply
```
