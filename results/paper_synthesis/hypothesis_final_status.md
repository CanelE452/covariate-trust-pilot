# Final status of the pre-registered hypotheses

Read from artifacts on 2026-08-09. Operational verdict labels stored inside the artifacts are
quoted as such and are **not** treated as scientific conclusions.

---

## H1 — |rho_interval| up, relative Hurdle advantage up

**Status: SUPPORTED_WITH_BOUNDARY**

```
source                              M5                        Favorita
──────────────────────────────────────────────────────────────────────────────────
Stage A spearman            +0.1065 [+0.0437,+0.1652]*  +0.0789 [+0.0205,+0.1405]*
threshold 15 / 20 / 30      +0.1064 / +0.1064 / +0.1057 +0.0799 / +0.0789 / +0.0767
posthoc raw                 +0.1065 [+0.0437,+0.1653]*  +0.0789 [+0.0204,+0.1405]*
posthoc relative            +0.1231 [+0.0697,+0.1775]*  +0.1178 [+0.0603,+0.1727]*
posthoc scaled              +0.0966 [+0.0390,+0.1536]*  +0.0928 [+0.0387,+0.1473]*
adjusted std. coefficient   +0.0317 (CI includes 0)     +0.0168 (CI includes 0)
```

```
SBC regime (M5, n=300 each)   raw                    relative               scaled
──────────────────────────────────────────────────────────────────────────────────────
smooth                 +0.129 [+0.016,+0.246]*  +0.087 [-0.032,+0.208]  +0.043 [-0.068,+0.161]
erratic                +0.081 [-0.032,+0.195]   +0.088 [-0.025,+0.198]  +0.078 [-0.038,+0.192]
intermittent           +0.107 [-0.009,+0.215]   +0.153 [+0.052,+0.261]* +0.123 [+0.013,+0.227]*
lumpy                  +0.014 [-0.106,+0.130]   +0.032 [-0.079,+0.146]  +0.028 [-0.089,+0.144]
```

Artifact verdict `H1_TARGET_REGIME_DIRECTION_SUPPORTED`.

**What may be claimed.** The marginal association reproduces in two independent public
datasets, is insensitive to the eligibility threshold, holds on three error scales, and is
present with intervals clear of zero in the intermittent regime on the relative and scaled
metrics.

**What may not.** Causality; that it survives adjustment for sparsity and scale; that it holds
in the lumpy regime; that it is a population estimate — the Stage A sample is balanced at 300
per SBC class while the M5 full pool is 23,053 / 5,942 / 984 / 496.

---

## H2 — the Point-favorable rule

**Status: SUPPORTED_WITH_BOUNDARY as a predictive selector; NOT_REPLICATED as a mechanism.**
These must be reported as two separate results.

```
analysis                                    n            effect                    CI excl 0
───────────────────────────────────────────────────────────────────────────────────────────
screen, M5                                  163      -0.0303 [-0.0954,+0.0588]        no
screen, Favorita                            151      -0.0224 [-0.1618,+0.1335]        no
independent M5, frozen rule            675 + 5018    -0.0230 [-0.0294,-0.0163]       yes
  point win rate difference                          +11.87 pp [+7.85,+15.81]        yes
seed 0 / 1 / 2                         675 + 5018    -0.0230 / -0.0211 / -0.0277     yes (3/3)
Favorita independent, M5-frozen rule    792 + 4613    CI excludes zero                yes
  point win rate difference                          +3.90 pp
Favorita Stage A pool                    18 + 117    CI includes zero                 no
  point win rate difference                          -14.53 pp   (sign disagrees)
overlap-adjusted association                5693    +0.0032 [-0.0033,+0.0094]         no
1:2 matching                            675 + 1350    worst SMD 0.614 on log scale   FAILED
```

**Selector reading.** Confirmed on M5 across three seeds and on an independent population;
partial transfer to unseen Favorita.

**Mechanism reading.** Once ADI, CV², scale and |rho_interval| are balanced, the association
is +0.003 with a CI containing zero, and matching fails because candidates and controls differ
by 0.61 SMD in scale. The rule selects series that differ in scale, so nothing here identifies
magnitude persistence as the operative cause.

---

## H3 — sparsity strengthens the occurrence effect

**Status: NOT_REPLICATED**

```
                    corr_high_ADI   corr_low_ADI   difference             CI excl 0
──────────────────────────────────────────────────────────────────────────────────
M5                     +0.0754        +0.1060       -0.0305 [-0.1418,+0.0912]   no
Favorita               +0.0234        +0.0661       -0.0428 [-0.1587,+0.0704]   no
```

Both signs are opposite to the prediction. The split used the ADI median (M5 1.304), while the
synthetic contrast was ADI 4 versus 8. Groups matching the synthetic contrast exist
(M5: 127 series at ADI 3–5, 52 at ADI ≥ 8; Favorita: 84 and 45) but were never used as a
primary test, so the failure is a failure **at the pre-registered split**, not a demonstration
that no interaction exists. H3 is excluded from the main contributions and reported as a
non-replication with that caveat.

---

## Occurrence gate mechanism

**Status: REJECTED as a real-data mechanism claim.**

```
                    BSS vs constant rate        per-series BSS > 0
────────────────────────────────────────────────────────────────────
M5              -0.0084 [-0.0411,+0.0241]            36.8%
Favorita        -0.0908 [-0.1401,-0.0438]*           30.8%
```

The fitted occurrence head does not beat a per-series constant rate; on Favorita it is
significantly worse. Whatever produces the H1 association in these datasets, it is not a
demonstrated improvement in occurrence prediction. The mechanism story therefore stays inside
the controlled study and is explicitly not carried into the real-data section.

---

## Routing

**Status: NOT_REPLICATED as a method; CONFIRMED as a boundary result.**

Binding state at the end of the development chain:
`HANDCRAFTED_FEATURE_GATE_STOP`, `RAW_SEQUENCE_GATE_STOP`, `ROUTING_MODEL_DEVELOPMENT_STOP`.

Two statements must stay separate:

- **Raw temporal history contains routing information the summaries discard** — evidence
  exists (FreshRetailNet: −0.506% → +2.648%, CI clear of zero, better on 3/3 folds).
- **Routing generalizes reliably across domains** — no evidence. The same model is −193.9%
  on UCI, and the earlier handcrafted gate was −2.43% on the first external dataset.
