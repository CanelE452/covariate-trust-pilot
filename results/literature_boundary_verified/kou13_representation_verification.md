# Kou13 — representation verification

Kourentzes, N. (2013). Intermittent demand forecasts with neural networks.
*International Journal of Production Economics* 143(1), 198-206.
doi:10.1016/j.ijpe.2013.01.009

Source read: author-deposited full text (Lancaster eprints 61613), 28 pages, extracted
locally. Sections 2 (model formulation), 3.1 (dataset), 3.4 (methods), 4 (results).
**Status: FULL-TEXT VERIFIED.**

The previous audit called NN-Dual "decomposed" and left it there. That was too coarse,
and it risked letting a reader equate it with this paper's Hurdle. This file fixes that.

---

## KOU-A — what NN-Rate outputs

```
inputs    lags of z_t (non-zero demand) and x_t (inter-demand interval)
output    ONE linear node: the demand rate Y'_t, directly
```

Verbatim: *"A more elegant way to avoid biasing the forecasts is to model the network to
output the required demand-rate directly, instead of introducing the subsequent division
step. Similar to NN-Dual this model uses lags from both z_t and x_t to output directly
Y'_t. The division and the required de-biasing is left to the network to approximate
from the data."*

Figure 2 caption: *"A single linear output provides the demand rate forecast."*

## KOU-B — what NN-Dual outputs

```
inputs    lags of z_t and x_t   (one shared network, not two)
outputs   TWO linear nodes: z'_t (non-zero demand) and x'_t (inter-demand interval)
combine   Y'_t = z'_t / x'_t          -- Croston's division, equation (1)
fix-up    a data-driven de-biasing coefficient c from the regression z_t/x_t = c Y_t;
          out-of-sample forecasts are multiplied by c
```

Figure 1 caption: *"Two linear output nodes provide the demand and interval forecasts."*

Verbatim on why the division is a problem: *"The resulting z'_t and x'_t have to be
divided as in (1) to provide the forecasts, thus the model suffers from the inversion
bias discussed by Syntetos and Boylan (2001)."*

---

## The distinction that matters

```
                       Kou13 NN-Dual                  this paper's Hurdle
--------------------------------------------------------------------------------
decomposition          size DIVIDED BY interval       occurrence probability
                       z / x                          TIMES positive magnitude
                                                      P(Y>0 | h) * E[Y | Y>0, h]
combination rule       ratio                          product
occurrence quantity    inter-demand interval,          P(Y>0), a probability with a
                       a positive real regressed       binary target
                       under a regression loss
indexing               event-indexed: the model sees   period-indexed: every period,
                       the sequence of non-zero        including zeros, is an
                       demands and the sequence of     observation
                       intervals
bias handling          inversion bias is present by    no inversion; the product is
                       construction and removed        an unbiased composition by
                       afterwards by a fitted          construction
                       coefficient c
lineage                neural Croston                  neural hurdle / two-part
```

`1/x` estimates an occurrence rate, so the two are **related** parameterizations of the
same conditional mean, and this file does not claim otherwise. They are not
**interchangeable**: the estimand of the occurrence branch differs (an interval versus a
probability), the loss differs, the indexing differs, and one carries an inversion bias
the other does not. A finite-sample comparison is exactly the kind of comparison in
which that difference can show up.

---

## KOU-C — are the two representations actually compared?

**Yes.** Both appear in every results table, alongside SES, MA, Croston variants,
CR-Naive and NN-GSM. The neural direct-versus-decomposed comparison is therefore
**precedent, and this paper concedes it.**

Reported outcome, for the record: NN-Dual ranks better than NN-Rate on the accuracy
ranks, while NN-Rate reaches the higher realised service levels (93.2% / 94.6% / 95.5% /
96.8% against NN-Dual's 88.9% / 91.0% / 92.4% / 94.6% at the four targets). The paper's
own conclusion is that NN-Rate is the best performing model once inventory metrics are
used.

---

## KOU-D, KOU-E, KOU-F, KOU-G — the budget question

```
KOU-D  configuration search    input lags I in {1,2,3} and hidden nodes H in {1,2,3};
                               45 model-parameter combinations simulated in total
KOU-G  separate tuning         YES.  Verbatim: "only the best performing parameters of
                               each model will be presented in section 4. The criteria
                               for selecting the model is minimum in-sample mean MAE
                               rank."  Each formulation is reported at ITS OWN best
                               configuration.
KOU-E  matched parameter       NO, on two independent grounds:
       count                   (i) the reported configurations are selected per model
                                   and need not coincide;
                               (ii) even at identical (I,H) the two differ by
                                   construction -- NN-Dual has two output nodes and
                                   NN-Rate one, so the output layer alone differs by
                                   H + 1 weights.
KOU-F  common training         PARTIAL.  The trainer is shared -- Levenberg-Marquardt,
       protocol                regularisation gamma = 0.9, max 1000 epochs, 5 random
                               initialisations with a median ensemble, inputs scaled to
                               [-0.8, 0.8], "all other network settings are kept
                               constant".  What is NOT shared is capacity, and the
                               reported configuration is chosen per model.
```

For contrast, this paper's design fixes 5,856 parameters for both arms, one trainer and
a 30-epoch budget (`claim_ledger_frozen.md` section 1, `T1`).

---

## KOU-H — is dependence an experimental factor?

**No.** Section 3.1: *"a large scale simulation of 1000 items is designed. The dataset
used by Syntetos and Boylan (2005) is used to identify realistic parameters to simulate
intermittent demand time series."* Empirical distributions of non-zero demand and of
inter-demand intervals are estimated and new series constructed from them. **One
population. No factorial, no sweep, no dependence level.**

The dependence Kou13 is about is also a **different** dependence: contemporaneous
size-interval association, not serial dependence. Verbatim: *"the distribution of
non-zero demand changes for different inter-demand intervals; hence they are not
independent as assumed by Croston's method"*, fitted as a second-order polynomial
selected by BIC. That is a cross-channel relation at the same event. This paper's axes
are serial — whether gaps cluster or alternate over time, and whether large orders
follow large ones.

---

## Wording guard carried out of this file

```
DO NOT WRITE   "prior work compares the same two representations"
DO NOT WRITE   "a prior direct-versus-hurdle comparison exists"
DO NOT WRITE   "NN-Dual is a hurdle model"

WRITE          "neural work compares a directly predicted demand rate against a
                Croston-style network that forecasts size and interval and divides
                them"
WRITE          the concession plainly: a neural direct-versus-decomposed precedent
                exists.
```

Registered as **LIT-W-KOU13** in `WARN_FAIL.md`. The distinction must survive into the
manuscript; it is the difference between conceding a precedent and conceding the paper.
