# Core reference list

Twelve records. Every DOI in the peer-reviewed block was resolved through the Crossref
REST API; the raw responses are in `crossref_verified.json` (11 lookups, 11 OK).

Citation keys are the keys used in `introduction_v4.md`.

---

## Peer-reviewed — verified against Crossref

```
[SBC05]   Syntetos, A.A., Boylan, J.E. and Croston, J.D. (2005).
          On the categorization of demand patterns.
          Journal of the Operational Research Society 56(5), 495-503.
          doi:10.1057/palgrave.jors.2601841

[KH06]    Kostenko, A.V. and Hyndman, R.J. (2006).
          A note on the categorization of demand patterns.
          Journal of the Operational Research Society 57(10), 1256-1257.
          doi:10.1057/palgrave.jors.2602211

[Cro72]   Croston, J.D. (1972).
          Forecasting and stock control for intermittent demands.
          Operational Research Quarterly 23(3), 289-303.
          doi:10.1057/jors.1972.50                              [see LIT-W1]

[SB05]    Syntetos, A.A. and Boylan, J.E. (2005).
          The accuracy of intermittent demand estimates.
          International Journal of Forecasting 21(2), 303-314.
          doi:10.1016/j.ijforecast.2004.10.001

[TSB11]   Teunter, R.H., Syntetos, A.A. and Zied Babai, M. (2011).
          Intermittent demand: linking forecasting to inventory obsolescence.
          European Journal of Operational Research 214(3), 606-615.
          doi:10.1016/j.ejor.2011.05.018

[WSS04]   Willemain, T.R., Smart, C.N. and Schwarz, H.F. (2004).
          A new approach to forecasting intermittent demand for service parts
          inventories.
          International Journal of Forecasting 20(3), 375-387.
          doi:10.1016/S0169-2070(03)00013-X

[ALR12]   Altay, N., Litteral, L.A. and Rudisill, F. (2012).
          Effects of correlation on intermittent demand forecasting and stock control.
          International Journal of Production Economics 135(1), 275-283.
          doi:10.1016/j.ijpe.2011.08.002

[Kou13]   Kourentzes, N. (2013).
          Intermittent demand forecasts with neural networks.
          International Journal of Production Economics 143(1), 198-206.
          doi:10.1016/j.ijpe.2013.01.009

[TJWC21]  Türkmen, A.C., Januschowski, T., Wang, Y. and Cemgil, A.T. (2021).
          Forecasting intermittent and sparse time series: a unified probabilistic
          framework via deep renewal processes.
          PLOS ONE 16(11), e0259764.
          doi:10.1371/journal.pone.0259764

[NAR26]   Nathan, B.S., Aravinth, P.M., Reddy, B.V.S., Sastry, C.C., Salunkhe, S.
          and Cep, R. (2026).
          Primacy of feature engineering over architectural complexity for
          intermittent demand forecasting.
          Scientific Reports 16, 4792.
          doi:10.1038/s41598-026-35197-y

[GDTP25]  Giannopoulos, P.G., Dasaklis, T.K., Tsantilis, I. and Patsakis, C. (2025).
          Machine learning algorithms in intermittent demand forecasting: a review.
          International Journal of Production Research, published online 31 Oct 2025.
          doi:10.1080/00207543.2025.2578701                     [see LIT-W2]
```

## Preprint — NOT peer-reviewed, must be labelled as such wherever used

```
[MC26]*   Muşat, F. and Căbuz, S. (2026).
          Switch-Hurdle: a MoE encoder with AR hurdle decoder for intermittent
          demand forecasting.
          arXiv:2602.22685 [submitted 26 Feb 2026].
          No peer-reviewed venue stated.
```

`*` marks the preprint. It is cited in the audit as evidence that hurdle-style neural
decoders are current practice. It is **not** used to support any claim in the
Introduction, and it is not counted in any novelty verdict.

---

## Why each is in the list

```
[SBC05]   the categorization scheme itself -- resolves CITATION NEEDED 1
[KH06]    corrects the SBC boundary; citing SBC without it is a known omission
[Cro72]   the origin of the factorized representation the paper studies
[SB05]    SBA; the bias correction that made the factorized estimator usable
[TSB11]   TSB; the probability-updating variant, and our classical baseline family
[WSS04]   an estimator that exploits occurrence dependence directly (two-state
          Markov transitions) -- the concrete instance behind "estimators have been
          proposed that exploit them"
[ALR12]   the only located study that manipulates interval autocorrelation, size
          autocorrelation and size-interval cross-correlation as experimental factors
          in generated intermittent demand -- resolves CITATION NEEDED 2, and is the
          strongest constraint on the Axis 2 novelty claim
[Kou13]   compares a directly-forecast demand rate against a decomposed
          size-and-interval network on simulated intermittent series -- the strongest
          constraint on the Axis 4 novelty claim
[TJWC21]  the peer-reviewed deep renewal-process record; unifies occurrence and size
          in a neural model, with illustrative synthetic patterns
[NAR26]   a recent single-stage vs two-stage product-form comparison at an identical
          feature set, on real data, reported in aggregate.  Both arms are LightGBM;
          the article does NOT state a capacity, training-budget or per-formulation
          tuning match (LIT-W-NAR26)
[GDTP25]  the current review; used to check that no located survey already states our
          question as answered
[MC26]*   current hurdle-decoder practice; context only
```

---

## Deliberately excluded, with reasons

```
Kim, Haddock & Willemain (1993), the binary bootstrap
          surfaced only as a secondary reference inside another paper's text; its own
          record was not verified, so it is not listed.
Gutierrez, Solis & Mukhopadhyay (2008), NN benchmark
          used as a benchmark inside [Kou13] and read only through it; not
          independently verified, so it is not listed.
dual-CNN (MENDEL) and transformer (Ann. Oper. Res.) intermittent papers
          seen at snippet level only.  They are near-neighbours of [MC26]* and would
          not change any verdict, so they were not pursued to full text; this is
          recorded as a coverage limit rather than a clearance.
Salinas et al., DeepAR
          general probabilistic forecasting, not an intermittent-demand
          representation study; no claim in this paper depends on it.
```
