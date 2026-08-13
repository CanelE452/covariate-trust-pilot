# Literature evidence matrix

Revised 2026-08-12 (third build). Records **evidence only**. Claim decisions live in
`precedent_intersection_map.md` under a separate `novelty_policy` field — see
`evidence_policy_separation.md` for why the two must not be mixed.

Changes in this build:
```
direct_neural_prediction  ->  direct_prediction_arm  +  a separate neural_model column
                              [NAR26] is LightGBM; the old schema forced a non-neural
                              direct arm to be flagged as neural (EC2)
croston_style_dual_neural ->  croston_style_dual_ratio
                              names the RATIO, which is the distinguishing property,
                              not the fact that Kourentzes used a network
occurrence_probability_..._hurdle -> ..._product
                              names the PRODUCT, symmetrically
matched_feature_set           added; it is what [NAR26] actually states
[NAR26] m_param P -> U, m_train Y -> U    the article does not state either (EC3)
```

```
Y = does it   P = partially   N = does not   U = source does not state it
-  = not applicable (no model is trained, so match fields do not arise)

key           dep_manip  marg_ctrl  neural     direct_arm cro_ratio  hurdle_prod d_vs_dec   d_vs_hur   m_feat     m_param    m_train    occ_ax     mag_ax     rep_x_dep  syn2real   grade
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SBC05         N          N          N          N          N          N          N          N          -          -          -          N          N          N          N          N1
KH06          N          N          N          N          N          N          N          N          -          -          -          N          N          N          N          N1
Cro72         N          N          N          N          Y          N          N          N          -          -          -          N          N          N          N          N1
SB05          N          N          N          N          Y          N          N          N          -          -          -          N          N          N          N          N1
TSB11         N          N          N          N          N          Y          N          N          -          -          -          N          N          N          N          N1
WSS04         N          N          N          N          N          N          N          N          -          -          -          P          N          N          N          N2
ALR12         Y          U          N          N          N          N          N          N          -          -          -          Y          Y          N          N          N3
Kou13         N          N          Y          Y          Y          N          Y          N          N          N          P          N          N          N          N          N3
TJWC21        N          P          Y          Y          N          P          N          N          N          N          P          N          N          N          N          N2
NAR26         N          N          N          Y          N          Y          Y          Y          Y          U          U          N          N          N          N          N3
GDTP25        N          N          N          N          N          N          N          N          -          -          -          N          N          N          N          N1
MC26          N          N          Y          N          N          Y          N          N          N          N          N          N          N          N          N          N1
THIS PAPER    Y          Y          Y          Y          N          Y          Y          Y          Y          Y          Y          Y          Y          Y          Y          --

dep_manip    temporal dependence manipulated as an experimental factor
marg_ctrl    marginal / intermittency characteristics controlled while it varies
neural       the model is a neural network
direct_arm   an arm that predicts the conditional mean or demand rate directly
cro_ratio    Croston-style: size and interval predicted, combined as a RATIO
hurdle_prod  occurrence probability and positive magnitude, combined as a PRODUCT
d_vs_dec     a direct arm compared against any decomposed arm
d_vs_hur     a direct arm compared against the PRODUCT form specifically
m_feat       matched feature set, as STATED by the source
m_param      matched parameter budget, as STATED by the source
m_train      matched training protocol, as STATED by the source
occ_ax       occurrence dependence varied as its own axis
mag_ax       magnitude dependence varied as its own axis
rep_x_dep    representation choice CROSSED with dependence structure
syn2real     the controlled pattern followed to a real-data transfer boundary
```

## Notes

```
SBC05        ADI / CV^2 categorization scheme
KH06         correction to the SBC categorization boundary
Cro72        origin of the size/interval ratio decomposition; classical, non-neural
SB05         SBA (1 - alpha/2) bias correction on the ratio form; classical baseline
TSB11        TSB updates the occurrence PROBABILITY every period; closest classical ancestor of the product form, but not a representation comparison
WSS04        two-state Markov occurrence inside a bootstrap; exploits occurrence dependence, does not manipulate it
ALR12        manipulates interval AC, size AC and cross-correlation in generated demand; compares Croston-family ESTIMATORS inside one already-factorized form. marg_ctrl = U: full text unobtainable, six routes failed (LIT-W3)
Kou13        NN-Rate = single linear output, demand rate. NN-Dual = two linear outputs, size and interval, combined as a RATIO then de-biased by a fitted coefficient. One simulated population; each arm reported at its own best (I,H)
TJWC21       deep renewal processes; occurrence and size modelled jointly; illustrative synthetic patterns incl. alternating inter-demand times; no matched contrast
NAR26        LightGBM, NOT neural. Single-stage regressor vs two-stage LightGBM classifier x Tweedie regressor = the PRODUCT form, at an identical feature set. m_param / m_train = U: NAR-E, NAR-F, NAR-G all NOT STATED (LIT-W-NAR26). No dependence factor and no regime breakdown (NAR-H verified negative)
GDTP25       review of ML for intermittent demand; no located statement that the question is settled
MC26         PREPRINT, not peer-reviewed; MoE encoder with a hurdle decoder; no controlled comparison
THIS PAPER   occurrence and magnitude dependence on separate axes; 5,856 = 5,856 parameters, one backbone, one trainer, one 30-epoch budget, one evaluation target
```

## Reading the matrix

**`d_vs_hur` is now Y for a prior row.** [NAR26] compares a direct single-stage arm
against a two-stage occurrence-probability × magnitude arm at an identical feature set.
That precedent is real and is conceded without qualification (component 6a = PRIOR). Any
earlier reading in which the product-form comparison was ours is withdrawn.

**No prior row states a match on capacity or training.** [NAR26] is `U` on both — not
verified absent, simply unstated. [Kou13] is `N` on parameters (two output nodes against
one, plus per-arm configuration selection) and `P` on training (shared trainer, unmatched
capacity). So a matched comparison is `NOT_FOUND_IN_AUDIT`, and because the nearest
neighbour is `U` rather than `N`, it carries `CLAIM_ONLY_IN_CONJUNCTION`.

**`dep_manip` and `d_vs_dec` still never co-occur in a prior row**, and `rep_x_dep`
is `N` for every prior row. That column is where the contribution sits.

**`marg_ctrl` for [ALR12] is `U` and stays `U`.** It is not promoted to `Y` because we
could not read the paper, and it is not read as `N` because that would favour us. The
corresponding novelty component is excluded by policy, not by evidence.
