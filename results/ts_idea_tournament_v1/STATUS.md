FINAL TOPIC RECOMMENDATION: RECOMMEND_CHARACTERIZATION_ONLY

TRACK X: NO_PHENOMENON
TRACK G: G_CHARACTERIZATION_ONLY
TRACK F: F_SELECTION_CHARACTERIZATION
TRACK V: V_NO_STRONG_SELECTION_INSTABILITY

| Track | Phenomenon | Simple baseline solves | New intervention | Dataset consistency | Bootstrap 95% CI | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| TRACK X | X_NO_SPILLOVER | no | no | same sign | ETTm1[-1.8255, +0.7527], Weather[-12.3233, +25.3954] | NO_PHENOMENON |
| TRACK G | G_PHENOMENON_GO | no | no | conflicting | ETTm1[-0.0649, +0.0044], Weather[-0.0253, +0.0399] | CHARACTERIZATION_ONLY |
| TRACK F | F_SELECTION_CONFOUNDING_PRESENT | no | no | conflicting | ETTm1[-0.0618, -0.0318], Weather[+0.0378, +0.0634] | CHARACTERIZATION_ONLY |

## Result roles

- `[CONFIRMATORY SCREEN]` G, F
- `[DIAGNOSTIC]` X
- `[BLOCKED]` none

## Run metadata

- runtime_tier: FULL
- projected_gpu_hours: 2.54
- peak_process_tree_rss_gb: 1.784
- peak_gpu_used_gb: 1.7265625
- peak_system_memory_pct: 56.5
- finalize_wall_s: 2.6
