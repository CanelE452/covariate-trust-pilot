# Missing requirements for the full protocol

What the cached artifacts do **not** provide, and what each gap would cost. Nothing here was
run.

---

## 1. Inner-validation predictions — blocks strict train-only `S_occ` / `S_mag`

```
required by   Phase D1/D2: component forecastability from inner rolling-origin windows
              INSIDE outer train
available     nothing.  A recursive search for valid/inner/val_ artifacts under
              results/external_validity_screen/ returned no prediction files.
              Only outer-test origins are cached.
consequence   Track T2 substitutes the FIRST outer origin for the inner window.  That is
              causally valid for predicting origins 2 and 3 but is NOT train-only, and it
              costs one third of the panel as target rows.
```

**Cost to close it properly.** Checkpoints already exist (`rule_replication/models/{point,hurdle}.pt`,
`favorita_transfer/models/`), so this is inference, not training:

```
scope         3 inner origins x 11,098 series x 28 steps, both arms
runs          2 datasets x 2 arms x 3 inner origins = 12 scoring passes
compute       inference only on cached checkpoints; no gradient steps
storage       ~2.8M rows in the existing 11-column schema, roughly 2-3x the current
              932k-row parquet footprint, order 100-200 MB
smoke first   100 series x 1 inner origin x both arms, to measure wall-clock before
              committing to the full pass
caveat        the M5 checkpoint is the rule_replication model and the Favorita checkpoint
              is the transfer model; whether both were trained under the identical
              protocol the panel assumes must be confirmed from
              seed_robustness/model_provenance.json before any scoring run
```

**Not started.** Per the execution rules this needs approval first.

## 2. FreshRetailNet-LT and UCI paired panels — blocks leave-one-dataset-out

```
required by   Phase F1 four-dataset LODO, and Gate 2
available     routing-gain artifacts only; no paired Point/Hurdle per-series RMSE in this
              schema, no origin-level predictions
consequence   the only external protocol available today is the 2-domain M5 <-> Favorita
              pilot, which is what was run.  It is reported as a pilot, never as external
              validation.
cost          building these two panels means training both arms on both datasets under the
              same protocol - a real run, not a re-score, and materially larger than item 1
```

## 3. Natural distribution for Stage A

```
available     the independent populations used here ARE the natural eligible pools
              (M5 5,693 of the eligible catalogue; Favorita 5,405)
not available a natural-distribution version of the original 2,400-series Stage A table
consequence   none for this analysis; the panel does not use the balanced sample.  It does
              mean the published Stage A policy numbers and these are not interchangeable.
```

## 4. Dense rolling origins

```
prereg note   splits.single_origin_note: "a SCREEN uses one pre-fixed origin; a confirmatory
              run would need every rolling origin"
available     three origins per series, stride 28
consequence   the origin-level oracle is estimated from three points per series, which is
              why series-level winner stability (18-26%) is measured with wide uncertainty
```
