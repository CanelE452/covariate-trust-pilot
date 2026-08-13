# Synthetic source transfer manifest

**Status: `SYNTHETIC_SOURCE_TRANSFER_REQUIRED`**

The controlled synthetic study is not on this machine. Nothing in
`results/paper_synthesis/` has been modified, and no synthetic run was created to
substitute for the missing artifacts.

---

## 1. What was searched, and how

A filesystem search rather than an assumption about the recorded path.

```
search                                                        result
──────────────────────────────────────────────────────────────────────────────────
find /c/Users /d /e /f /g -maxdepth 6 -iname "*m5dataset*"    3 history copies in this
                                                              repo + 1 Claude session
                                                              directory; no repository
find for *.bundle, *m5*.tar*, *m5*.zip under /e and           0 hits
  /c/Users/User
experiments/temporal_dependence/                              absent
experiments/decomposition_when_helps/                         absent
```

Independent corroboration: a separate session recorded the same search on 2026-08-08 in
`_docs/history/2026-08-08.md` and reached the same result — `m5dataset`,
`temporal_dependence` and `decomposition_when_helps` returned **0 hits**, and
`experiments/unified_temporal_27_v3/` has code only, with no `results/`, `data/` or `runs/`.

The only local trace of the source project is a Claude Code project directory,
`C:/Users/User/.claude/projects/-home-minjae-Documents-github-m5dataset/`, which contains
three memory notes and **no artifacts**. Those notes are about the earlier RCOI
regime-to-objective work, not about the Point/Hurdle factorization study, and they are not
usable as a source of truth for anything in this manifest.

---

## 2. Source location, from the local record

```
repository        ~/Documents/github/m5dataset
host              the Linux machine referenced throughout the migrated history
                  (`/home/minjae/...`, see results/report.md and
                  _docs/history/2026-08-07-migration.md)
git remotes       none recorded
last commit       2026-06-25 (per _docs/history/2026-08-07-migration.md)
sibling repo      ~/Documents/github/timeseries  == this project, which is why the
                  2026-08-01..05 history here says "today's work happened in the
                  other repository"
```

The migration on 2026-08-07 deliberately copied only the runtime import closure of the
external-validity screen — 21 python files, two processed parquet files, three raw CSVs and
17 result artifacts. The synthetic study was outside that closure and was therefore left
behind.

---

## 3. What is already here, and what it is not

This matters: part of the C1 *method* is local, but none of the C1 *results* are, and the
two local synthetic packages are **not** the paper DGP.

```
local package / file                              sha256(16)         what it is
─────────────────────────────────────────────────────────────────────────────────────────
experiments/unified_temporal_27_v3/scenarios.py   77c442d28c99439e   27-scenario catalogue,
                                                                     12 A + 9 B + 6 C.
                                                                     Group A is a matched
                                                                     pair design: structure
                                                                     {occurrence, magnitude,
                                                                     both} x strength
                                                                     {weak, strong} x variant
                                                                     {iid, non_iid}, where the
                                                                     iid arm is a split-local
                                                                     uniform random reorder of
                                                                     the structured pool -- a
                                                                     genuine marginal-preserving
                                                                     control.
experiments/unified_temporal_27_v3/config.py      6700f08593d6b08a   n_series 300, length 512,
                                                                     lookback 96, horizon 24,
                                                                     train_end 358, val_end 409,
                                                                     data_seed 42, train_seed 0,
                                                                     rho weak 0.30 / strong 0.70,
                                                                     p0 0.25, positive mean 10,
                                                                     CV^2 0.49.
experiments/unified_temporal_27_v3/
  conditional_targets.py                          35cad343c94d8779   markov_transition(rho) and
                                                                     the closed-form conditional
                                                                     target used as the oracle.
experiments/om_factorization_killtest/prereg.py   4ea4ca163ff1bb72   4-cell kill test.  Its own
                                                                     DGP block says verbatim
                                                                     "exploratory kill-test DGP,
                                                                     not the paper DGP".
                                                                     PRIMARY_METRIC =
                                                                     "rmse_mean_truth".
experiments/om_factorization_killtest/models.py   2fb6fdfe6029ce03   PointDLinearParamMatched and
                                                                     FactorizedDLinear.
```

**Why neither is the paper DGP.** The migrated audit
(`_docs/history/m5dataset_2026-08-07.md`, line 71) records the synthetic study's split as
`lookback 96, horizon 24, length 576, train/val/test = 384/480/576`. The local
`unified_temporal_27_v3` config is length 512 with 358/409, and the kill test disclaims itself
in its own preregistration. So a third configuration — the one that produced H1, H2 and H3 —
is the thing that is missing.

No number from any of these local files is used as a stand-in for a synthetic result.

---

## 4. What to transfer

### Priority A — the whole repository (strongly preferred)

```
~/Documents/github/m5dataset          including .git
```

One `git bundle --all` or a full archive of the directory including untracked files. The
repository has no remote, so the working tree may hold scientific artifacts that no commit
contains; a bundle alone would silently drop them. Send **both** a bundle and an archive of
the untracked/modified files, or simply archive the entire directory.

Suggested, run on the old machine:

```
cd ~/Documents/github/m5dataset
git bundle create /tmp/m5dataset_$(git rev-parse --short HEAD)_$(date +%Y%m%d).bundle --all
git status --porcelain > /tmp/m5dataset_status.txt
git rev-parse HEAD > /tmp/m5dataset_head.txt
git remote -v > /tmp/m5dataset_remotes.txt
tar czf /tmp/m5dataset_full_$(date +%Y%m%d).tar.gz -C ~/Documents/github m5dataset
sha256sum /tmp/m5dataset_* > /tmp/m5dataset_hashes.txt
```

Nothing is reset, cleaned or checked out. The working tree is left exactly as it is.

### Priority B — the minimum, if the whole repository cannot be moved

Directory names are known; individual file names inside them are **not** assumed here and
must be taken from whatever the old repository actually contains.

```
#   source path (on the old machine)                   why it is needed          consumer
────────────────────────────────────────────────────────────────────────────────────────────
1   m5dataset/experiments/temporal_dependence/         the rho sweep package;    C1, Fig 2,
      (whole directory, code + any results)            named as the home of      Stage 4,
                                                       analysis.py, the          H1 provenance
                                                       series-cluster paired
                                                       bootstrap, stage1_figures
                                                       and recomputed_contrasts
2   m5dataset/experiments/decomposition_when_helps/    the study CLI
      (whole directory, code + any results)            (audit / build / stage-a  C1, Stage 1,
                                                       / run / report) and the   Stage 3
                                                       prediction-curve
                                                       diagnostic
3   m5dataset/_docs/history/2026-08-01.md              the detailed records the  provenance for
    m5dataset/_docs/history/2026-08-02.md              local history explicitly  every stage
    m5dataset/_docs/history/2026-08-03.md              defers to; 08-03 records
    m5dataset/_docs/history/2026-08-04.md              the Stage 1 generator
    m5dataset/_docs/history/2026-08-05.md              provenance audit
                                                       (CONDITIONALLY_VALID) and
                                                       the 18-cell stationary
                                                       Markov rho sweep
4   whatever preregistration / pre-analysis            fixes the hypotheses and  C1-G6, H1/H2/H3
    specification the synthetic study froze            the metric before results mapping
    (search for prereg*, pre_analysis*, *_spec.json)
5   the synthetic study's config for the               establishes the           C1-G1, C1-G2
    length-576 / 384-480-576 setting                   marginal control and the
                                                       dependence manipulation
6   Stage 1 per-cell result table                      the C01..C08 condition    Stage 1
    (search for stage1*, C0[1-8], *contrast*)          improvements and the      verification
                                                       four factor effects
7   Stage 3 mechanism diagnostic outputs               p_hat / mu_hat / forecast Stage 3, and
    (search for the prediction-curve or gate           traces on chosen series   the wording
    diagnostic outputs)                                                          split against
                                                                                 the real-data
                                                                                 occurrence BSS
8   Stage 4 / 18-cell stationary rho sweep results     rho_interval x            Stage 4, Fig 2,
    (search for the PHASE C sweep outputs)             rho_magnitude x sparsity  H1 and H2
                                                       cells with CIs            provenance
9   seed / run manifests for the above                 replications and          C1-G7
                                                       provenance
10  the tests covering the synthetic generator         confirms the DGP behaves  C1-G1, C1-G2
    and the Point/Hurdle models                        as specified
11  figure source data for the synthetic heatmap       Fig 2 without re-running  Fig 2
    (cell-level table, axes, metric, uncertainty)      anything
12  m5dataset/src/rcoi/ and any config the above       import closure, so the    reproducibility
    modules import                                     transferred code runs
```

Items 1 and 2 are the two that matter most; if only two things can be moved, move those
directories in full, including untracked outputs.

---

## 5. What is blocked until this arrives

```
blocked item                              current state
────────────────────────────────────────────────────────────────────────────────────
Contribution C1                           cannot be raised above PARTIALLY_VERIFIED
C1-G1 marginal control verified           the mechanism is visible in the local
                                          unified_temporal_27_v3 Group A design, but
                                          that is not the paper DGP
C1-G2 dependence manipulation verified    same
C1-G3 Point/Hurdle fairness verified      PointDLinearParamMatched and FactorizedDLinear
                                          are local; the paper study's training budget
                                          and split are not
C1-G4 Stage 1 result artifact             ABSENT
C1-G5 Stage 4 result artifact             ABSENT
C1-G6 metric / sign convention verified   "rmse_mean_truth" is defined in the local kill
                                          test prereg; whether the paper study used the
                                          same definition is unconfirmed
C1-G7 seed / replication provenance       ABSENT
Figure 2                                  RECONSTRUCTION_BLOCKED - no cell-level table
Abstract sentence 4                       stays a placeholder
Stage 1 / Stage 3 / Stage 4 verification  not started
Hypothesis provenance mapping             not started
```

Everything downstream of the synthetic study — C2, C3, the routing chain, the empirical
hypothesis statuses — is unaffected and remains as recorded in `results/paper_synthesis/`.

---

## 6. Numbers quoted in local history but NOT verified

Recorded here so they are not mistaken for evidence later. They appear in
`_docs/history/2026-08-08.md`, written by a different session, and **no artifact backing them
exists on this machine**:

- condition set `C01`–`C08` with per-condition hurdle improvement rates,
- four factor effect sizes quoted as `+7.83 / −4.58 / −6.26 / −16.74`, point estimates only,
  with no interval,
- a magnitude manipulation check quoted as `−0.73`, attributed to a theoretical value of
  `−25/34 = −0.735`,
- an 18-cell stationary Markov rho sweep, described but not quantified.

None of these is used anywhere in `results/paper_synthesis/`, and none may be cited until the
underlying artifact is transferred and hashed.

---

## 7. What was deliberately not done

- No synthetic run was created to fill the gap.
- `results/paper_synthesis/` was not modified, and its verdict was not promoted.
- No `results/paper_synthesis_verified/` was created — it is defined only for the case where
  verification actually happened.
- The routing stop is untouched: `HANDCRAFTED_FEATURE_GATE_STOP`,
  `RAW_SEQUENCE_GATE_STOP`, `ROUTING_MODEL_DEVELOPMENT_STOP`.
- No model was trained, no TEST was scored, nothing was committed or pushed.

---

## 8. Second search, 2026-08-09 — recovery attempt on this machine

A recovery-package protocol was run against this machine on the assumption that the
repository was here. It is not. The search this time followed the protocol's own criteria
($HOME/Documents, $HOME, /home/<user>) rather than repeating the earlier one.

```
check                                                  result
────────────────────────────────────────────────────────────────────────────────────
uname                                                  MINGW64_NT-10.0-26200
                                                       DESKTOP-KVUSKN8  (Windows)
$HOME                                                  /c/Users/User
$HOME/Documents/github/m5dataset                       absent
$HOME/m5dataset                                        absent
/home/minjae/Documents/github/m5dataset                absent
/home                                                  does not exist
WSL distributions                                      none — wsl.exe reports the
                                                       subsystem is not installed
$HOME/Documents/GitHub/                                federated-route-finding, 새 폴더
every .git under /e/CODING, $HOME/Documents,           16 repositories, none of them
  $HOME/Desktop, $HOME/Downloads (depth 5)             m5dataset
temporal_dependence / decomposition_when_helps         0 hits
  under /e/CODING and $HOME (depth 7)
```

This machine is the **new** one. The repository is on the **old** Linux host referenced
throughout the migrated history as `/home/minjae/...`. No substitute repository was used,
and nothing was inferred from the local `unified_temporal_27_v3` or
`om_factorization_killtest` packages, which section 3 already established are not the paper
DGP.

**Verdict for this machine: `SYNTHETIC_SOURCE_NOT_FOUND`.**

### What was produced instead

`make_recovery_package.sh`, in this directory. It implements steps 1–16 of the recovery
protocol and is meant to be copied to the old machine and run there. It has been
syntax-checked (`bash -n`, clean) and its not-found branch was exercised here, but its
archiving path has never executed because there is nothing here to archive.

What it does, in order: seals the git state before touching anything; writes
`source_path.txt`, `git_head.txt`, `git_branch.txt`, `git_status_porcelain.txt`,
`git_remotes.txt`, `git_log_head.txt`, tracked / untracked / ignored / modified / staged
file lists and the repository size; runs a name-agnostic semantic search for the Stage
1/3/4, preregistration, DGP, bootstrap and seed artifacts into
`scientific_candidate_paths.txt`; tars the **entire directory including `.git` and every
untracked file** into a sibling `_recovery_exports/` folder; verifies the listing contains
`.git/HEAD` and every untracked file; writes and checks a SHA256; creates and verifies a
`git bundle --all` with its own SHA256; hashes every modified/untracked file into
`working_tree_scientific_manifest.csv`; builds
`synthetic_critical_artifact_inventory.md` across the eleven categories A–K; greps for any
file that disclaims itself with `"not the paper DGP"` and for the length-576 / 384-480-576
configuration so the paper DGP is identified rather than assumed; spot-checks source-versus-
archive hashes; writes `recovery_package_manifest.csv`; and finally re-reads
`git status --porcelain` to prove `SOURCE_UNTOUCHED`.

It performs no reset, clean, checkout, restore, commit, push or remote creation, excludes
nothing, and writes nothing inside the source repository.

---

## 9. Third search, 2026-08-09 — unbounded, and by content rather than name

A recovery protocol was issued a second time, this one opening with "assume this machine is
the original Linux machine". It is not, so the search was repeated in a way that could
actually falsify the earlier conclusion rather than repeat it: no depth limit, and a content
signature instead of a directory name.

```
check                                                        result
──────────────────────────────────────────────────────────────────────────────────────
mounted drives                                               C: and E: only
                                                             (D:, F: report type unknown)
find /e /c/Users -type d -name m5dataset                     0 hits
  -o -name temporal_dependence                               0 hits
  -o -name decomposition_when_helps                          0 hits
  (no -maxdepth: full recursion of both drives)
grep -rl --include=*.py -E                                   exactly 1 hit, and it is the
  "FactorizedDLinear|decomposition_when_helps" /e /c/Users   already-known local kill test:
  (completed, exit 0)                                        covariate-trust-pilot/experiments/
                                                             om_factorization_killtest/models.py
```

Both searches are complete. The name search at unbounded depth over both real drives returned
**0 hits**. The content search returned a single file, which is the kill-test package already
inventoried in section 3 of this document and already excluded as not the paper DGP by its own
preregistration. No trace of `m5dataset`, `temporal_dependence` or `decomposition_when_helps`
exists on this machine by name or by code signature.

Machine facts that settle the premise:

```
uname       MINGW64_NT-10.0-26200 DESKTOP-KVUSKN8   (Windows 11)
$HOME       /c/Users/User                            (the source record says /home/minjae)
/home       does not exist
WSL         not installed; wsl.exe reports the subsystem is absent
git repos   16 enumerated across /e/CODING, $HOME/Documents, $HOME/Desktop,
            $HOME/Downloads — none is m5dataset
```

**Verdict, fourth independent confirmation: `SYNTHETIC_SOURCE_NOT_FOUND` on this machine.**

Assuming the machine is the original one does not make the files exist, and archiving some
other repository in its place is precisely the contamination that rule 19 of the same
protocol ("do not presume another DGP is the paper DGP") exists to prevent. Step 0 of the
protocol instructs an immediate stop in this case, which is what was done.

`make_recovery_package.sh` in this directory remains the deliverable: it is the protocol,
written to run on the machine that does have the repository.
