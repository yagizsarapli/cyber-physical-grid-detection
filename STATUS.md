# GRIDRA Cyber-Physical Microgrid — Status

Last updated: 2026-09-21. Supersedes the "Current backbone / Next" section
in `README.md`, which only described the Aug-11 starting point and was
never updated through phases 2B–2T.

## Summary

**Headline finding, corrected 2026-09-21 (see §2.5): topology-aware
relational features do not measurably help.** A 5-bus/0.4kV synthetic
microgrid (PV as a GFL inverter, BESS as a GFM inverter) with a WLS AC
state estimator was used to test whether topology-aware relational
features let a classifier (a) tell a cyber false-data-injection attack
apart from a legitimate physical disturbance, (b) localize which bus
is affected, and (c) do both within a one-cycle protection-relevant
computational target (50 Hz = 20 ms, confirmed from `net.f_hz`, not
assumed). The original answer to (a)/(b) — yes, confirmed across 4
seeds — was real but built on an incomplete comparison. §2 below is
kept as the accurate history of how that result was reached and
validated; §2.5 explains the ablation that overturned it: once
`topology_fusion` is compared against a genuinely topology-free
`residual_plus_prior` set (rather than against `residual_only` or
`prior_only` individually), the two are statistically indistinguishable
for detection and `residual_plus_prior` is actually **ahead** for
localization — across both stress conditions and all 4 seeds. The gain
over `prior_only` alone is real and large; it was never a topology
effect.

After profiling and fixing the actual latency bottleneck (a software
inefficiency, not the model), the full pipeline meets the 20ms budget
at the median, p95, and p99 (16.4 / 17.0 / 17.1 ms), with a single
outlier over 300 trials still exceeding it (21.3 ms max) — this part
of the finding is unaffected by the §2.5 correction.

Tested at a bigger, standard scale (IEEE 14-bus, §5) at three
escalating sample sizes, ending at n=500 — matching the 5-bus study's
own 300-scenario test-set size exactly, and built with a completely
independent, size-agnostic feature design. That design turned out to
need the *same* correction as §2.5, found independently during this
audit (§5's own "second instance" writeup). Once fixed and confirmed
across 4 independent seeds (same discipline as the 5-bus study),
**detection agrees exactly with 5-bus**: residual-only (0.750) and the
topology-free ablation (mean 0.750) both edge out topology-fusion
(mean 0.741), consistently, sign never flipping across seeds.
**Localization does not agree — a real, 4-seed-confirmed reversal, not
a second null result**: topology-fusion keeps a small but consistent
lead (mean gap +0.018, range +0.013 to +0.033, never flipping sign).
Read together with §2.5, this is not "the same null result twice" —
it's "detection: the same null result twice; localization: a genuine,
replicated reversal at IEEE-14 specifically" — a more complicated and
more interesting answer than either a clean yes or a clean no, and
(see below) not simply a function of scale either. **A third
network (IEEE 30-bus, §5.6) looked, in a single run, like it confirmed
this pattern too — it doesn't, once replicated the same way.** Re-run
across the same 4 seeds used at IEEE-14, neither gap holds a consistent
sign at IEEE-30 (detection: −0.007/+0.000/+0.013/−0.003; localization:
+0.020/+0.007/+0.013/−0.020) — a clean null, unlike IEEE-14's uniformly
one-directional gaps. The honest picture across all three networks: the
localization reversal is real and seed-confirmed at exactly one scale
(IEEE-14), not a general "standard/meshed networks" property — the
single-run IEEE-30 "corroboration" was itself an instance of the
single-run-narrow-margin mistake this project's own discipline exists
to catch, caught this time before it reached the paper.

One honest limit found by testing it directly: this advantage
(residual+prior fusion, not topology) is **pattern recognition of
known attack types, not zero-day generalization**. A model trained
with one cyber attack type entirely withheld catches almost none of it
at test time (0–4% recall, vs. 89–100% when that type is in training).
Framed accurately, not overclaimed — see §4.

**⚠ Everything above this line, and everywhere else in this file before
§6, describes the pre-audit numbers.** §6 documents a post-submission
code-level audit (2026-09-22) that found and fixed three real bugs
(oracle-leaked "residual" features at IEEE-14/30, test-set model
selection at all three networks, non-topology-free baselines at
5-bus) and re-ran all three networks. The corrected picture is
materially different — most notably, IEEE-30 detection goes from null
to a real, 4-seed-consistent topology *advantage*, and 5-bus
localization goes from a small residual_plus_prior edge to a clean
null. `paper/main.tex`, `PAPER_DRAFT.md`, and `README.md` have **not**
been updated to match and currently describe the old, superseded
story. Read §6 before trusting any specific number anywhere else in
this file or in the paper files.

## Method

- **Testbed**: `exploration/01_microgrid_topology.py` — 5 buses, 4 lines, fixed
  roles (PCC, PV/GFL, load A, BESS/GFM, load B). `net.f_hz = 50.0`,
  `net.sn_mva = 0.1`.
- **State estimation**: WLS via `pandapower.estimation.estimate`
  (`21_nonlinear_stealth_fdia_bdd_benchmark*.py`), 23 measurements/cycle
  (5×V, 5×P, 5×Q bus injections, 4×P, 4×Q line flows).
- **Attack model**: two families —
  - *naive*: single-sensor voltage spoof, magnitude drawn from a fixed
    range.
  - *stealth (model-consistent FDIA)*: exact nonlinear
    `a = h(x_attack) - h(x_true)` construction, so it is consistent with
    the AC measurement model and evades a simple chi-square residual test
    by construction.
- **Benign hard negative**: `physical_load_disturbance` — a legitimate
  load step (25–70% multiplier), deliberately similar in residual
  footprint to an attack.
- **Features compared** (`23_topology_aware_cyber_physical_localization*.py`):
  `residual_only` (conventional WLS residuals), `prior_only` (a
  separately protected operating-point prior/innovation), and
  `topology_fusion` (both, plus neighbor-graph relational features:
  local-vs-neighbor innovation/residual, incident-edge aggregates).
- **Models**: LogisticRegression, RandomForest, HistGradientBoosting,
  compared against 1-feature interpretable-score baselines.
- **Evaluation**: replication-group-safe train/validation/test splits
  (70/15/15), group-aware bootstrap confidence intervals.

## 1. Leakage check on the original Phase 2S result

The original result was suspicious: 1.000 on every detection *and*
localization metric, 3 different model types, zero-width bootstrap
CIs, n=72 test. `feature_columns()` claimed (in a comment, not in code)
that `eval_*`/`label_*` columns are never model features — the code
only excluded 6 named metadata columns. Fixed in
`23_..._FIXED.py`. **Re-running changed nothing** — `build_graph_features()`
never pulled those columns in to begin with. Real defensive gap, kept
fixed, but not the cause. The actual cause (already flagged by the
script's own printed interpretation): Phase 2Q's original attacks had
large, obvious state bias — the task was too easy, not leaked.

## 2. Realistic stress test — and why the first read of it was wrong

**Attack magnitude**, reduced ~3x toward the noise floor
(`21_..._HARD.py`):

| | Naive magnitude | Stealth dv | Stealth da |
|---|---|---|---|
| Original | U(0.035, 0.060) pu | U(0.012, 0.035) pu | U(0.40, 2.20)° |
| Hard | U(0.012, 0.025) pu | U(0.004, 0.012) pu | U(0.10, 0.50)° |

**Load-forecast uncertainty**, doubled on top of the hard attack
magnitude (`22_..._HARDER.py`): sigma 2.5%→5.0% per load, clip
±6%→±12%.

First pass used n_rep=120 (72 test scenarios) and looked like a
genuine reversal: prior_only (bal-acc 0.903) beat topology_fusion
(0.889) at detection, while topology_fusion kept localization
(1.000). **That reversal did not hold up.** Regenerating with n_rep=500
(300 test scenarios, 4x the statistical power) shows it was sampling
noise, not a real effect:

| | Original (easy, n=72 det.) | Hard (n=72 det.) | **Hard (n=300 det.)** | **Harder: +2x forecast noise (n=300 det.)** |
|---|---|---|---|---|
| Best detection (bal-acc) | topology_fusion 1.000 | prior_only 0.903 | **topology_fusion 0.950** (prior_only 0.873) | **topology_fusion 0.953** (prior_only 0.83–0.90) |
| Best localization (top-1, n=36/150/150 loc. resp.) | topology_fusion 1.000 | topology_fusion 1.000 | **topology_fusion 0.980** (prior_only 0.780) | **topology_fusion 0.980** (prior_only 0.767) |
| Clean-data false-alarm rate | 0.0 | 0.28–0.50 | 0.067 | ~0.03–0.19 |

95% bootstrap CI for the best detector at n=300 (hard):
topology_fusion/HistGradientBoosting bal-acc **[0.923, 0.977]**,
non-overlapping with prior_only/LogisticRegression **[0.830, 0.897]**.

**Conclusion (superseded by §2.5 below, kept as accurate history)**:
this section correctly established that topology_fusion beats
prior_only alone, is stable under a second stress axis, and is
statistically supported (non-overlapping CIs) — all still true. What
this section did *not* establish, because the comparison was never
run, is whether that gap is a topology-specific effect or simply an
effect of topology_fusion having strictly more information than
prior_only alone. §2.5 runs that comparison. The n=72 "detection tie"
was a genuine research dead-end correctly caught by insisting on a
harder test — but the fix for a suspicious result should also include
checking whether the test set was simply too small before trusting a
reversal, which the first pass here skipped; §2.5 is the same lesson
applied to a different kind of gap in the comparison, not the sample
size.

## 2.5. The topology advantage did not survive its own ablation

**What was missing from §2**: every comparison above is
`topology_fusion` vs. `residual_only` *or* `prior_only`, one at a
time. Nothing above compares `topology_fusion` against the
*combination* of both — the actual topology-free control. Building
that control (prompted by a request to isolate exactly what topology
contributes, as a next-step ablation) surfaced a real bug.

**The bug**: `feature_columns()` (`23_..._HARD.py`) defined
`residual_only`/`prior_only` by matching column-name substrings
(`"residual"`, `"innovation"`). But the explicit topology-relational
columns are *named after* the quantities they compare across
neighbors — e.g. `b0_neighbor_mean_residual`,
`b0_incident_edge_max_innovation`, `b0_local_minus_neighbor_residual`
— so the same substring match silently pulled most of them into the
"baseline" sets. Measured directly: of 65 explicit relational columns
inside `topology_fusion` (147 columns total), 30 had leaked into
`residual_only` (64 columns) and 35 into `prior_only` (83 columns).
`residual_only ∪ prior_only` equaled `topology_fusion` exactly, column
for column — meaning `topology_fusion` could never have been compared
against a condition that had all the same non-relational information
and nothing else, because that condition was never constructed.

**The fix**: exclude columns matching explicit relational markers
(`neighbor_`, `incident_edge_`, `local_minus_`, `local_over_`) from
`residual_only`/`prior_only`, and add their corrected union as a new,
fourth, genuinely topology-free `residual_plus_prior` set. Applied to
both `23_..._HARD.py` and `23_..._HARDER.py`, and to the localization
side (`node_feature_sets()`, which used an explicit whitelist rather
than regex and was *not* buggy in the same way — it just never had a
"both, no topology" entry either). Verified empirically after the fix:
zero relational-column leakage into any non-topology set;
`residual_only`=34, `prior_only`=48, `residual_plus_prior`=82,
`topology_fusion`=147 (unchanged), with `topology_fusion −
residual_plus_prior` exactly equal to the 65 relational columns.

**Corrected result** (best model per cell, $n$=300 test):

| | Detection (bal-acc) | Localization (top-1) |
|---|---|---|
| Hard: topology_fusion | 0.950 | 0.980 |
| Hard: **residual_plus_prior** | 0.950 | **0.993** |
| Harder (+2x noise): topology_fusion | 0.953 | 0.980 |
| Harder (+2x noise): **residual_plus_prior** | 0.953 | **0.993** |

Detection is an exact tie in both conditions. Localization: the
topology-free ablation is *ahead* of topology_fusion, not behind it.
Both still comfortably beat prior_only alone (0.853–0.867 detection,
0.767–0.780 localization) and residual_only alone (0.687–0.690
detection, 0.700 localization) — the gain over either single baseline
is real; it was never a topology effect.

**Confirmed across all 4 seeds** (re-running `26_multi_seed_replication.py`
with the corrected feature sets):

| Seed | topology_fusion det. | residual_plus_prior det. | gap | topology_fusion loc. | residual_plus_prior loc. | gap |
|---|---|---|---|---|---|---|
| 20260812 | 0.950 | 0.950 | 0.000 | 0.980 | 0.993 | −0.013 |
| 42 | 0.943 | 0.957 | −0.013 | 0.973 | 0.973 | 0.000 |
| 777 | 0.907 | 0.913 | −0.007 | 0.987 | 0.987 | 0.000 |
| 2024 | 0.970 | 0.960 | +0.010 | 1.000 | 1.000 | 0.000 |
| **mean ± std** | 0.943±0.026 | 0.945±0.022 | **−0.003±0.010** | 0.985±0.011 | 0.988±0.011 | **−0.003±0.007** |

Both gaps are consistent with zero, and — unlike the original
`topology_fusion`-vs-`prior_only` gap, whose sign never flipped across
the same 4 seeds — the corrected gap's sign does not hold a consistent
direction (2 seeds slightly favor topology_fusion on detection, 2
slightly favor residual_plus_prior; localization is 3 exact ties and 1
seed favoring residual_plus_prior). This is the signature of noise
around zero, not a real, small effect. For reference, the original
`topology_fusion`-vs-`prior_only` gap recomputed on this same corrected
run is +0.089±0.037 (detection) / +0.207±0.032 (localization) —
essentially identical to §2's original numbers, confirming that gap
was real and reproducible, just entirely attributable to
`residual_plus_prior` matching `topology_fusion`, not to `prior_only`
catching up.

**Conclusion**: the corrected, honest finding is that topology-aware
relational features give no measurable benefit over a fair,
same-information (`residual_plus_prior`) ablation, on the 5-bus
network, under either stress condition, across all 4 validated seeds.
Combining residual and prior-innovation information is what drives the
real, multi-seed-confirmed gain over either alone — not topology. This
also resolves what looked, before this fix, like a puzzle: why would a
real, well-validated 5-bus effect fail to transfer to IEEE 14-bus
(§5)? It doesn't need to — it was never there to transfer.

**Why, structurally, not just empirically**: predicted each of the 65
relational columns from `residual_plus_prior` alone, 5-fold
cross-validated ridge regression, on the train split. Mean out-of-fold
R² = 0.960 (median 1.000, min 0.652, 91% of columns above 0.8). On a
fixed, small (5-bus, 4-line) topology, a neighbor-mean or
incident-edge aggregate over a fixed, small neighbor set is close to
an affine function of the very per-bus values `residual_plus_prior`
already has in full — so a flexible classifier (HistGradientBoosting)
can reconstruct nearly all of that relational information internally
without ever being given it explicitly. This is a genuine explanation,
not just a restatement of the result: it predicts *when* topology
might start to matter (a network/feature design where relational
aggregates are *not* near-linearly redundant given local information),
and gives a cheap pre-check (this same out-of-fold R² test, no
classifier training required) anyone could run on a candidate network
before investing in a full topology-aware pipeline. It is specific to
the 5-bus mechanism, though — it does not by itself explain the
independent IEEE 14-bus null result, which uses an unrelated feature
design; the two results are consistent, not mechanistically identical.

## 3. Real-time latency: found the bottleneck, fixed most of it

Budget: 1 cycle @ confirmed 50 Hz = 20 ms. One-time network/model
setup excluded from the timed loop (matches real deployment — a relay
doesn't rebuild its model every cycle).

| Step | Median | p95 | p99 | Max | vs. 20ms |
|---|---|---|---|---|---|
| Original (`run_wls`, RandomForest, flat init, tol=1e-7, max_iter=40) | 42.98 ms | — | — | — | 2.15x over |
| + WLS warm-start (`init="results"`) | 37.14 ms | — | — | — | 1.06x speedup only — **dead end** |
| Diagnosis: split `run_wls` | `add_measurement_vector` 19.9ms + `estimate()` 19.4ms | | | | found the real 50/50 split |
| Diagnosis H1: `max_iterations` 40→5 | no speed change | | | | rules out iteration count |
| Diagnosis H1: `max_iterations`=3 | convergence collapses to 1/20 | | | | true need is ~4-5 iterations |
| Diagnosis H2: `tolerance` 1e-7→1e-4 | ~14% faster on `estimate()` alone | | | | state error ~2e-9 pu, safe |
| **Fix 1**: bulk-write `net.measurement["value"]` instead of looping `pp.create_measurement()` 23x/cycle | 19.4ms → **0.017ms (1158x)** | | | | verified **bit-identical** WLS output |
| **Fix 2**: RandomForest → LogisticRegression | 2.88ms → 0.20ms inference | | | | competitive accuracy, see §2 |
| **Final** (Fix 1 + Fix 2 + tol=1e-4/max_iter=10, N=300, 0 convergence failures) | **16.41 ms** | **16.95 ms** | **17.09 ms** | 21.35 ms | **median/p95/p99 within budget; 1 outlier over, ~6ms margin otherwise** |

**Root cause was never the ML model or solver iteration count** — it
was `pp.create_measurement()` called in a Python loop every cycle to
rebuild a measurement table whose structure never changes, only its
values. That one fix did most of the work; swapping the classifier
did the rest.

**Honest remaining gap**: at N=300, 299/300 decisions meet budget with
margin; the single outlier (21.35ms) comes from `estimate()`'s own
internal variance, which isn't further tunable from the outside
without a custom estimator for this fixed topology instead of
pandapower's general-purpose WLS implementation.

## 4. Multi-seed replication and honest generalization limits

The §2 table came from one seed's data draw with only a within-run
bootstrap CI — not the same as independent replication. Re-ran full
generation + evaluation (`26_multi_seed_replication.py`) for 4
independent seeds (20260812, 42, 777, 2024), n_rep=500 each:

| Seed | topology_fusion detection | prior_only detection | gap | topology_fusion localization | prior_only localization | gap |
|---|---|---|---|---|---|---|
| 20260812 | 0.950 | 0.873 | +0.077 | 0.980 | 0.780 | +0.200 |
| 42 | 0.943 | 0.850 | +0.093 | 0.973 | 0.780 | +0.193 |
| 777 | 0.907 | 0.857 | +0.050 | 0.987 | 0.807 | +0.180 |
| 2024 | 0.970 | 0.837 | +0.133 | 1.000 | 0.747 | +0.253 |
| **mean ± std** | **0.943 ± 0.026** | **0.854 ± 0.015** | **+0.088 ± 0.035** | **0.985 ± 0.011** | **0.778 ± 0.025** | **+0.207 ± 0.032** |

The gap's sign never flips across 4 independent draws — this is a
real effect, not noise from one lucky dataset.

**Held-out attack-type generalization** (`27_held_out_attack_type_generalization.py`).
Note this replaces an earlier plan to "re-run Phase 14 (zero-day)/17
(hard-negative) against the HARD data" — those scripts turned out to
depend on a completely different upstream data lineage
(`phase2g_v2_scenario_metadata.csv`, a richer 5-cyber/3-physical event
taxonomy) that isn't compatible with the Phase 2Q/2R/2S graph-feature
files without substantial bridging work. This script asks the same
*kind* of question directly on the data already validated here
instead: train HistGradientBoosting with one cyber attack type
**completely removed** from training, test recall on exactly that
type. Originally run against `topology_fusion` only, on data that
(unnoticed at the time) happened to be from the seed=2024 pass of
§4's multi-seed loop rather than the primary seed; re-run here against
the current primary-seed (20260812) data **and** the corrected
`residual_plus_prior` set, closing the gap flagged in §9/Limitations
("not independently re-tested for residual_plus_prior").

| Held out | Feature set | Recall when withheld | Recall when trained on it |
|---|---|---|---|
| naive_single_sensor_corruption | topology_fusion | **0.000** | 0.827 |
| naive_single_sensor_corruption | residual_plus_prior | **0.000** | 0.827 |
| nonlinear_model_consistent_fdia (stealth) | topology_fusion | **0.033** | 1.000 |
| nonlinear_model_consistent_fdia (stealth) | residual_plus_prior | **0.047** | 1.000 |

(The 0.893/0.040 numbers previously reported here were computed
against the seed=2024 snapshot of `phase2s_hard_graph_feature_matrix.csv`,
left over from an earlier step in this session rather than a
computation error — reproducible, just not from the primary-seed data
the rest of this document uses. Corrected above.)

**Honest conclusion**: the detector does not extrapolate to an attack
mechanism it has never seen — it recognizes the specific statistical
signatures of the attack types in its training data (including the
stealthy one, very well) but has no mechanism to generalize beyond
them, exactly as expected for a supervised classifier with only two
attack families to learn from. This holds **identically regardless of
whether topology-relational features are included** — `topology_fusion`
and `residual_plus_prior` land within 1.4 points of each other on
every cell above, exactly as §2.5's in-distribution finding would
predict for a question (generalizing to an unseen mechanism) that has
nothing to do with topology specifically. Any real deployment claim
needs this stated plainly: strong within-distribution discrimination,
no demonstrated zero-day capability, for either feature set.

## 5. Scale study: IEEE 14-bus — confirmed, not just piloted

Everything above ran on a custom 5-bus toy microgrid. Tried scaling up
to a real, standard test system.

**CIGRE MV benchmark tried first, rejected**: `pandapower.networks.
create_cigre_network_mv()` represents its switches via 3 auxiliary
internal buses (15 `net.bus` rows vs. 18 in `net._ppc["bus"]`), which
broke this project's `h_ac()` — it indexes by `net.bus.index` directly
and assumes that equals the internal ppc bus count. IEEE 14-bus
(`pandapower.networks.case14()`) has none of that (14 `net.bus` == 14
ppc buses, confirmed by inspection) and is arguably the more
appropriate choice anyway — it's the single most commonly used test
system in the power-system state-estimation/FDIA literature
specifically, more so than CIGRE MV.

Every WLS/measurement/attack function (`build_measurement_model`,
`h_ac`, `measurement_schema`, `run_wls`, both attack generators) is
already network-size-agnostic — confirmed by inspection (they iterate
`net.bus.index`/`net.line.index`/`model["n_bus"]` generically, never a
hardcoded 5) and reused **unmodified** on IEEE 14-bus in
`28_ieee14_scale_replication.py`. What had to be rebuilt: the feature
engineering, since the original `b0_.../b4_...` fixed per-bus columns
don't generalize past N=5. Replaced with a genuinely size-agnostic
design: global summary statistics (mean/std/max/2nd/peak-ratios) of
residual and innovation arrays, plus topology context for only the
single argmax (most suspicious) node — this is a cleaner design than
the original, not just a scaled copy.

**Three passes, escalating power, same direction each time:**

1. n_rep=80 (~24 test cases, argmax-only localization): genuinely too
   small to trust — inconsistent across model types, the same noise
   signature §2 already taught this project to distrust.
2. n_rep=200 (798 scenarios, 60 test cases, with a real trained
   localizer — per-node features + HistGradientBoosting ranking every
   node's P(is target), not just argmax): the pattern first became
   internally consistent (detection and localization told the same
   story) rather than noisy model-to-model disagreement.
3. **n_rep=500 (1993 scenarios, 300 test cases for detection, 150 for
   localization — matching the 5-bus study's own n=300 power exactly):
   confirmed and sharpened, not reversed.**

Passes 1 and 2 above (n_rep=80, 200) predate the bug fix described
next and used the original, uncorrected `residual_only`/`prior_only`/
`topology_fusion` definitions — their role is establishing that n=80
was too small to trust (a sample-size lesson, independent of the
feature-definition bug). Only the final n_rep=500 pass below uses the
corrected features.

**A second, independent instance of §2.5's bug, found during this
audit.** The claim below (first written the same day as §2.5) that
this feature design was "never touched by the §2.5 bug" was wrong —
checked directly and it wasn't true. `feature_sets()` (detection)
included the single-argmax node's neighbor-comparison columns
(`resnode_argmax_neighbor_mean/max`, `..._local_minus_neighbor`,
`..._n_neighbors`) inside `residual_only`/`prior_only` via the same
kind of prefix match that caused §2.5's bug. The localization feature
dict was worse — an *explicit* list, not even a prefix collision:
`"residual_only": ["node_res", "node_res_nb_mean", "node_res_nb_max",
"node_res_local_minus_nb", "node_n_neighbors"]` — 4 of 5 listed
columns are neighbor/degree features, deliberately included under
"residual_only". `topology_fusion` was, once again, defined as exactly
`residual_only + prior_only`, so it never held any information beyond
what those two "baselines" already had between them, for either task.
Fixed identically to §2.5: relational/degree columns excluded from
`residual_only`/`prior_only` by an explicit `"neighbor"` name marker
(also catches `n_neighbors`), their union added as a fourth
`residual_plus_prior` set. Re-ran the full n_rep=500 pass with the fix.

**Corrected n=500 result** (best model per feature set, single run):

| | 5-bus (n=300 det./n=150 loc. test) | **IEEE 14-bus, n=500 (corrected, single run)** |
|---|---|---|
| Detection | topology_fusion 0.950 = residual_plus_prior 0.950 | residual_only/LR **0.750** best; residual_plus_prior 0.743, topology_fusion 0.737 close behind — neither the ablation nor topology_fusion leads |
| Localization | residual_plus_prior **0.993** > topology_fusion 0.980 | topology_fusion **0.560**, narrowly ahead of residual_only/prior_only (0.547) and residual_plus_prior (0.533) |

**Then multi-seed-confirmed** (30_ieee14_multi_seed_replication.py,
same 4 seeds as the 5-bus study, n_rep=500 each — run specifically
because the localization margin above was too narrow to trust off one
run, per this project's own established rule):

| Seed | Detection gap (topo − best non-relational) | Localization gap (topo − best non-relational) |
|---|---|---|
| 20260921 | −0.013 | +0.013 |
| 42 | −0.013 | +0.013 |
| 777 | −0.011 | +0.033 |
| 2024 | −0.021 | +0.013 |
| **mean ± std** | **−0.015 ± 0.004** | **+0.018 ± 0.010** |

Sign never flips in either direction, across any seed.

**Honest reading, updated four times now.** First pass (before §2.5):
"topology helps at 5-bus, not at 14-bus — scale-dependent." Second
pass (after §2.5, before the §5 fix): "topology never helped at 5-bus
either — 14-bus independently confirms a null result, and wasn't
touched by the bug" (wrong on the "wasn't touched" part — it was).
Third pass (after the §5 fix, single IEEE-14 run): "no feature set is
confidently ahead of the topology-free ablation at either scale" —
wrong too, just not caught until the multi-seed re-run above. The
correct reading, now that detection and localization have each been
checked with the same rigor at both networks: **detection agrees
cleanly at both networks** (topology_fusion never ahead of the best
non-relational alternative — an exact tie at 5-bus, a consistent
4-seed deficit at 14-bus). **Localization does not agree, and this is
now a confirmed, not a suspected, reversal** — but the 5-bus side of
it is smaller than it looks from the single primary-seed numbers
alone: those are 0.993 vs. 0.980 (a "decisive"-looking 1.3-point gap),
but the 4-seed mean gap is only −0.003 ± 0.007 (0.3 points, topology
never ahead across the 4 seeds but only barely behind). So the fourth
correction: **the topology-free ablation is never worse at 5-bus,
with only a small mean advantage, not "decisively" ahead** — while
topology_fusion keeps a comparably-sized but oppositely-signed lead at
IEEE-14 (mean gap +0.018, never flipping sign). Both are small; what's
real is the *direction* reversing between the two networks, not a
large effect at either one. This is the one place in the whole project
where topology-relational features show a real, replicated advantage
— and it only survived because the same "don't trust one run"
discipline that caught the original n=72 false reversal (§2) was
applied here too, rather than accepting the single-run "probably
noise" read.

That the residual_only/prior_only feature-definition bug was made
independently in two separately-written pipelines remains worth
noting on its own: it suggests this specific comparison mistake is an
easy trap in "topology-aware feature" engineering generally, not a
one-off slip in one script — but, now confirmed, it does not mean
"topology never helps anywhere." It means: check correctly, then take
the answer you actually get, even when (like here) it's more
complicated than a clean yes or no.

## 5.5. A literature-positioning claim was checked directly and didn't hold up as stated

The paper's Related Work section originally claimed that topology-aware
GNN papers for power systems "generally" don't compare against a
same-information, non-topological ablation — i.e., that this project's
own §2.5/§5 correction was catching a gap the field also has. Asked
directly to verify this against the actual nearest 10-15 papers before
submission, rather than leave it as an assumption.

**What checking directly (full text, not abstracts) actually found**:
the three closest papers ([RELATED_WORK.md](RELATED_WORK.md) entries
25-27) split two ways, not one. [arXiv:2506.03493](https://arxiv.org/abs/2506.03493)
(entry 25) and an earlier, closely related study by an overlapping
author group, [arXiv:2212.04592](https://arxiv.org/pdf/2212.04592)
(new entry 57, found while checking this) **both** compare their GNN
against a "regular DNN" baseline that receives the identical PMU-derived
node feature matrix and differs only in not receiving the adjacency
matrix — a clean, correctly-scoped, same-information ablation, read
directly from each paper's own Table II. That is exactly the kind of
check this project's own original (buggy) comparison lacked, done
correctly, in a real published paper. [arXiv:2503.22721](https://arxiv.org/pdf/2503.22721)
(entry 26) is genuinely ambiguous on the text available (a "fully
connected NN" baseline whose exact input features aren't confirmed
identical). [arXiv:2603.23357](https://arxiv.org/pdf/2603.23357)
(entry 27) couldn't be fetched at full-text level at all (HTML 404,
PDF returned as unreadable binary).

**Conclusion**: the original "the field generally doesn't do this"
claim does not survive contact with the 3 closest actual papers —
2 of 3 checkable ones clearly *do* perform this kind of ablation. This
is the same lesson `RELATED_WORK.md`'s own honesty caveat already
established for quantitative numbers (two invented figures caught
there previously), now applied to a qualitative claim: it's just as
easy to overclaim "nobody checks this" as to invent a statistic, and
just as checkable. **Fixed** in `paper/main.tex` §II: the claim is now
narrower and defensible — same-architecture ablations (GNN vs. DNN,
same features) are a known, sometimes-used check in this literature;
what this project additionally found is a *different*, lower-level
failure mode (a feature-*definition* bug that silently breaks an
intended non-topological baseline before any model ever sees it,
undetectable by an architecture-level ablation alone) plus a cheap,
complementary pre-check for it (the out-of-fold linear-redundancy test
already in §2.5). Source count updated throughout (40/56 → 41/57,
reflecting the one genuinely new, deeply-read source this pass added).

## 5.6. A third network (IEEE 30-bus) — the single-run corroboration didn't survive replication

Requested specifically to test whether IEEE-14's localization reversal
(§5) is IEEE-14-specific or a property of standard/meshed networks
generally. `31_ieee30_scale_replication.py` is a direct copy of the
now-fully-corrected `28_ieee14_scale_replication.py` with `pn.case14()`
swapped for `pn.case30()` and output filenames changed — everything
else (WLS/measurement/attack machinery, feature-set definitions, model
training/evaluation) is identical by construction, so this is a clean
scale-only comparison. IEEE 30-bus confirmed compatible before
committing compute: `net.bus` == internal ppc bus count (30 == 30,
checked directly, the same check that ruled out CIGRE MV originally),
so `h_ac()` needs no changes. Smoke-tested at n_rep=30 (0 failed
replications, all 4 feature sets present), then run in full at
n_rep=500 (2000 scenarios, 0 failed replications, 300 test scenarios
matching every other scale study in this project).

**First result (single run, seed 20260921) — appeared to closely match
IEEE-14 on both tasks:**

| | Detection (best) | Localization (best) |
|---|---|---|
| topology_fusion | 0.747 | **0.533** |
| residual_plus_prior | **0.753** | 0.507 |
| prior_only | 0.700 | 0.513 |
| residual_only | 0.750 | 0.513 |

Detection: topology_fusion (0.747) trails residual_plus_prior (0.753)
and residual_only (0.750) — gap −0.007, same direction as IEEE-14
(mean −0.015, range −0.021 to −0.011) and 5-bus (tied, not ahead).
Localization: topology_fusion (0.533) leads every non-relational
alternative (0.507–0.513) — gap +0.020, inside IEEE-14's own 4-seed
range (+0.013 to +0.033) almost exactly, and the opposite direction
from 5-bus. This is exactly what a §5-style single-run, narrow-margin
result looks like — and this project's own rule, established at §2 and
applied again at §5, is not to trust one of those without seed
replication. So, before writing up "two standard IEEE systems agree,"
we ran it.

**Second result (`30_ieee14_multi_seed_replication.py`'s sibling,
`32_ieee30_multi_seed_replication.py`, same 4 seeds, same n_rep=500
each) — it does not replicate:**

| Seed | Detection gap | Localization gap |
|---|---|---|
| 20260921 | −0.007 | +0.020 |
| 42 | +0.000 | +0.007 |
| 777 | +0.013 | +0.013 |
| 2024 | −0.003 | −0.020 |
| **mean ± std** | **+0.001 ± 0.009** | **+0.005 ± 0.018** |

Neither gap is sign-stable. Detection flips from −0.007 to +0.013 and
back; localization flips from +0.020 down to −0.020 — the same
magnitude, opposite sign, at the two ends of the 4 seeds. Compare to
IEEE-14's own table (§5): detection −0.013/−0.013/−0.011/−0.021 (never
positive), localization +0.013/+0.013/+0.033/+0.013 (never negative).
IEEE-30 looks nothing like that once you look past the first seed —
the first seed (20260921, this project's own primary seed throughout,
used first for exactly that reason) simply happened to land close to
IEEE-14's pattern by chance, in both directions, at once.

**Corrected reading: this is a null result at IEEE-30, not a
corroboration.** The "two standard IEEE test systems now agree with
each other" framing written after the single run was premature — it
was itself an instance of the exact mistake this project's multi-seed
discipline exists to catch, caught this time before it reached the
paper rather than after. The honest three-network picture is: the
localization reversal is confirmed, seed-robust, at exactly one scale
(IEEE-14); it's a clean null (topology-free ablation not needed, but
not beaten either — no confident direction) at the other two (5-bus
shows the opposite, seed-robust direction; IEEE-30 shows no stable
direction at all). Whether IEEE-30's null result means the IEEE-14
reversal really is IEEE-14-specific, or whether IEEE-30 just needs
more than 500 replications per seed to resolve an effect this small
against this network's own noise floor, is genuinely open — a
question for a fourth system or a higher-replication IEEE-30 run, not
something this data settles. `paper/main.tex`, `PAPER_DRAFT.md`, and
`README.md` are all updated to report this corrected finding, not the
single-run one.

## 6. Post-submission methodological audit (2026-09-22): three real bugs found, fixed, and re-run — the paper's numbers are now stale

**This section documents an audit that invalidates every result number currently
written in `paper/main.tex`, `PAPER_DRAFT.md`, and `README.md`.** Those three
files have **not** been updated yet — they still describe the pre-audit numbers
and narrative throughout (title, abstract, all result tables, Discussion,
Conclusion). Treat everything in this repo's paper-facing files as superseded
until a future session rewrites them from the numbers in this section. The
`results/phase2s_hard_*`, `phase2u_*`, `phase2v_ieee14_*`, `phase2w_ieee14_*`,
`phase2x_ieee30_*`, and `phase2y_ieee30_*` CSVs on disk **have** already been
regenerated by the fixed code and reflect the corrected numbers below — so the
repo is currently in a mixed state: code and data are fixed, paper text is not.

**How this started.** A static, code-level (not re-executed) audit of the full
repo -- everything the paper draws on, not just the paper files themselves --
surfaced ten distinct concerns. Each was independently verified against the
actual code (file:line, not taken on trust) before anything was changed. All
ten held up:

1. **IEEE-14/30's "residual" feature was computed as `|vm_hat - vm_true| * 100`**
   (`vm_true` = the simulator's hidden ground-truth state) instead of the
   deployable WLS/chi-square residual `z - h(x_hat)` that the shared `run_wls()`
   function (`21_nonlinear_stealth_fdia_bdd_benchmark_HARD.py:601-612`) already
   computes and returns as `result["norm_residual"]`. The 5-bus pipeline used
   the correct quantity all along (`22_graph_ready_protected_prior_telemetry_HARD.py:398`).
   **Fixed** (below).
2. **Test-set model selection at IEEE-14/30**: a train/validation/test split
   was constructed but "validation" was never read again -- all 3 candidate
   models (LR/RF/HGB) were fit on train and scored directly on test, and the
   best test score was reported as "the" result. **Fixed.**
3. **The same test-set-model-selection bug also existed at 5-bus**, in a more
   disguised form: `run_learned_detection()`/`run_learned_localization()`
   (`23_topology_aware_cyber_physical_localization_HARD.py`) did compute
   `validation_balanced_accuracy`/`validation_top1_accuracy` per model, but
   nothing downstream ever used it to pick a winner -- the saved CSVs kept all
   3 models' rows per feature set, and every consumer (`26_multi_seed_replication.py`,
   `29_paper_figures.py`) took `.max()` over the **test** column. **Fixed.**
4. **The latency benchmark's classifier is trained on random dummy data**
   (`24_realtime_latency_benchmark_FINAL.py:107-114`: `X_dummy = rng.normal(...)`,
   `y_dummy = rng.integers(...)`, `clf.fit(X_dummy, y_dummy)`), and its feature
   computation also uses `vm_true`/`va_true`. The ~16ms state-estimation
   sub-cost is still real (that part times an actual WLS solve), but the
   framing as "full decision pipeline" / "actual detector" latency is not
   accurate as currently measured. **Not yet fixed** -- would need a real
   trained detector on deployable features timed in the same harness.
5. **5-bus's "topology-free" localization baselines were not topology-free**:
   `node_feature_sets()` (`23_topology_aware_cyber_physical_localization_HARD.py:748`)
   prepended `degree` + four `role_*` one-hot flags (`role_pcc`, `role_pv_gfl`,
   `role_load`, `role_bess_gfm`) to *every* feature set, including
   `residual_only`/`prior_only`/`residual_plus_prior`. `degree` is graph
   topology by definition; `role_load` is worse -- the attack generator only
   ever targets load buses (`target_bus = rng.choice(load_buses)`), so handing
   the localizer `role_load` is close to handing it the eligible-target set
   directly, in every feature set including the ones meant to be non-relational.
   **Fixed** -- removed from all four sets (including `topology_fusion`: this
   structural/role metadata was never what this paper means by
   "topology-relational," which is specifically the neighbor/incident-edge
   comparison features).
6. **README reproduction is broken on a clean clone**: the first documented
   command needs `data/phase2e_7day_operating_dataset.csv`, which is
   gitignored (matches the blanket `data/` rule) and not regenerated by
   anything at the documented path -- `exploration/08_multirate_operating_dataset.py`
   has `ROOT = Path(__file__).resolve().parent`, so it writes to
   `exploration/data/`, not root `data/`, which is what 21/22/24/25 actually
   read from. Confirmed directly (both the gitignore match and the path
   mismatch). **Not yet fixed.**
7. **Cross-network SNR/prior mismatch**: `STD_V_PU`/`STD_P_MW`/`STD_Q_MVAR`
   (21_..._HARD.py:77-81) are fixed absolute values shared unscaled across
   5-bus/14-bus/30-bus, and prior forecast-error std is 2.5%
   (`22_...py:220-221`, clipped ±6%) at 5-bus vs. 5% (`28_ieee14_scale_replication.py:172`,
   looser clip) at IEEE-14/30. Confirmed directly. A real confound on any
   cross-network comparison. **Not yet fixed.**
8. **No committed script produces the R² = 0.960/0.959 redundancy numbers**
   quoted in the paper's Discussion. Repo-wide search for `Ridge`/`r2_score`
   found exactly one hit, a comment in `31_ieee30_scale_replication.py`
   referencing the check, not an implementation. The Acknowledgment's "all
   reported quantitative claims are reproducible from the accompanying
   scripts" is not true for these two numbers as things stand. **Not yet
   fixed** -- needs a new script.
9. **Zero-day test protocol is apples-to-oranges**:
   `27_held_out_attack_type_generalization.py:119` evaluates the held-out
   condition on `dataset_split != "train"` (validation+test combined) but the
   seen-in-training reference (line 128) only on `dataset_split == "test"`.
   Confirmed directly. **Not yet fixed.**
10. **GFL/GFM is steady-state P/Q injection, not converter-control dynamics**:
    `exploration/01_microgrid_topology.py:49-50` uses `pp.create_sgen(...,
    name="PV - GFL")` -- a static injection with a descriptive name, no droop
    control or virtual inertia modeled. The paper's GFL/GFM language should
    say "steady-state operating-point model; converter control dynamics are
    not modeled" rather than implying dynamic behavior. **Not yet fixed**
    (wording only, no rerun needed).

Also flagged, not yet acted on: seed sets differ by one value between 5-bus
(`20260812, 42, 777, 2024`) and IEEE-14/30 (`20260921, 42, 777, 2024`) --
three of four match, not all four, so "the same four seeds" language in the
paper is slightly overstated; and several statistical claims ("real effect,"
"statistically indistinguishable") are supported by sign-consistency across 4
seeds, which is good evidence but not a formal paired hypothesis test or CI --
worth softening to "consistent across the four tested seeds" on a future pass.

### Fixes applied and re-run (items 1, 2, 3, 5 above)

**Item 1 (deployable residual).** `measurement_schema()` (phase21) orders all
`n_bus` voltage measurements first, so `result["norm_residual"][:n_bus]` is
exactly the per-bus voltage-measurement residual with no oracle information --
same deployable quantity the 5-bus pipeline already used, no new machinery
needed. Applied to `28_ieee14_scale_replication.py` and
`31_ieee30_scale_replication.py` (identical fix, both scripts share this bug
since 31 is a copy of 28).

**Item 2/3 (validation-based model selection).** IEEE-14/30: added a
validation-scored selection step in the detection loop, so exactly one
(validation-picked) model's row is reported per feature set instead of the
max of three test-scored rows. 5-bus: added
`.sort_values("validation_balanced_accuracy"/"validation_top1_accuracy",
ascending=False).drop_duplicates(subset="feature_set", keep="first")`
immediately after `run_learned_detection()`/`run_learned_localization()`
return, in both `23_..._HARD.py` and `23_..._HARDER.py`, for both detection
and localization.

**Item 5 (degree/role removal).** Removed `base_structural` (degree + 4 role
flags) from all four sets in `node_feature_sets()`, both `_HARD.py` and
`_HARDER.py`.

**Re-run, all three networks, same discipline as everywhere else in this
project (primary seed first, then the same 4 seeds):**

**5-bus, HARD, primary seed (20260812) -- before → after:**
| | Detection | Localization |
|---|---|---|
| topology_fusion | 0.950 → 0.950 (unchanged) | 0.980 → **0.9667** |
| residual_plus_prior | 0.950 → 0.950 (unchanged) | 0.993 → **0.9667** |

Detection was already correct (HistGradientBoosting was already the genuinely
best model for both feature sets, so validation-selection changed nothing).
Localization's old "residual_plus_prior wins" result is gone: both feature
sets now score **identically** at 0.9667. HARDER condition: identical pattern
(0.9533=0.9533 detection, 0.9667=0.9667 localization) -- the tie is not a
one-condition fluke.

**5-bus, 4-seed (`26_multi_seed_replication.py`), gap = topology_fusion −
residual_plus_prior:**
| Seed | Detection gap (before → after) | Localization gap (before → after) |
|---|---|---|
| 20260812 | −0.013 → +0.000 | −0.013 → **+0.000** |
| 42 | (n/a, old run used different seeds) → +0.003 | → **−0.033** |
| 777 | → −0.007 | → **+0.053** |
| 2024 | → +0.017 | → **+0.000** |
| mean±std | −0.003±0.010 → +0.003±0.010 | −0.003±0.007 → **+0.005±0.036** |

Detection stays null (sign-unstable both before and after -- consistent).
**Localization's old finding ("residual_plus_prior consistently, if narrowly,
ahead -- never worse across 4 seeds") does not survive**: the gap is now pure
noise around zero, sign flipping seed to seed (−0.033 to +0.053, two exact
ties). This is consistent with the redundancy-check's own R²=0.960 finding
(topology-relational information is near-linearly reconstructable from
non-relational features at 5-bus) in a way the old, asymmetric result never
quite was -- a real tie is the mechanistically expected outcome of genuine
redundancy; a small, one-sided, unexplained edge for the ablation was always a
bit of a loose thread.

**IEEE-14, primary seed (20260921), gap = topology_fusion − best
non-relational alternative -- before → after:**
| | Detection | Localization |
|---|---|---|
| gap | −0.009 → **−0.017** | +0.013 → **0.000** (exact tie) |

Also notable: `residual_only`'s detection balanced accuracy **dropped from
0.750 to 0.630** once it used the real residual instead of the oracle one --
the theoretically expected direction (model-consistent stealth attacks are
specifically designed to keep the true WLS residual small), and independent
evidence the fix is doing what it's supposed to, not just producing different
numbers.

**IEEE-14, 4-seed (`30_ieee14_multi_seed_replication.py`) -- before →
after:**
| Seed | Detection gap | Localization gap |
|---|---|---|
| 20260921 | −0.013 → **−0.017** | +0.013 → **+0.000** |
| 42 | −0.013 → **+0.003** | +0.013 → **+0.047** |
| 777 | −0.011 → **+0.007** | +0.033 → **+0.020** |
| 2024 | −0.021 → **+0.014** | +0.013 → **+0.007** |
| mean±std | −0.015±0.004 → **+0.002±0.013** | +0.018±0.010 → **+0.018±0.021** |

Detection: the old "topology consistently trails" result does not survive --
sign now flips (1 of 4 negative, 3 of 4 positive), mean near zero. This makes
IEEE-14 detection look like every other network's detection result (null), a
*more* consistent three-network detection story than before, not less.
Localization: the mean gap is, remarkably, almost unchanged (+0.018 both
times) and still never goes negative across the 4 seeds -- but std roughly
doubled (0.010 → 0.021) and one seed is now an exact tie rather than clearly
positive, so this finding survives in direction but with weaker precision
than previously claimed.

**IEEE-30, 4-seed (`32_ieee30_multi_seed_replication.py`) -- before →
after:**
| Seed | Detection gap | Localization gap |
|---|---|---|
| 20260921 | −0.007 → **+0.083** | +0.020 → **−0.007** |
| 42 | +0.000 → **+0.040** | +0.007 → **+0.000** |
| 777 | +0.013 → **+0.020** | +0.013 → **+0.007** |
| 2024 | −0.003 → **+0.013** | −0.020 → **+0.033** |
| mean±std | +0.001±0.009 → **+0.039±0.032** | +0.005±0.018 → **+0.008±0.018** |

This is the biggest surprise in the whole audit. Localization is unchanged in
character (null, sign-unstable, both before and after). **Detection is not**:
the old null result (sign flipping, mean ≈ 0) is replaced by a real, 4-seed-
robust *advantage* for `topology_fusion` -- every seed positive, mean +0.039,
nearly 4× the magnitude of IEEE-14's own (now-null) detection gap. This did
not exist in any previous version of this project's results, in either
direction.

### The honest picture right now (provisional -- items 4/6/7/8/9/10 above are
still open, and any of them could still move these numbers again)

- **Detection**: null at 5-bus, null at IEEE-14, a real and 4-seed-consistent
  *advantage* at IEEE-30 (mean +0.039, never negative). This specific pattern
  is new to this audit; nothing before it showed a detection advantage
  anywhere.
- **Localization**: null at 5-bus (new -- the old small residual_plus_prior
  edge dissolved), a real but now less precisely estimated advantage at
  IEEE-14 (mean +0.018, never negative, std doubled), null at IEEE-30
  (character unchanged by the fix).

A speculative, explicitly-unverified reading: once the "cheap" local
residual/prior signal is denied to a classifier (by using an oracle-free
residual that a well-designed stealth attack is specifically built to evade),
relational/neighbor-comparison information becomes relatively more valuable
-- and that may show up differently by task and by how much redundant
neighbor structure a given network provides, rather than as a single
"topology helps here, not there" rule. This is not tested, only offered as
one candidate explanation for why the surviving advantages are task-specific
(IEEE-14: localization) and network-specific (IEEE-30: detection) rather than
uniform.

**What this means for the current paper files**: `paper/main.tex`,
`PAPER_DRAFT.md`, and `README.md` describe a three-network story (uniform
detection null; localization null at 5-bus/IEEE-30, real advantage at
IEEE-14) that was accurate for the pre-audit numbers but is no longer
consistent with the corrected data above (specifically: IEEE-30 detection is
no longer null). None of those files have been touched in this audit pass --
rewriting them (title, abstract, Results IV, Table III/IV/V, Discussion,
Conclusion, Limitations, figures) is explicitly left for a future session,
once items 4/6/7/8/9/10 are either also fixed or deliberately deferred with
their own honest caveats.

### Update, same day: items 4, 6, and 9 also fixed and verified; item 8's
### script now exists

Continuing directly from the fixes above, in the same session.

**Item 6 (README reproduction path) -- fixed and verified end-to-end.**
`exploration/08_multirate_operating_dataset.py` had `ROOT = Path(__file__)
.resolve().parent` used for *both* finding its sibling script
(`01_microgrid_topology.py`, correctly in `exploration/`) *and* its output
paths (incorrectly `exploration/data/` instead of root `data/`, which is
what 21/22/24/25 actually read). Split into `ROOT` (unchanged, still used
for `load_module()`) and a new `PROJECT_ROOT = ROOT.parent` for
`DATA`/`RESULTS`/`FIGURES`. Re-running it after the fix hit a second,
unrelated bug (`load_a_p[spike_start:spike_stop] += 0.020` raised
`TypeError: Index does not support mutable operations` -- `load_multiplier()`'s
return value is apparently a pandas `Index`, not a plain array, in the
currently-installed pandas/numpy combination), fixed by wrapping the load
arrays in `np.asarray(..., dtype=float)`. With both fixed, the generator now
runs to completion and writes to the correct root-level `data/`/`results/`/
`figures/` -- confirmed by timestamp and by checking the saved file's path
directly, not just exit code. `RNG = np.random.default_rng(20260811)` is a
fixed seed, so this is a real, deterministic regeneration, not new data
drawn from thin air (the existing `data/phase2e_7day_operating_dataset.csv`,
dated Aug 11 -- the same date as its own seed -- was almost certainly
produced by an earlier, differently-invoked run of this same generator, not
manually placed). Also added the missing first step to `README.md`'s
reproduction command list.

**Item 9 (zero-day protocol asymmetry) -- fixed.**
`27_held_out_attack_type_generalization.py`'s held-out condition evaluated
on `dataset_split != "train"` (validation+test) while the seen-in-training
reference evaluated on `dataset_split == "test"` only; changed the held-out
condition to `== "test"` too, so both conditions' recall numbers share the
same evaluation-set composition. Also fixed a second asymmetry noticed while
looking at this file: its classifier was a bare, default-hyperparameter
`HistGradientBoostingClassifier`, different from the tuned configuration
(`max_iter=300, learning_rate=0.06, max_leaf_nodes=15, l2_regularization=1.0`)
that `23_topology_aware_cyber_physical_localization_HARD.py` actually reports
as "the" 5-bus detector -- now matched, so the zero-day result describes the
same detector's generalization, not an unrelated, unconfigured one.

**Re-run, and the finding survives essentially unchanged.** Both
`topology_fusion` and `residual_plus_prior`, both held-out attack types,
evaluated on the now-symmetric test-only split (n=75 per condition, half
the old validation+test n since only "test" counts now):

| Feature set | Held-out type | Held-out recall | Seen-in-training recall |
|---|---|---|---|
| topology_fusion | naive | 0.000 | 0.907 |
| topology_fusion | model-consistent | 0.013 | 1.000 |
| residual_plus_prior | naive | 0.000 | 0.907 |
| residual_plus_prior | model-consistent | 0.013 | 1.000 |

0-1.3% held-out recall vs. 91-100% seen-in-training -- matches the paper's
existing "0-4% recall, vs. 89-100% when that type is in training" claim
almost exactly, despite fixing both the evaluation-split asymmetry and the
classifier-configuration mismatch. Unlike items 1/2/3/5 above, this is one
of the few audit findings that does **not** change the paper's story at
all -- the zero-day generalization failure was real, and remains real,
under a properly matched comparison.

**Item 4 (latency benchmark) -- fixed and verified the timing conclusion is
unchanged.** `24_realtime_latency_benchmark_FINAL.py`'s classifier used to be
fit on `X_dummy`/`y_dummy` (pure random noise and random labels); its
feature computation used `vm_true`/`va_true` for both the "residual" and
"innovation" quantities (the latter not previously flagged explicitly, found
while reading the file to fix the former -- comparing the WLS estimate to
ground truth and calling it "innovation" has the same oracle-leakage problem
as the residual one, and is inconsistent with how "innovation" is defined
everywhere else in this project, i.e. estimate-vs-independent-prior, not
estimate-vs-truth). Fixed by: (a) running 150 warm-up trials through the
same simulated pipeline (mix of clean/attack scenarios) to get real
(feature, label) pairs to train on, instead of noise; (b) computing the
residual as `z - h(x_hat)` (matching the fix already applied to
28/31_*.py); (c) building a genuinely independent `prior_net` (same context,
independently-drawn 2.5% load-forecast error, matching the 5-bus
convention) once, and using its state for the innovation comparison instead
of `va_true`.

Re-ran the benchmark twice -- once (by mistake) while the IEEE-14
regeneration job above was still running in the background, and once in
true isolation as this script's own header comment requires ("no other
heavy background jobs"). The contaminated run showed clearly elevated tail
latency (p95 20.1ms, p99 24.1ms, both over budget, max 89.5ms) --
**discarded**, not reported, specifically because it violates this
project's own established discipline about not trusting a measurement taken
under conditions the method itself warns against. The isolated re-run:

| Stage | median | p95 | p99 | max |
|---|---|---|---|---|
| WLS state estimation | 16.63 | 17.31 | 18.11 | 81.10 |
| Feature computation | 0.087 | 0.090 | 0.096 | 0.125 |
| Model inference | 0.192 | 0.214 | 0.220 | 0.262 |
| **TOTAL** | **16.91** | **17.59** | **18.41** | **81.41** |

Median/p95/p99 all within the 20ms budget -- essentially unchanged from the
paper's existing 16.4/17.0/17.1ms claim, exactly as predicted before
re-running it: `pred_ms` and `feat_ms` are wall-clock time for a
fixed-size computation (dot-product-plus-sigmoid; mean/max/subtraction over
a fixed-length array respectively), which does not depend on whether the
classifier was meaningfully trained or whether the residual used real vs.
oracle values -- only on the array/model *size*, which is unchanged. The
single-outlier max jumped from 21.3ms (paper) to 81.4ms (this run) -- also
not attributable to the fix (nothing touched by it affects `wls_ms`
timing); almost certainly ordinary tail-latency variance in the state
estimator's own solve time, the same phenomenon the paper's own text already
attributes the (smaller) original outlier to. **The corrected script closes
the "not a real detector" objection with the timing conclusion intact** --
warm-up training balanced accuracy on real data was 1.000 (150 scenarios,
83 attack/67 clean), confirming the classifier is now actually detecting
something, not just exercising a random weight vector.

**Item 8 (missing R² redundancy script) -- written, run, and it holds up.**
New script, `33_feature_redundancy_diagnostic.py`: 5-fold cross-validated
out-of-fold Ridge regression (alpha=1.0), predicting each explicit
topology-relational column from the full non-relational (`residual_plus_prior`)
feature vector, matching the paper's own described method as closely as
possible. Reuses each network's own feature-set-defining function
(`feature_columns()` from `23_..._HARD.py` for 5-bus; a pivoted version of
`28_ieee14_scale_replication.py`'s per-node relational columns for IEEE-14)
rather than re-deriving column lists independently, specifically to avoid
building a second, independent version of the exact feature-leakage bug
this whole audit is about.

| | 5-bus (new, reproducible) | 5-bus (paper, uncommitted) | IEEE-14 (new, reproducible, primary seed 20260921) | IEEE-14 (paper, uncommitted) |
|---|---|---|---|---|
| mean R² | 0.950 | 0.960 | 0.966 | 0.959 |
| median R² | 1.000 | 1.000 | 0.999 | 0.996 |
| min R² | 0.390 | 0.652 | 0.677 | (not stated) |
| % above 0.8 | 87.7% | 91% | 91.7% | 96% |

Close to, but not identical to, the paper's existing (previously
unreproducible) numbers -- expected, since those were never backed by
committed code and were almost certainly computed with different exact
settings (fold count/alpha/random state all unspecified in the paper text).
**The qualitative conclusion the paper actually rests on is intact and, if
anything, slightly stronger**: both networks show high linear redundancy
(mean/median both ~0.95-1.00), and IEEE-14 is not less redundant than
5-bus -- 0.966 vs. 0.950, IEEE-14 slightly *higher* now, reinforcing rather
than weakening the Discussion's "redundancy hypothesis does not explain the
[localization] reversal" finding. The paper's specific 0.960/0.959 numbers
should be replaced with 0.950/0.966 (or whatever this script reports at the
time of the eventual rewrite) so the Acknowledgment's reproducibility claim
is actually true for these two numbers.

**Still open**: item 7 (SNR/prior mismatch) -- no code changes made;
documented as a limitation to state honestly in the eventual rewrite rather
than a redesign-and-rerun, since standardizing the noise model across three
very differently-scaled networks is a real experimental-design decision, not
a bug fix, and re-running all three networks a third time was judged not
worth it for a robustness question the paper can instead disclose directly.
Item 10 (GFL/GFM wording) and the statistical-language precision note are
also still wording-only, deferred to the same rewrite.

## Positioning against related work

[arXiv:2605.17256](https://arxiv.org/pdf/2605.17256) (2026,
"Latency-Aware Deep Learning Benchmark for Real-Time Cyber-Physical
Attack and Fault Classification in Inverter-Dominated Power Grids")
independently reports the same category of gap this project set out
to close: 8 architectures (MLP–Transformer), trained on data from an
industrial-grade EMT simulator, all reach sub-cycle (<15ms)
*classification* time, yet measured end-to-end *pipeline* latency is
50–90 ms — "a critical gap between algorithmic capability and
protection-grade deployment," for which the authors call for further
optimization and hardware acceleration, leaving it open.

This project reproduces the same qualitative pattern independently
(a ~3ms classifier embedded in a ~43ms pipeline) on a different,
smaller testbed, then profiles the pipeline itself rather than
treating the gap as inherent. The dominant cost was an avoidable
software inefficiency — a measurement table rebuilt row-by-row every
cycle through a general-purpose element-creation API — not the
estimation or learning algorithms. Fixing it (pure software, commodity
hardware, no algorithmic changes) closed the reported-style gap from
2.15x-over-budget to within-budget at the median/p95/p99. This
suggests at least part of the gap reported in this line of work may be
implementation overhead rather than an irreducible property of the
algorithms being benchmarked — worth checking before reaching for
hardware acceleration.

Other located context (not deeply verified, cited for scope only):
protection challenges in renewable-integrated microgrids are an active
review topic (Energy Informatics, 2026); PMU-based ML fault
classification for low-inertia systems reports 85–97% accuracy
depending on synthetic vs. physical data (*Mathematics*, 2026);
FDIA detection via CNN-LSTM and multi-sensor fusion is an active
line of work in the smart-grid cybersecurity literature generally.

## Limitations (explicit, for anyone deciding whether to write this up)

1. **Detection shows no topology-specific advantage at any of the
   three scales tested, confirmed across 4 seeds at each** (§2.5, §5,
   §5.6), using two unrelated feature-engineering implementations.
   **Localization is different, and network-specific rather than
   uniform: a real, 4-seed-confirmed reversal at IEEE-14 only** — the
   topology-free ablation is never worse at 5-bus across the 4 seeds,
   with only a small mean advantage (−0.003±0.007, not the
   "decisive"-looking 0.993 vs. 0.980 the single primary-seed numbers
   alone suggest), topology-fusion
   keeps a comparably small but consistent lead at IEEE-14 (§5), and a third
   network (IEEE 30-bus, §5.6), re-tested across the same 4 seeds
   after an initial single run looked like it matched IEEE-14, shows
   no stable direction either way. Whether that makes IEEE-14 the
   outlier or means IEEE-30 needs more replications to resolve a small
   effect is open. All three are still limited to synthetic
   networks/loads; a real feeder would test generalization beyond
   standard synthetic test cases entirely.
2. **No zero-day generalization**: demonstrated directly in §4 — 0–4%
   recall on a completely withheld attack type. The detection/
   localization numbers above only hold for attack types represented
   in training. This was tested for `topology_fusion` specifically,
   not independently re-run for `residual_plus_prior`.
3. **Third stress lever not testable as a free parameter**: restricting
   which measurements a stealth attack compromises is a physical
   consequence of the AC model consistency construction, not a knob —
   noted, not evaded.
4. **Attack family**: only two synthetic families (naive,
   model-consistent FDIA) exist in this data lineage. Phase 14/17's
   richer taxonomy lives in an incompatible upstream pipeline (§4) —
   bridging it is future work, not done here.
5. **Latency benchmark** measures this specific Python/pandapower
   implementation on one machine — not a hardware/RTOS claim — and was
   not re-run on the smaller `residual_plus_prior` feature set (§3
   argues this would not change the conclusion, since state estimation
   dominates the budget regardless of feature set, but this is an
   argument, not a re-measurement).

## New files this session

`21_..._HARD.py`, `22_..._HARD.py` / `_HARDER.py`,
`23_..._FIXED.py` / `_HARD.py` / `_HARDER.py`,
`24_realtime_latency_benchmark.py` / `_WARMSTART.py` / `_OPTIMIZED.py` /
`_FINAL.py`, `25_wls_overhead_diagnostic.py`,
`26_multi_seed_replication.py`, `27_held_out_attack_type_generalization.py`,
`28_ieee14_scale_replication.py`.
All outputs under
`*_hard`/`*_harder`/`*_fixed`/`*_optimized`/`*_final`-prefixed filenames
in `data/`, `results/`, `figures/` — nothing original overwritten (two
accidental overwrites of un-prefixed Phase 2R diagnostic files happened
mid-session from incomplete `sed` renames; both were caught and
restored immediately by re-running the original, unmodified,
same-seed script, then the HARD/HARDER scripts were fixed so it can't
recur).

## Next, if this becomes a write-up

1. ~~Push `28_ieee14_scale_replication.py` to n_rep≈500~~ **Done** — §5
   is now confirmed at a matched 300-scenario test-set size, not a
   pilot. ~~Build the `residual_plus_prior` ablation and re-check
   whether the 5-bus topology advantage is real~~ **Done (§2.5)** —
   it wasn't; corrected across 2 stress conditions and 4 seeds.
   ~~Audit whether IEEE-14 has the same feature-definition bug~~
   **Done (§5)** — it did, independently; fixed the same way. ~~Confirm
   the narrow IEEE-14 localization margin across seeds, not just one
   run~~ **Done (§5)** — real, 4-seed-confirmed reversal (topology
   wins at 14-bus for localization specifically, loses everywhere
   else). ~~Try a third standard system (IEEE 30-bus), and confirm it
   across the same 4 seeds before trusting a single run~~ **Done,
   §5.6** — `31_ieee30_scale_replication.py` then
   `32_ieee30_multi_seed_replication.py`. The single run looked like a
   match; the 4-seed replication shows neither gap is sign-stable at
   IEEE-30, unlike IEEE-14 — a null result, not a corroboration. The
   localization reversal is now confirmed at exactly one of three
   tested scales, not a general standard/meshed-network property.
2. Some form of unseen-attack robustness beyond pure supervised
   classification (e.g. anomaly-based pre-filter, or bridging Phase
   14/17's richer taxonomy) — §4 shows the current approach has none,
   and that's worth addressing rather than only disclosing.
3. A direct numeric comparison against arXiv:2605.17256's own
   architectures on the same data, if their code/data is available.
4. ~~Re-run §4's zero-day test on `residual_plus_prior` specifically~~
   **Done** — collapses identically to `topology_fusion`; also caught
   a stale seed=2024 data snapshot in the process (corrected numbers
   now in §4).
5. ~~Run §2.5's out-of-fold redundancy check on IEEE-14's own
   features, to test whether lower redundancy explains the
   localization reversal~~ **Done — result: it doesn't.** Mean
   $R^2$ = 0.959 (median 0.996, 96% of 84 node×column pairs above 0.8)
   predicting each node's relational features from all 14 nodes' local
   residual/innovation values — statistically indistinguishable from
   5-bus's 0.960, not lower. The "less redundant at scale" hypothesis
   in `paper/main.tex` §VIII's Discussion was wrong; corrected there to
   report this directly instead of leaving it as untested speculation.
   Current best guess (also now in §VIII, explicitly flagged as a
   guess, not verified): the redundancy is present at both networks, but
   whether a tree-based classifier with no built-in graph structure
   can actually *find* the right subset of a flat, unstructured
   28-column input to combine for each node may get harder as the
   input gets less structured/bigger (14 nodes vs. 5), even when the
   information is linearly there — an open, still-untested question
   about model discoverability, not information content.
6. **Post-submission audit (§6) — code and data fixed, paper files not
   yet rewritten.** Three real bugs found and fixed (oracle-leaked
   residual, test-set model selection, non-topology-free 5-bus
   baselines), all three networks re-run. Before rewriting
   `paper/main.tex`/`PAPER_DRAFT.md`/`README.md` to match §6's
   numbers, still open:
   - Fix the latency benchmark's dummy-trained classifier + oracle
     features (§6 item 4) and re-measure, or explicitly scope the
     existing number as "state-estimation sub-cost only."
   - Fix the README's broken clean-clone reproduction path (item 6):
     either move `exploration/08_multirate_operating_dataset.py`'s
     output to root `data/`, or document running it from the right
     directory.
   - Standardize or explicitly document the cross-network SNR/prior-
     uncertainty mismatch (item 7) before treating any 5-bus-vs-IEEE
     comparison as clean.
   - Write the missing out-of-fold Ridge/R² redundancy script (item
     8) so the paper's 0.960/0.959 numbers are actually reproducible,
     and consider re-running it on the now-corrected 5-bus features.
   - Fix the zero-day protocol's validation+test-vs-test-only asymmetry
     (item 9) and re-check the held-out-attack-type result.
   - Soften GFL/GFM language (item 10) and the strongest statistical
     claims (sign-consistency across 4 seeds, not a formal test) —
     wording only, no rerun needed.
   Once those are resolved (fixed or deliberately deferred with an
   honest caveat), the entire back half of the paper needs rewriting
   to match §6's numbers — this is not a wording pass like the
   previous rounds, the actual three-network finding has changed.
