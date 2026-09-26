# GRIDRA Cyber-Physical Microgrid — Status

Last updated: 2026-09-24 (final documented latency rerun added; §6
still contains the authoritative post-audit model/feature results). Supersedes the "Current backbone / Next"
section in `README.md`, which only described the Aug-11 starting point
and was never updated through phases 2B–2T.

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
inefficiency, not the classifier), two documented independent N=300
reruns on Apple arm64 / macOS 15.6.1 (Python 3.12.4,
pandapower 2.14.10, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2),
both with 0 convergence failures, give 16.155-17.015 /
16.429-18.211 / 16.952-19.190 ms across the median/p95/p99 and
89.067-92.657 ms maxima. Thus median, p95, and p99 remain within the
20 ms target in both reruns, while rare solver-tail events exceed it.
Stage medians are dominated by WLS state estimation
(15.855-16.664 ms) rather than residual/feature/inference work.

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
§6, describes the pre-audit numbers — SUPERSEDED, kept only as the
research log's chronology.** §6 documents a post-submission code-level
audit (2026-09-22) that found and fixed three real bugs (oracle-leaked
"residual" features at IEEE-14/30, test-set model selection at all
three networks, non-topology-free baselines at 5-bus) and re-ran all
three networks. The corrected picture is materially different — most
notably, IEEE-30 detection goes from null to a gap positive in all 4
tested seeds, and 5-bus localization goes from a small
residual_plus_prior edge to a clean null. §6 was itself extended by a
fourth review round (2026-09-23, §6's second dated update) that found
and fixed a further localization-protocol mismatch, a non-end-to-end
latency timer, a stale Fig. 4 data source, a sign-test miscalculation,
and softened this project's own "real"/"genuine" advantage language
once the n=4-seed statistical caveat was worked out precisely — read
that update too, not just the first one. **`paper/main.tex`,
`PAPER_DRAFT.md`, and `README.md` are, as of the latest commit, kept in
sync with §6 in full** (they were not, for a period after §6 was first
written — that gap is closed). Read all of §6, in order, before
trusting any specific number anywhere else in this file or in the
paper files; nothing before §6 should be cited as this project's
current result.

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

## 6. Post-submission methodological audit (2026-09-22 onward, still the current section): three real bugs found, fixed, and re-run, then two further review rounds' worth of fixes

**Historical state at the start of this audit — superseded by the updates
later in this section.** The paragraph immediately below, and the
"paper's numbers are now stale" framing it describes, was true only in
the narrow window between when this audit began and when the paper
files were rewritten to match it (documented later in this same
section, under "the full rewrite is done" and the two further dated
updates after it). As of the latest commit, `paper/main.tex`,
`PAPER_DRAFT.md`, and `README.md` **are** kept in sync with this
section's corrected numbers — read the whole section, in order, rather
than stopping at this paragraph.

**This section documents an audit that invalidated every result number
written in `paper/main.tex`, `PAPER_DRAFT.md`, and `README.md`, at the
time it was written.** Those three
files had **not** been updated yet at that point — they still described the pre-audit numbers
and narrative throughout (title, abstract, all result tables, Discussion,
Conclusion). The `results/phase2s_hard_*`, `phase2u_*`, `phase2v_ieee14_*`, `phase2w_ieee14_*`,
`phase2x_ieee30_*`, and `phase2y_ieee30_*` CSVs on disk **had** already been
regenerated by the fixed code and reflected the corrected numbers below — so the
repo was, at that point in the audit, in a mixed state: code and data fixed, paper text not yet.

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

### Update, same day: the full rewrite is done

`paper/main.tex`, `PAPER_DRAFT.md`, and `README.md` have all been rewritten to
report the numbers and finding above as the paper's actual, current story --
title kept (it was already neutral to whichever specific pattern emerged),
but abstract, every Results section, Discussion, Limitations, and Conclusion
are new prose, not edits layered on the pre-audit version. Table III/IV/V (or
their Markdown equivalents) report the corrected 4-seed means and per-seed
gaps directly from `results/phase2u_multiseed_summary.csv`,
`phase2w_ieee14_multiseed_summary.csv`, and `phase2y_ieee30_multiseed_summary.csv`.
All four figures were regenerated against the corrected result files
(`29_paper_figures.py`, including two stale hardcoded strings caught and
fixed in the process: Fig. 4's suptitle and a code comment still described
the pre-IEEE-30-detection-advantage finding, and Fig. 2's latency waterfall's
final bar still showed the pre-real-classifier-fix number). `paper/main.pdf`
compiles clean (12 pages, no undefined references, no new overfull boxes
beyond one pre-existing 1.98pt one unrelated to this content). One drafting
error was caught and fixed before finalizing: Table III's bold-leader
markers were placed on the wrong cell (or left on a cell whose gap sign
actually flips across seeds) for the IEEE-14-detection and IEEE-30-
localization rows -- both are correctly unbolded now, matching the rule
stated in the table's own footnote.

Deliberately not done in this pass: items 4/6/7/8/9/10's own *wording*
subtleties (GFL/GFM phrasing, statistical-language precision, the SNR/prior
mismatch) are folded into the rewritten prose directly rather than tracked
as separate open items any more -- see the Limitations section of the
rewritten paper itself for how each is now stated. What remains genuinely
open, and is stated as such in the paper: whether the IEEE-30 detection
advantage is explained by feature redundancy (untested -- no detection-level
redundancy check has been built for IEEE-30, unlike the localization check
already run at 5-bus/IEEE-14), and whether a fourth or fifth network would
extend, contradict, or add a third pattern to the current
"each standard IEEE system favors a different task" result.

### Update, 2026-09-23: fourth review round — statistical language, localization protocol, Fig. 4 data source, and a stale Table 1 caught while re-verifying

A fourth review pass of the rewritten paper found four more issues, two
flagged critical. All four are now addressed; a fifth, unrelated issue
surfaced while re-verifying the fix for the third one below, and is
also now fixed.

1. **Statistical overclaim.** "Real"/"genuine" advantage language for the
   IEEE-30 detection and IEEE-14 localization findings is not defensible at
   n=4 seeds: a standard t-based 95% CI crosses zero for both
   ($+0.039\pm0.050$ and $+0.018\pm0.033$), and a distribution-free sign
   test on 4 same-signed seeds gives one-sided $p=0.5^4=0.0625$, short of
   0.05. Added an explicit caveat paragraph and a new Limitations item to
   `paper/main.tex`, and swept "real"/"genuine advantage" language throughout
   (Abstract, Contributions, Results IV, Discussion, Fig. 4 caption/suptitle,
   Conclusion) to "positive/never-negative in all four tested seeds" /
   "consistent in sign." Mirrored into `PAPER_DRAFT.md` and `README.md`.
   Legitimate non-statistical uses ("a real bug," "a genuinely topology-free
   ablation") were left alone.

2. **Latency timer wasn't fully end-to-end.** `total_ms` in
   `24_realtime_latency_benchmark_FINAL.py` summed only 3 of 5 timed stages,
   silently excluding the post-state-estimation residual/innovation calc and
   the feature-array-assembly gap. Fixed by timing and summing all 6 stages
   (`wls`, `poststate_residual_innovation`, `feat`, `assembly`, `pred`,
   `TOTAL (t5-t0, nothing excluded)`). Three earlier attempts this session
   were contaminated by unrelated system load (two zombie Node.js processes
   from an unrelated project, then Steam, both killed with explicit user
   permission) and deferred rather than forced.

   **Re-measured successfully once the system settled** (load average down
   to 3.4 from the earlier 6.4-7.3, confirmed via `ps aux` immediately
   before each run): **11.64ms median, 12.43ms p95, 12.57ms p99, 65.01ms
   max** — well within the 20ms budget except the single-outlier max, a
   materially *lower* total than the pre-fix 16.9ms despite the fix adding,
   not removing, timed work; the two newly-included stages are negligible
   (well under 0.1ms combined), so this is read as ordinary run-to-run
   wall-clock variance (a different quantity from the bit-for-bit numerical
   determinism verified in item 3 — timing is never claimed to be
   deterministic), not a hidden bug. Given how much the total moved, ran two
   more independent 300-trial repeats to check: medians 11.64/11.69/11.64ms
   and p95s 12.43/12.39/12.64ms across all three agreed to within 5%, but
   the tail did not — max was over budget in all three (65-69ms, same
   outlier signature each time) while p99 was within budget in two of three
   and over it in the third (36.2ms), i.e. whether a rare slow-solve trial
   lands in the top 1% of 300 is itself a coin flip. Reported the most
   recent run as the headline number and added a sentence describing this
   cross-run tail variability explicitly, rather than picking whichever run
   looked best. Updated: Abstract, §5/Results II prose, Fig. 2's waterfall
   final bar (16.91→11.64), `PAPER_DRAFT.md`, `README.md`. Figures
   regenerated.

3. **5-bus localization protocol mismatch.** `run_learned_localization()` in
   `23_topology_aware_cyber_physical_localization_HARD.py` trained (and
   validated/test-selected) on *all* train-split nodes, including
   clean/physical-disturbance scenarios where every node is a negative —
   unlike `28_ieee14_scale_replication.py`/`31_ieee30_scale_replication.py`,
   which only ever train on cyber-scenario nodes, despite a code comment
   claiming the two protocols already matched. `rank_localization_metrics()`
   already filtered to cyber scenarios internally for the *evaluation*
   metric, so reported top-1/top-2 numbers were never computed over
   non-cyber rows — only the *training* data composition was wrong. Fixed by
   adding `node = node[node["is_cyber_graph"] == 1].copy()` as the first
   line of `run_learned_localization()`, in both `23_..._HARD.py` and
   `23_..._HARDER.py` (the latter currently unused by any other script, but
   fixed for consistency). Verified IEEE-14/IEEE-30 already filter correctly
   (`cyber_nodes = node_df[node_df["is_cyber"] == 1]` before the train/test
   split) — no change needed there, no rerun needed there.

   Re-ran the full 4-seed 5-bus replication (`26_multi_seed_replication.py`,
   which also picked up `residual_only` tracking added this round — see item
   4 below) with the fix applied: detection gap $+0.008\pm0.012$ (range
   $-0.010$ to $+0.017$, was $+0.003\pm0.010$), localization gap
   $-0.007\pm0.016$ (range $-0.020$ to $+0.013$, one exact tie; was
   $+0.005\pm0.036$, two exact ties). **Conclusion unchanged**: both still
   null/sign-unstable, matching the paper's existing 5-bus story.

   The detection gap's mean also moved slightly despite the fix touching
   only the localization code path. Verified this is not a bug introduced
   here: (a) `22_..._HARD.py`'s data generation is bit-identical across two
   runs at the same seed (MD5-checked); (b) `23_..._HARD.py`'s detection
   output is bit-identical given fixed input data, checked twice; (c) the
   pre-fix and post-fix versions of `23_..._HARD.py` (via `git show` on the
   immediately prior commit) produce byte-identical detection output on
   identical cached input data. The discrepancy versus the previously-written
   numbers predates this round and was not chased further (small, within
   noise) — except where it also surfaced in Table 1, below.

4. **Fig. 4's 5-bus data source.** `29_paper_figures.py` read
   `phase2s_hard_cyber_detection_metrics.csv`/`..._localization_metrics.csv`
   directly for the 5-bus panel — these get overwritten by
   `26_multi_seed_replication.py`'s *last* seed (2024), so Fig. 4 was
   silently showing one seed's numbers (0.970/0.953), not the true 4-seed
   mean (0.945/0.938) that Table III correctly used. Root cause of why this
   went unnoticed: `26_...py`'s `extract_headline()` never tracked
   `residual_only`, so Table III's `res.` column for 5-bus had to keep a
   single-primary-seed asterisked fallback, and nobody had switched the rest
   of Fig. 4's 5-bus data source over to the 4-seed file while that asterisk
   was still needed for one column. Fixed both: added `residual_only`
   tracking to `extract_headline()`, then pointed Fig. 4's 5-bus panel at
   `phase2u_multiseed_replication.csv` (computing means directly, same
   pattern already used for the IEEE-14/30 panels) instead of the stale
   single-seed file. The asterisk/footnote in Table III is gone; all four
   `res.` cells are now genuine 4-seed means.

5. **A stale Table 1 found while re-verifying item 3.** While confirming the
   localization-protocol fix didn't touch detection (the byte-for-bit checks
   in item 3), re-ran Table 1's "Hard" condition fresh and found its claimed
   exact tie on detection (0.950 vs. 0.950) does not reproduce from the
   current pipeline at all (current: 0.950 vs. 0.937) — a discrepancy that
   predates this round (same byte-for-bit reasoning as item 3: not caused by
   this round's fix) but had never been caught. Regenerated both the "Hard"
   and new "Harder" (+2× noise) rows fresh at the primary seed: detection
   0.950/0.937 (Hard) and 0.950/0.947 (Harder); localization 0.967/0.967
   (Hard, still an exact tie) and 0.960/0.967 (Harder). Rewrote Table 1's
   surrounding prose to describe this accurately (a small, direction-
   inconsistent single-seed gap, not a clean tie) and pointed the reader at
   the 4-seed replication immediately below as the test that actually
   matters — neither column is bolded, matching the existing rule.

**Verification.** `paper/main.tex` recompiles clean (12 pages, same single
pre-existing 1.98pt overfull hbox, no new warnings) after every edit in this
round. All 4 figures regenerated from the corrected result files. Changes
mirrored into `PAPER_DRAFT.md` and `README.md`, including a sweep for
leftover "exact tie" / "real effect" phrasing in both that the main.tex edit
had already superseded.

**Not done in this pass**: the 🟡 lower-priority ablation suggestion from
this same review round (a `node_n_neighbors`/load-bus-degree confound
check) — flagged as optional by the reviewer, not started. All four
originally-flagged items (plus the Table 1 bonus finding) are otherwise
closed as of this update.

### Update, 2026-09-23 (same day, second pass): a sign-test arithmetic
### error, an overstated latency abstract, a reproducibility ordering
### bug, and two stale-text cleanups

A fifth review pass, on the just-completed fourth-round fixes above, found
two real precision problems plus three smaller cleanups.

1. **The IEEE-14 localization sign test was computed wrong.** The caveat
   paragraph above reported one-sided $p=0.5^4=0.0625$ for *both* surviving
   gaps. That is correct for IEEE-30 detection (four of four seeds strictly
   positive: $+0.083,+0.040,+0.020,+0.013$). It is wrong for IEEE-14
   localization ($0.000,+0.047,+0.020,+0.007$): one of those four is an
   *exact tie*, and a standard sign test excludes ties from the count rather
   than treating them as a positive — verified independently via
   `scipy`/hand calculation before fixing. The effective sample is three of
   three non-zero seeds positive, giving $p=0.5^3=0.125$, not $0.0625$ —
   weaker evidence than IEEE-30's, not the same number. Fixed in both
   places this appeared (§7's caveat paragraph, the Limitations item).

2. **The "10–20 seeds" claim conflated two different questions.** The text
   said resolving the caveat "would need substantially more seeds (roughly
   10–20 ... for a sign test alone to reach conventional significance)."
   This is not what the arithmetic says: a sign test's own significance
   threshold is a low bar if the true pattern holds — one more same-signed
   seed at IEEE-30 (five of five) already gives one-sided $p=0.5^5=0.03125
   <0.05$, and a sixth would clear the two-sided threshold ($2\times
   0.5^6=0.03125$). Verified this arithmetic directly (`scipy.stats`) before
   fixing. The 10–20 figure is a defensible estimate for a *different* and
   harder goal — a tight, stable $t$-based confidence interval on the
   effect's *magnitude* — not for sign-test significance, which a handful
   of further same-signed seeds could reach on its own. Reworded both
   instances to separate the two claims rather than conflate them.

3. **The abstract's latency figures read more settled than the three-run
   measurement actually supports.** "11.6/12.4/12.6 ms at the
   median/p95/p99" is technically the most recent run's exact numbers, but
   a reader who hasn't reached the body text (which already describes the
   cross-run tail variability, added in the fourth-round update above)
   could read it as "p99 reliably under budget," when one of the three runs
   had p99 at 36.2 ms. Reworded the abstract in `paper/main.tex` and
   `PAPER_DRAFT.md` to state explicitly, at the abstract level, that median
   and p95 are the stable, reliably-in-budget quantities and that both the
   maximum and (in one of three runs) p99 are not. `README.md`'s
   corresponding passage already had this nuance from the fourth-round
   update and needed no further change.

4. **A reproducibility-sequencing bug in `README.md`'s canonical reproduce
   list.** `33_feature_redundancy_diagnostic.py`'s own `--help` text already
   warns that its IEEE-14 $R^2$ figure needs a *fresh primary-seed* run of
   `28_ieee14_scale_replication.py` — `30_ieee14_multi_seed_replication.py`
   overwrites `data/phase2v_ieee14_node_feature_matrix.csv` with its own
   last seed (2024), not the primary seed (20260921) the paper's IEEE-14
   $R^2=0.966$ figure is computed from. Verified this precisely: `28`'s
   own code is the only writer of that file; `30`'s `SEEDS` list starts at
   20260921 and ends at 2024, confirming it does overwrite the file with
   the wrong seed's data if run first. The README's reproduce sequence
   never ran `33` at all (28 → 30 → 31 → 32 → 29), so following it verbatim
   would not reproduce the paper's IEEE-14 redundancy number even if `33`
   were appended at the end. Fixed by inserting `33` between `28` and `30`,
   with a comment explaining why the ordering matters.

5. **Two stale-text cleanups.** A code comment in `29_paper_figures.py`
   still described the IEEE-30 detection finding as "a real, 4-seed-
   consistent topology advantage" — reworded to match the paper's own
   established, softened convention ("a gap positive in all 4 tested
   seeds"). Separately, this file's own pre-§6 "Summary" and "Limitations"
   sections still stated superseded conclusions (most visibly, "detection
   shows no topology-specific advantage at any of the three scales," no
   longer true once IEEE-30's consistent-sign gap was found) without a
   clear marker — a reviewer grepping this repo for the project's
   conclusion could land on either section and read a contradiction with
   no indication which is current. Added an explicit **SUPERSEDED** banner
   to the Limitations section, strengthened the existing pre-§6 warning at
   the end of the Summary section (which itself had gone stale — it still
   claimed the paper files hadn't been updated to match §6, which is no
   longer true), and updated this file's own "Last updated" date. Left the
   historical prose within §6 itself untouched (its own "Update" subsection
   structure already makes the chronology explicit, and retroactively
   sanitizing an audit's own log of how understanding evolved would defeat
   its purpose as a research log).

**Verification.** `paper/main.tex` recompiles clean (12 pages, same single
pre-existing 1.98pt overfull hbox) after all edits in this pass; visually
re-read every page against the changes. All arithmetic claims above
(sign-test p-values, seed-count thresholds) were independently computed
via `scipy.stats`/hand calculation before editing, not taken on the
reviewer's word alone — matching this project's established discipline.

### Update, 2026-09-23 (same day, third pass): the fourth-round README fix
### was correct but incomplete — the 5-bus side of the same bug was still
### live, and it turned out the published 5-bus R² was itself not
### reproducible

A sixth review pass, on the just-completed README reproducibility fix above,
found the same bug's 5-bus half still unfixed, plus one documentation-hygiene
item.

1. **`33_feature_redundancy_diagnostic.py`'s 5-bus side can read a stale
   seed too, and the previous round's fix didn't cover it.** The prior
   update above fixed `README.md`'s reproduce sequence so `33` runs between
   `28` and `30` (protecting its IEEE-14 side from `30`'s last-seed
   overwrite). It missed that `33`'s 5-bus side has the identical problem
   one step earlier: `run_5bus()` reads `data/phase2s_hard_graph_feature_matrix.csv`
   directly, prints `=== 5-BUS (HARD, primary seed) ===`, but has no check
   that the file actually holds the primary seed — and
   `26_multi_seed_replication.py`'s own seed loop (`SEEDS = [20260812, 42,
   777, 2024]`) overwrites that exact file on every iteration, leaving
   seed 2024's run on disk when it finishes. The README's sequence runs 22
   → 23 → **26** → 27 → 24 → 28 → 33 → 30 → ... — so by the time 33 runs,
   the 5-bus snapshot is whatever seed 26 last used (2024), not the primary
   seed (20260812) the paper's redundancy figure is documented as using.
   `27_held_out_attack_type_generalization.py` reads the same file
   directly and has the identical exposure, with no warning at all (not
   even the informal one `33` prints).

   **Fixed at the source rather than in each reader**, per the reviewer's
   own suggested approach: `26_multi_seed_replication.py` now re-runs
   `22_..._HARD.py --seed 20260812` + `23_..._HARD.py` one more time after
   its own multi-seed summary CSVs are safely saved, restoring the
   canonical primary-seed snapshot on disk before it exits. This fixes `27`
   and `33` (and any future script that reads these files directly)
   simultaneously, rather than adding a one-off check to each. Updated
   `README.md`'s reproduce-sequence comments on both the `26` and `33`
   lines to explain why.

2. **Verifying this surfaced a second, more serious problem: the published
   5-bus R² (0.950) does not actually reproduce, even from the correct
   primary-seed data.** Before concluding the fix above was sufficient,
   ran `33 --skip-ieee14` three times to check: once against whatever was
   on disk before the fix (confirmed stale — seed 2024 leftover, giving
   mean $R^2=0.911$), then twice against freshly-regenerated primary-seed
   (20260812) data (bit-identical both times, $R^2=0.958$ — confirmed
   `33`'s own `KFold(..., random_state=seed)` call always uses the fixed
   default `seed=0`, so it is not a source of run-to-run variance; the
   Ridge/KFold diagnostic itself is fully deterministic given fixed
   input). Neither run reproduces the paper's published 0.950. IEEE-14's
   $R^2$, by contrast, reproduced exactly (0.966/0.999/0.677/91.7%,
   matching the paper precisely) from whatever was already on disk for
   it — so this is specific to the 5-bus side, not a bug in the diagnostic
   script or a general non-reproducibility problem. Most likely
   explanation, consistent with two earlier findings this same day (Table
   1's stale exact-tie claim, the small 5-bus detection-mean drift): the
   published 0.950 was extracted from a 5-bus pipeline state at an earlier
   point in this project's many fix rounds, before the current, fully
   corrected code. **Treating the fresh, reproducible, twice-confirmed
   measurement as authoritative** (the same resolution applied to both of
   those earlier findings): updated the 5-bus redundancy figure throughout
   `paper/main.tex`, `PAPER_DRAFT.md`, and `README.md` to $R^2=0.958$
   (median 1.000 unchanged, minimum $0.390\to0.623$, columns above 0.8
   $88\%\to90.8\%$). The qualitative conclusion this figure supports —
   IEEE-14's redundancy ($R^2=0.966$) is at least as high as 5-bus's, not
   lower, so redundancy does not explain the IEEE-14 localization reversal
   — is unchanged: $0.966 > 0.958$ still holds.

3. **Documentation hygiene: §6's own opening paragraph had gone stale.**
   This section's header and its first paragraph still said "the paper's
   numbers are now stale" and "have not been updated yet," which was true
   only in the narrow window at the very start of this audit and had
   itself become misleading now that the paper files are kept current
   throughout this section's later updates. Marked the original paragraph
   explicitly as **"Historical state at the start of this audit —
   superseded by the updates later in this section,"** per the reviewer's
   suggested framing, rather than rewriting or deleting it — it remains an
   accurate description of that specific moment, just no longer of the
   present. Updated the section header similarly. Left every other dated
   "Update" subsection within this section untouched, for the same reason
   given in item 3 of the pass above: this section's own chronological
   structure already makes clear which parts are current.

**Verification.** Confirmed via direct execution, not assumption: `33`'s
determinism (identical output across two runs on identical data), the
stale-vs-fresh 5-bus $R^2$ discrepancy (0.911 vs. 0.958, both computed
directly), and IEEE-14's exact reproduction (0.966, matching the paper to
three decimal places). `paper/main.tex` recompiles clean (12 pages, same
single pre-existing overfull hbox) after the $R^2$ update.

### Update, 2026-09-23 (same day, fourth pass): the last real ablation --
### raw node degree was still inside `topology_fusion` at IEEE-14/30 -- plus
### a group-safety fix to the redundancy diagnostic and one editorial slip

A seventh review pass identified the most consequential remaining
methodological question in the whole project, plus two smaller,
independently-verified items.

1. **IEEE-14/30's `topology_fusion` still contained raw node degree.**
   `28_ieee14_scale_replication.py`/`31_ieee30_scale_replication.py`'s
   `argmax_node_topology()` returns `"argmax_n_neighbors": len(nbs)`
   (detection), and `per_node_rows()` returns `"node_n_neighbors": len(nbs)`
   (localization) -- both flow into `topology_fusion` (the "neighbor"
   substring filter that keeps them out of `residual_only`/`prior_only`
   does not apply to `topology_fusion`, which was simply `res_all +
   innov_all`, unfiltered). Since `target_bus = int(rng.choice(load_buses))`
   in `one_replication()` -- attacks only ever target load buses, never
   generator/slack buses -- raw degree could let a model learn "which bus
   types are typically targets" as a static prior, independent of any
   scenario-specific residual/innovation signal. This is exactly the
   reasoning that already removed `degree`/`role_*` from every 5-bus
   feature set, `topology_fusion` included (this section, above); IEEE-14/
   30 had never received the analogous fix, breaking three-network
   consistency in exactly the two findings the paper's whole scale study
   rests on (IEEE-30 detection, IEEE-14 localization).

   **Fixed by excluding `n_neighbors`-suffixed columns from
   `topology_fusion`** at both networks -- `topology_fusion = [c for c in
   res_all + innov_all if "n_neighbors" not in c]` (detection) and
   dropping `"node_n_neighbors"` from the localization feature list
   (localization). `residual_only`/`prior_only`/`residual_plus_prior` are
   structurally unchanged, since degree was never in them. Smoke-tested
   both scripts at `--n-rep 15` before committing to the full rerun.

   **Re-ran the full 4-seed replication at both networks
   ($N{=}500$, sequentially, to avoid CPU contention) and checked whether
   the paper's two headline findings survive.** They do, essentially
   unchanged:
   - **IEEE-30 detection**: still positive in all 4 seeds. Gap $+0.013$ to
     $+0.080$ (was $+0.013$ to $+0.083$), mean $+0.039\pm0.029$ (was
     $+0.039\pm0.032$) -- the mean is identical to three decimal places;
     only the std tightened slightly.
   - **IEEE-14 localization**: still non-negative in all 4 seeds (one
     exact tie, three strictly positive). Gap $+0.000$ to $+0.047$ (was
     the same range), mean $+0.023\pm0.021$ (was $+0.018\pm0.021$) -- if
     anything, a *larger* mean gap after removing degree, not smaller,
     which rules out degree having been doing positive work for this
     finding.
   - Both networks' detection-only and localization-only *columns*
     (`residual_only`/`prior_only`/`residual_plus_prior`) are numerically
     identical to the pre-fix values, confirming the fix touched only
     `topology_fusion` as intended, with no side effects elsewhere.
   - IEEE-14 detection and IEEE-30 localization remain null/sign-unstable,
     as before (unaffected in character; specific per-seed values shifted
     slightly since `topology_fusion` is one of the models compared for
     "best non-relational alternative" selection at each seed, so removing
     one of its columns has some downstream numerical effect even on
     columns nominally unrelated to it -- but the qualitative story for
     both is unchanged).

   Updated every dependent number throughout `paper/main.tex`,
   `PAPER_DRAFT.md`, and `README.md`: Table III's IEEE-14/IEEE-30
   `topo.` cells, Tables IV/V in full, the Abstract, the §7 "Result"/
   "Interpretation" paragraphs, the statistical-caveat paragraph's CIs and
   the "roughly 1.7$\times$" magnitude comparison (recomputed, still
   accurate), and the Limitations item. Regenerated Fig. 4. Added a note
   to §7's methodology paragraph documenting this as a third,
   independent audit-and-fix round on top of the feature-leakage and
   oracle-residual/model-selection fixes already described there.

2. **`33_feature_redundancy_diagnostic.py`'s `KFold` wasn't group-safe.**
   Every accuracy result elsewhere in this project splits train/test at
   the *replication* level specifically so a single replication's matched
   scenarios never straddle the split -- this diagnostic's own
   out-of-fold $R^2$ check used a plain `KFold(shuffle=True)` instead,
   risking a relational column "predicting itself" via a near-duplicate
   row from the same replication landing in the opposite fold. Fixed by
   switching to `GroupKFold(n_splits=5)`, grouped by the 5-bus
   dataframe's own `replication` column and, for IEEE-14, by the numeric
   prefix of `scenario_id` (`f"{rep}_{case_name}"` -- splitting on the
   first underscore isolates `rep` regardless of how many underscores
   `case_name` itself has, since `rep` is purely numeric). Re-ran both:
   5-bus $R^2$ moved from 0.958 to 0.956 (median unchanged at 1.000,
   minimum $0.623\to0.613$); IEEE-14 $R^2$ was unchanged to three decimals
   (0.966). Both stayed within the ~0.95--0.97 range the reviewer
   anticipated, so this is a clean, low-consequence fix -- updated the two
   sub-0.01 figure changes throughout the paper files, no narrative
   change needed.

3. **One editorial slip, purely a stale cross-reference.** The
   "Positioning" paragraph comparing against a second literature data
   point (a knowledge-distilled LightGBM detector's inference cost) still
   quoted this paper's *pre-timer-fix* state-estimation/inference
   sub-step numbers (16--17\,ms / $\sim$0.19\,ms/sample) even though the
   main latency paragraph a few lines earlier already had the corrected,
   clean-remeasurement figures ($\sim$11--12\,ms / $\sim$0.13\,ms/sample)
   from the fourth-round update above. Updated to match.

**Verification.** Both `28_...py` and `31_...py` smoke-tested at
`--n-rep 15` before the full rerun. Full reruns completed without error;
per-seed and cross-seed numbers extracted directly from the saved CSVs,
not estimated. `33`'s group-safe determinism confirmed the same way as
the previous update (re-run twice, identical output). `paper/main.tex`
recompiles clean (12 pages, same single pre-existing overfull hbox) --
the added text initially pushed this past 12 pages a second time; trimmed
several sentences (Abstract, §7 Result/Interpretation, Conclusion) to
reclaim it without cutting content, the same discipline as the earlier
12-page recovery in this section's second update. Visually re-read every
changed page against the source diff before committing.

### Update, 2026-09-23/24: editorial-voice cleanup, then several rounds
### of author-driven polish directly to the source, then a fresh
### documented latency rerun

Two things happened in this stretch, summarized rather than narrated in
full -- the point of this pass was specifically to get that kind of
blow-by-blow *out* of the paper's own prose, and duplicating it here in
just-as-much detail would defeat the purpose.

1. **A full editorial pass separated the audit chronology from the final
   science.** The paper's prose used to narrate its own debugging history
   in the Abstract, Introduction, Methodology, Results, and Discussion
   ("first pass appeared to show X, then a bug was found, then a second
   audit found Y..."). Rewrote throughout to state only the final,
   corrected methodology and results, with one brief technical mention
   per correction category (kept deliberately, not hidden) and the full
   before/after chronology left here in §6 where it already lived.
   `paper/main.tex` dropped from 12 to 10 pages with zero loss of any
   verified number or statistical caveat. Followed by two further rounds
   of smaller phrasing fixes (a Limitations cross-reference bug, several
   "real"/"more likely"/informal-register cleanups) -- see `git log
   --oneline paper/main.tex` for the individual commits if the exact
   wording history matters; it isn't re-narrated here.
2. **Several rounds of author-driven polish landed directly on `main`**,
   reviewed and verified after the fact rather than authored in this
   session: a genuine factual fix (the paper had been saying "three
   independently-implemented pipelines," but IEEE-14 and IEEE-30 actually
   share one size-agnostic implementation -- three networks, two
   pipelines, not three of each); an author-affiliation fix (the IEEE
   author block had a degree title where a department name belongs);
   honest venue-status framing (this file is a full manuscript/preprint,
   not implied SmartGridComm-compliant); a bibliography upgrade (6 entries
   moved from arXiv preprints to their verified published versions with
   DOIs -- spot-checked 2 of the new DOIs directly against the CrossRef
   API, both resolved to exactly the cited paper); and reproducibility
   metadata (`24_realtime_latency_benchmark_FINAL.py` now records and
   saves the exact machine/library-version environment alongside every
   latency run, instead of the benchmark script staying silent about it).
3. **A fresh, fully-environment-documented latency rerun replaced the
   three-run stability-check framing from the update above.** New result,
   confirmed directly against `results/phase2t_latency_final.csv` and
   `results/phase2t_latency_environment.json` on disk (not just the
   commit message): **17.015 / 18.211 / 19.190\,ms** median/p95/p99,
   **92.657\,ms** max, 0 convergence failures, on Apple arm64 / macOS
   15.6.1 / Python 3.12.4 / pandapower 2.14.10. Stage medians: WLS
   16.664\,ms, post-estimation residual/innovation 0.057\,ms, feature
   computation 0.088\,ms, array assembly 0.008\,ms, inference 0.200\,ms
   -- sums to the reported total exactly. `29_paper_figures.py`'s Fig. 2
   now reads its final bar from this CSV dynamically instead of a
   hardcoded value, so it cannot silently drift from the manuscript again
   after a future rerun.

   **Worth flagging explicitly, not silently overwriting**: this is the
   *fourth* latency measurement taken on this same machine this project
   (16.9\,ms original-era baseline; three back-to-back runs in this
   section's second update at 11.6--11.7\,ms; now 17.0\,ms). The median
   alone has ranged 11.6--17.0\,ms across these four runs -- a larger
   swing than the $\pm5\%$ stability the three-run check found *among
   itself*, meaning whatever varies between sessions on this machine
   (thermal state, background load, something else not isolated) has
   more effect than run-to-run noise within one sitting. This run's own
   p99 (19.190\,ms) is now inside the 20\,ms budget by only 0.81\,ms --
   a closer call than the three-run update's p99s (12.4--12.7\,ms in two
   of three runs). The qualitative conclusion (WLS state estimation
   dominates; median and p95 are typically in-budget; the tail is not
   reliably) is unchanged, and reporting one clean, fully-documented run
   is a legitimate simplification of the three-run framing -- but a
   reader should not take 17.0\,ms, or the earlier 11.6\,ms, as *the*
   number this pipeline always produces on this hardware. Left as the
   author's call whether to add this variability back into the paper's
   own Limitations item; not done unilaterally here.

**Verification.** All new numbers in `paper/main.tex`/`PAPER_DRAFT.md`/
`README.md` checked directly against the CSV/JSON files on disk, not
taken from the commit diffs alone. Found and fixed one mirroring gap the
author's own commits missed: `PAPER_DRAFT.md`'s Abstract and Limitations
item 6 still had the old 11.6--12.6\,ms figures and the generic "one
machine" phrasing after the "Mirror documented final latency benchmark"
commit -- the Results II body paragraph had been updated correctly, the
Abstract had not. Two bibliography DOIs spot-checked against CrossRef
directly. Full `pdflatex` $\to$ `bibtex` $\to$ `pdflatex` $\times 2$
rebuild: no bibtex warnings, no undefined citations, still 10 pages,
same single pre-existing overfull hbox. Fig. 2 regenerated and visually
confirmed to read the new value.

### Update, 2026-09-25: 20 author commits pulled (CI build, arXiv
packaging, a second latency rerun), external pre-arXiv feedback
triaged, one real comparator-labeling fix made

**Twenty commits pulled from `origin/main`, all author-authored, none
seen before this pull.** Reviewed the full diff rather than trusting
the commit messages. Contents: a GitHub Actions workflow
(`.github/workflows/build-paper.yml`) that rebuilds the paper on push;
a new `make_arxiv_submission.sh` plus a checked-in `arxiv/main.tex`
mirror (differs from `paper/main.tex` only in figure paths --
confirmed with `diff`) and a committed `arxiv_preview.pdf`; further
framing/figure-caption polish and Table I typesetting fixes; and,
separately, **a second independent $N{=}300$ latency rerun**, now
reported in the Abstract and Section~V as a range across both runs
(16.155--17.015\,ms median, 16.429--18.211\,ms p95, 16.952--19.190\,ms
p99, 89.067--92.657\,ms max) rather than a single run's point values.
This is a direct, author-made resolution of the exact question flagged
open in the entry immediately above ("left as the author's call
whether to add this variability back into the paper"); verified
directly against `results/phase2t_latency_final.csv` and
`results/phase2t_latency_environment.json` on disk -- the file's
current contents match the range's lower bound to 6 significant
figures, confirming a real second run happened rather than the range
being invented after the fact. The upper bound matches the single run
verified in the entry above exactly.

**External feedback triaged.** The author received (and pasted in,
translated/summarized) a detailed pre-arXiv review covering: framing/
"story" clarity, raising seed count to 10--20, cutting prose length,
the development-narrative voice (already fixed, see the editorial-
voice-cleanup entry above), the Abstract's density, figure polish
(also already done in the commits just pulled), overuse of em dashes
as an "AI tone" tell (judged mostly not applicable -- em dashes are
normal academic style; the real issue was self-referential "we tested
it rather than leaving it as speculation"-type phrasing, already
removed), a missing GitHub/code link in the paper, a final reference
sanity pass, the 90\,ms latency tail (judged not worth chasing further
for v1 -- correctly noting $N{=}300$ makes p99.9 meaningless and that
any GC/scheduling explanation would be unmeasured speculation), and
advisor/endorsement questions. The author's own reply reprioritized
this into: comparator consistency first, then GitHub public-release
audit + code link, then Abstract tightening, then (time permitting)
10--20 seeds -- restated once more concretely as comparator check
$\to$ Abstract $\to$ repo audit \& code link $\to$ final reference
check $\to$ arXiv v1.

**Comparator consistency, checked.** The concern: Section~IV (5-bus)
explicitly compares \texttt{topology\_fusion} against a fixed
\texttt{residual\_plus\_prior} baseline, but Section~VII's (IEEE-14/30)
table captions and prose described the comparator as "the best
non-relational alternative each seed" -- wording that reads as a
per-seed-selected baseline, a different (and not obviously matched)
estimand from 5-bus's fixed one. Checked the actual computation in
`30_ieee14_multi_seed_replication.py` (line 64) and
`32_ieee30_multi_seed_replication.py`:
\texttt{best\_nonrelational = max(residual\_plus\_prior, residual\_only,
prior\_only)}, computed independently per seed. Checked the raw output
directly (`results/phase2w_ieee14_multiseed_replication.csv`,
`results/phase2y_ieee30_multiseed_replication.csv`): in all 16 cells
(2 networks $\times$ 2 tasks $\times$ 4 seeds), the
\texttt{best\_nonrelational} column is bit-identical to
\texttt{residual\_plus\_prior} -- confirmed also by hand-subtracting
Table III's 4-seed-mean columns, which reproduce the headline gap
figures exactly. So \texttt{residual\_plus\_prior} was, in fact, the
strongest non-relational baseline in every single case; the reported
numbers were always correct, this was a labeling-precision issue, not
a data bug -- no rerun needed, no headline number changed.

**Fix applied.** "the best non-relational alternative each seed" to
"\texttt{residual\_plus\_prior} (the strongest non-relational baseline
in every seed)" in both Table IV/V captions, the Cross-Network Result
paragraph's closing clause, and the Abstract -- in `paper/main.tex`,
`arxiv/main.tex` (kept in sync by hand, since the build script doesn't
write to it), and `PAPER\_DRAFT.md`. `README.md` had no occurrence
(checked). This also makes the comparator's name identical in wording
across all three networks for the first time. While in `PAPER_DRAFT.md`,
also found and fixed a second mirroring gap the author's own latency-
rerun commits had left: the Abstract still had the pre-rerun single-run
numbers (17.0/18.2/19.2/92.7\,ms) even though Results II's body
paragraph already had the correct two-run range.

**Verification.** Full `pdflatex` $\to$ `bibtex` $\to$ `pdflatex`
$\times 2$ rebuild of `paper/main.tex`: clean, no undefined citations,
still 10 pages. Rendered Table IV/V pages visually inspected -- longer
captions wrap cleanly within the IEEE column width, no overflow.
`make_arxiv_submission.sh` re-run end to end to regenerate
`arxiv_preview.pdf`/`arxiv_submission.zip` from the corrected source;
completed clean (its own undefined-reference/fatal-error guards both
passed).

**Not yet done, by design.** Per the author's own priority order, only
step 1 (comparator check) has been done this round. Abstract tightening,
the GitHub public-release audit, adding a code-availability line to the
paper, the final reference sanity pass, and any seed-count increase are
still open -- the repo-visibility change and the arXiv submission itself
are also actions that need the author's own direct action, not
something to do unilaterally.

### Update, 2026-09-25 (same day, continued): repo audit, Abstract
tightened, code-availability line added, and a real citation problem
found and fixed in the Introduction's motivating example

**Repo public-release audit.** Repo is currently private
(`gh repo view`: `"isPrivate":true`). Checked: no credentials/API
keys/private-key material in HEAD or anywhere in full git history
(`git log -p --all` grepped for common secret patterns -- clean); no
credential-shaped filenames ever committed and removed; no phone
numbers or non-author personal emails (a regex sweep's only hits were
DOI strings, false positives); no unprofessional language in the
tracked `.md` files; all 69 tracked files eyeballed by name -- numbered
phase scripts, `exploration/`, docs, figures, paper source, CI
workflow, `requirements.txt` -- nothing out of place; `results/`/
`data/`/per-phase diagnostic PNGs correctly gitignored, repo is small
(35 MB `.git`, largest tracked file 532 KB). One gap: **no `LICENSE`
file** -- flagged as a decision for the author (license choice affects
reuse rights; not picked unilaterally). Conclusion: repo is safe to
make public whenever the author confirms; that flip itself is a GitHub
account-settings change and still needs the author's own explicit
go-ahead, not implied by "continue with the steps."

**Abstract tightened**: 259 words to 205. Cut the four null/
sign-unstable per-network gap figures (5-bus det/loc, IEEE-14 det,
IEEE-30 loc) from the Abstract -- they remain fully reported in
Section~IV/Table III/the Cross-Network Result paragraph, just not
repeated in the Abstract -- keeping only the two surviving consistent-
sign gaps (IEEE-30 detection, IEEE-14 localization) plus one compressed
non-confirmation caveat instead of a separate sentence. Latency and
zero-day sentences kept at full numeric precision (not what the
external feedback's length complaint was about). Mirrored identically
across `main.tex`, `arxiv/main.tex`, `PAPER_DRAFT.md`.

**Code and Data Availability section added**: a new unnumbered
`\section*` right after the Conclusion, before the bibliography --
"Code and reproducibility materials are available at
https://github.com/yagizsarapli/cyber-physical-grid-detection." --
in `main.tex`, `arxiv/main.tex`, and `PAPER_DRAFT.md`. The `url`
package was already loaded, so no preamble change needed. This link
will 404 for anyone else until the repo audit above is acted on and
the repo is actually made public -- the text is ready, the visibility
flip is not done.

**Reference sanity check, and a real problem found.** Fetched primary
sources directly (`WebFetch`/`WebSearch`, not trusting the bibliography
entry's own text) for the highest-risk citations: the one non-academic,
real-world-incident source, plus the two citations whose specific
numbers are quoted directly in this paper's own prose. Three checked
out exactly (title/authors match, and for
\texttt{abukhousa2026latency} and \texttt{ogiesoba2026cyberattack} the
quoted "sub-15\,ms / 50--90\,ms" and "54--67\,ms per 1000 samples"
figures were independently confirmed verbatim from each paper's own
abstract). Two more (\texttt{suri2025powergnn}, \texttt{yaniv2026robust})
checked out on title/author. The fourth, \texttt{certpolska2026followup}
(the December-2025 Poland energy-sector cyberattack, cited once, in the
Introduction's opening motivating example), did not:

- The citation pointed at CERT Polska's \emph{follow-up} report
  (August 2026), but fetching it directly showed it covers a
  \emph{different}, later-disclosed issue -- a private-APN
  misconfiguration affecting a second, smaller CHP plant, 50{,}000
  residents, a brief heat-supply interruption. It does not contain the
  firmware/attribution/scale details the paper's sentence actually
  describes.
- Those details belong to CERT Polska's \emph{original} January 2026
  report (a different URL, not previously cited at all). Fetched that
  directly and confirmed: $>$30 wind/solar farms plus one CHP plant
  attacked, confirmed (matches); firmware damage and wiper malware
  confirmed, but "forced into endless restart cycles" specifically is
  \emph{not} in the primary source -- an unconfirmed embellishment,
  removed; "Dragonfly" confirmed as one of four vendor names for the
  same actor (Cisco: Static Tundra, CrowdStrike: Berserk Bear,
  Microsoft: Ghost Blizzard, Symantec: Dragonfly) -- the paper's
  attribution language was already accurate and is unchanged.
- The paper's "500,000 people" blackout figure is the CHP plant's
  \emph{normal customer base} (what the primary source calls the
  at-risk population), not an actual outage count -- the same source
  states explicitly that the attacks "did not affect the ongoing
  production of electricity" and "did not achieve the attacker's
  intended effect of disrupting heat supply." The original wording
  ("was contained before it caused a blackout affecting an estimated
  500,000 people") was defensible on a careful reading but genuinely
  easy to misread as a partial blackout that did happen; independently
  cross-checked against a `WebSearch` sweep (Hacker News, SecurityWeek/
  AP, SC Media, Wikipedia's incident article) -- all consistent with
  the primary source, none support a "500,000 affected" reading.

Fixed: new bib key \texttt{certpolska2026incident} pointing at the
January report's correct URL, replacing \texttt{certpolska2026followup}
(which was cited nowhere else in the paper, and was removed rather than
left orphaned); the Introduction sentence rewritten to state only the
three confirmed facts (facility count and type, firmware/wiper damage,
customers \emph{served} rather than blacked out, attribution) without
the unconfirmed restart-cycle mechanism or the ambiguous near-blackout
framing. Applied to `paper/references.bib`, `arxiv/references.bib`,
`main.tex`, `arxiv/main.tex`, `PAPER_DRAFT.md`. Not otherwise mentioned
in `README.md`, `STATUS.md`, or `RELATED_WORK.md` (checked).

**Verification.** Full `pdflatex` $\to$ `bibtex` $\to$ `pdflatex`
$\times 2$ rebuild after the bibkey rename: no bibtex warnings, no
undefined citations, citation position unchanged (still resolves as
[3], since nothing else in the citation order changed), still 10
pages. Rendered pages 1 and 9 visually checked -- Abstract, Introduction
paragraph, Code and Data Availability section, and the corrected [3]
bibliography entry all read as intended.

**Not yet done.** The remaining un-spot-checked 2025/2026 entries
(\texttt{falas2026learning}, \texttt{li2026physically}, \texttt{lin2026state},
\texttt{sakr2026explainable}) were not individually re-verified this
round -- none carry a directly-quoted number or a real-world factual
claim the way the four checked ones did, so this was judged acceptable
for a "final sanity pass" rather than a full re-audit. The repo-
visibility flip and any seed-count expansion remain the author's calls.

### Update, 2026-09-25 (same day, continued): repo reorganized before
going public -- superseded scripts archived, live pipeline untouched

The author asked, ahead of making the repo public, for it to not look
like unreviewed AI-iteration debris and to be organized around only
what the current study actually uses. Rather than guess, built the
real dependency graph first: grepped every `.py` file (root and
`exploration/`) for `load_module(...)` call sites and their actual
filename arguments (these use `importlib.util.spec_from_file_location`
for dynamic same-directory imports, so a plain `import`/`from` grep
alone would have missed real dependencies), plus a separate sweep for
which scripts read which `data/phase2*.csv` file. This caught a real,
pre-existing inconsistency in the README's own claim that
"`exploration/` contributes nothing except `01_microgrid_topology.py`"
-- untrue: `exploration/08_multirate_operating_dataset.py` generates
`data/phase2e_7day_operating_dataset.csv`, read by nine different
root-level scripts (21/21\_HARD/22/22\_HARD/22\_HARDER/24/24\_FINAL/
24\_OPTIMIZED/24\_WARMSTART/25), and is explicitly the first step in
the README's own "Reproducing this" sequence. Would have broken the
reproduction path if archived on the README summary's word alone.

**Confirmed genuinely unused** (zero references anywhere, by filename
string, in any `.py` file): all of `exploration/02` through `20`
(including every `_FIXED`/`_V3`/`_RESEARCH_FIXED` variant and `07a`)
except `01` and `08` (plain); and, at the root, the pre-stress-test
originals `21_nonlinear_stealth_fdia_bdd_benchmark.py` (plain),
`22_graph_ready_protected_prior_telemetry.py` (plain),
`23_topology_aware_cyber_physical_localization.py` (plain), and
`23_..._FIXED.py` -- all four superseded once the `_HARD` stress test
was introduced, and the paper only ever reports Hard/Harder-condition
numbers, never the pre-stress-test ones.

**Deliberately left alone**: every other root-level variant, in
particular `24_realtime_latency_benchmark.py`/`_WARMSTART`/
`_OPTIMIZED`/`_FINAL`. These aren't code dependencies of anything
either, but each is the individual, still-cited evidence for one step
of the paper's own Section~V optimization narrative (42.98\,ms
baseline $\to$ 37.14\,ms warm-start $\to$ the 1158$\times$ measurement-
update fix) -- archiving these would have quietly made a paper claim
non-reproducible, which is the opposite of what a public-release audit
should do. This is a real, principled distinction, not just caution:
"superseded experimental design, nothing current depends on it" vs.
"an individually-cited measurement step," decided per file, not by
suffix pattern.

**Moved** (`git mv`, history preserved, nothing deleted): the 26
confirmed-unused `exploration/` files to `exploration/archive/`; the 4
confirmed-unused root files to a new top-level `archive/`. Every
moved file defines `ROOT = Path(__file__).resolve().parent` (checked:
all 30, no exceptions) -- now one directory deeper, so all 30 got a
mechanical `.parent` $\to$ `.parent.parent` fix to keep their own
internal data/results paths correct if anyone runs one directly later;
`python3 -m py_compile` on all 30 confirmed no syntax breakage.
Re-ran the full dependency grep after the move (not just before) to
confirm zero dangling references from any remaining live file, and
`py_compile`'d every remaining root/`exploration/` script too.

**README updated** to match: the Repository map's `exploration/` entry
now correctly names both live files (not just one) and points to
`exploration/archive/`; a new `archive/` entry added; the "numbering
is a research log" paragraph reworded to state the actual, now-
per-file-justified policy (kept alongside at the root when
individually cited by the paper, moved to `archive/` when fully
superseded) instead of a blanket "everything is kept alongside on
purpose" claim that was no longer true for these four. Root now holds
17 `.py` files instead of 21; `exploration/`'s own top level holds 2
instead of 26. Checked `.github/workflows/build-paper.yml` and every
other tracked doc for a reference to any moved path -- none found, so
none needed updating beyond the README sections above.

Not done as part of this pass (out of scope, not asked for): rewriting
comments/docstrings in the scripts that stayed. Spot-checked first --
no emoji anywhere, no generic AI-tell phrasing, the codebase's own
comments are specific technical warnings (e.g. leakage-column and
group-safety notes), not filler -- so there was nothing there to fix.

### Update, 2026-09-25 (same day, continued): 4 seeds to 16 across all
three networks and both tasks -- the central finding changed from
task-specific to network-specific, and is now mostly statistically
confirmed

**What triggered this.** The author's own reply to the external
pre-arXiv feedback ranked seed count as the single most scientifically
valuable open item, ahead of everything else if time allowed, with an
explicit constraint: it had to be the same protocol at all three
networks, both tasks -- not just re-running the two seeds that already
looked good.

**What was done.** Appended 12 new seeds (3, 11, 19, 37, 53, 71, 97,
131, 163, 197, 229, 251 -- arbitrary, distinct from the existing 5)
to the \texttt{SEEDS} list in \texttt{26\_multi\_seed\_replication.py},
\texttt{30\_ieee14\_multi\_seed\_replication.py}, and
\texttt{32\_ieee30\_multi\_seed\_replication.py}, identically, bringing
each to 16 total. Confirmed first that this was safe to do purely by
appending: all three scripts index the primary seed as
\texttt{SEEDS[0]} rather than a hardcoded count, so nothing else in the
pipeline depends on there being exactly four. Ran all three scripts as
parallel background processes (5-bus, IEEE-14, IEEE-30 write to
different result files, confirmed no shared-file race) rather than
sequentially, since a full sequential run would have been several
hours; wall-clock was roughly 1-2 hours per network, overlapped.

**Verification before trusting any of it.** Read each script's raw
per-seed output directly, not just its own printed summary. Confirmed
determinism: the first 4 rows of each new 16-row result file are
byte-identical (to displayed precision) to the values already published
in Tables IV/V of the previous version of \texttt{paper/main.tex} --
e.g.\ IEEE-14 seed 42: \(-0.0134/+0.0467\) matches the paper's
\(-0.013/+0.047\) exactly. This rules out a reseeding bug or any
divergence in the underlying pipeline between the original run and
this one; the new 12 seeds are a clean superset, not a different
experiment. Recomputed every statistic independently with \texttt{scipy}
(\texttt{ttest\_1samp}, \texttt{binomtest} for the sign test, a
$t$-based 95\% CI) directly from the raw per-seed CSVs, not by hand and
not by trusting the scripts' own printed summaries.

**The result, in full** (all values 16-seed; the pre-existing 4-seed
values are given for contrast):

| Network | Task | 4-seed mean$\pm$std | 4-seed CI excl.\ 0? | 16-seed mean$\pm$std | 16-seed CI excl.\ 0? | sign $p$ | $t$-test $p$ |
|---|---|---|---|---|---|---|---|
| 5-bus | Det. | $+0.0075\pm0.012$ | no | $-0.0004\pm0.0086$ | no | 0.500 | 0.849 |
| 5-bus | Loc. | $-0.0067\pm0.016$ | no | $+0.0029\pm0.0169$ | no | 0.212 | 0.500 |
| IEEE-14 | Det. | $-0.0033\pm0.013$ | no | $+0.0090\pm0.0124$ | \textbf{yes} | 0.059 | 0.011 |
| IEEE-14 | Loc. | $+0.0233\pm0.021$ | no | $+0.0175\pm0.0170$ | \textbf{yes} | $<$0.001 | $<$0.001 |
| IEEE-30 | Det. | $+0.0392\pm0.029$ | no | $+0.0281\pm0.0192$ | \textbf{yes} | $<$0.001 | $<$0.001 |
| IEEE-30 | Loc. | $+0.0017\pm0.023$ | no | $+0.0129\pm0.0164$ | \textbf{yes} | 0.018 | 0.007 |

At 4 seeds, zero of six gaps had a CI excluding zero -- matching what
the paper already honestly said. At 16 seeds, four do. 5-bus stays
null on both tasks, tighter than before. IEEE-30 detection and IEEE-14
localization -- the two gaps that were already sign-consistent at 4
seeds -- both held up and tightened. IEEE-14 detection and IEEE-30
localization -- both null at 4 seeds -- flipped to small, positive,
CI-excluding-zero gaps at 16. A Bonferroni correction across all six
simultaneous tests (needing $p<0.0083$ for family-wise 95\% confidence)
still clears IEEE-14 localization, IEEE-30 detection, and IEEE-30
localization; IEEE-14 detection ($p=0.011$ uncorrected) does not clear
it, and is explicitly flagged throughout as the weakest-supported of
the four rather than folded in uncritically.

**Why the 4-seed IEEE-14 detection mean was negative when the true
effect (per the 16-seed estimate) is positive**: not a red flag, an
expected consequence of a small effect (0.009) sitting well inside a
per-seed standard deviation (0.012-0.019 throughout). A 4-observation
mean of a distribution with that little separation from zero is highly
volatile; the four-seed pilot's own explicit caveat (a $t$-based CI
crossing zero, a 10--20-seed recommendation) said exactly this, before
any of this round's data existed. This round is that caveat being
acted on, not contradicted.

**The paper's central finding changed as a result**, from
"task-and-network-specific, no uniform story" (the pattern the 4-seed
pilot showed) to "network-specific, not task-specific" (both standard
IEEE systems, both tasks, one custom microgrid, neither task) -- a
materially cleaner, more publishable, and now mostly statistically
confirmed story. Rewrote, on the author's explicit instruction ("sen
yaz"): Abstract; Results~I's multi-seed paragraph and Fig.~1 caption
(16-seed 5-bus numbers, still null); Table~III (16-seed 4-feature-set
means); Fig.~4 and its caption; replaced the two separate 4-seed
per-seed tables (Tables IV/V) with one combined 16-seed summary-
statistics table (new Table~IV: mean, 95\% CI, sign $p$, $t$-test $p$,
all six network/task cells, with a Bonferroni footnote) -- a per-seed
listing of all 16 rows $\times$ 2 networks would not have fit
comfortably in the IEEE column width, and the inferential statistics
are the more useful presentation at this $n$ regardless; Cross-Network
Result; the renamed "Statistical Confirmation at 16 Seeds" subsection
(was "Statistical Caveat," and needed a new \texttt{\textbackslash
label\{sec:scale-stats\}} since Section~\ref{sec:method}'s "16 seeds"
sentence now forward-references it); Interpretation; all three
Discussion paragraphs (the redundancy-diagnostic paragraph's own
conclusion -- reduced linear recoverability does NOT explain IEEE-14's
pattern -- was already correct and unchanged in substance, but its
surrounding framing needed updating since the mystery it was originally
explaining, "why only localization at IEEE-14," no longer exists in
the same form); Limitations items 1 and 2; Conclusion; and the
Contributions bullet in the Introduction that still described the old
task-specific pattern (found by a full-file grep sweep after the
section-by-section edits, not caught by section-by-section editing
alone -- worth remembering to always do a final whole-file sweep after
a rewrite this size). Mirrored identically to \texttt{arxiv/main.tex}
(regenerated programmatically from the updated \texttt{paper/main.tex}
via the same \texttt{"../figures/" -> ""} substitution
\texttt{make\_arxiv\_submission.sh} itself uses, rather than
hand-repeating $\sim$15 edits a second time) and to
\texttt{PAPER\_DRAFT.md} (by hand, converting LaTeX macros to its own
plain-prose/backtick style, including its own copy of the new combined
table and its end-of-file "still needs" checklist).

Also updated \texttt{29\_paper\_figures.py}: Fig.~1's suptitle
("four independent seeds" $\to$ "16") and Fig.~4's three x-axis labels
("4-seed mean" $\to$ "16-seed mean") were the only two hardcoded seed-
count strings; the actual aggregation code already used \texttt{.mean()}/
\texttt{.std()}/\texttt{len(seeds)} generically and needed no change to
correctly average all 16 rows once the underlying CSVs had 16 rows.
Regenerated all four figures; visually confirmed Fig.~1's 16 dots per
bar and Fig.~4's new "(16-seed mean)" labels render correctly.

Also updated \texttt{README.md}: the "Short answer" teaser line; the
5-bus 4-seed numbers in "The headline result" (now shows both the
original 4-seed figures and the current 16-seed ones, explicitly
framed as "tighter, not overturned," rather than silently replacing
history); inserted a new dated "Update, 2026-09-25" paragraph plus a
rewritten statistics/caveat paragraph after the existing IEEE-14/30
narrative, rather than deleting that narrative -- it remains accurate
history of what the 4-seed pass found; the three multi-seed scripts'
descriptions and timing comments in the Repository map and reproduce
sequence (also added an explicit note there that 30/32, unlike 26,
do not auto-restore the primary-seed snapshot afterward -- a
pre-existing, already-documented asymmetry, not something this pass
introduced, but worth surfacing directly at the point of use); and the
"Scale" bullet in the closing status section.

**Rebuild and final verification.** One new overfull hbox appeared
after adding the 6-row summary table (16.29pt too wide) -- traced to
the table itself (6 columns, no \texttt{\textbackslash resizebox}), not
its footnote text (tightening the footnote wording first did not
change the warning at all, which is what pointed at the table); wrapped
it in \texttt{\textbackslash resizebox\{\textbackslash columnwidth\}}
matching Table~I's existing pattern, confirmed the warning disappeared
on rebuild. Full \texttt{pdflatex} $\to$ \texttt{bibtex} $\to$
\texttt{pdflatex} $\times 2$ cycle after that: clean, no undefined
citations, no overfull/underfull warnings of note, still 10 pages
(the new combined table replacing two roughly balanced the added
prose). Visually inspected pages 1 (Abstract), 7-9 (Table III/IV,
Cross-Network Result, Statistical Confirmation, Discussion,
Limitations, Conclusion) -- all read correctly, table wraps cleanly.
\texttt{make\_arxiv\_submission.sh} re-run end to end; clean.

**What this does not change.** The Redundancy Diagnostic numbers
(5-bus $R^2=0.956$, IEEE-14 $R^2=0.966$) are computed from the
primary-seed feature matrix only and are independent of the multi-seed
loop; not re-run, not stale. Table~I (the single-primary-seed ablation)
is likewise untouched. The latency benchmark, zero-day generalization
test, and bibliography are unrelated to this pass.

### Update, 2026-09-26: a 28-commit author editorial pass reviewed and
pulled, then two small follow-up fixes

**The author's own 28-commit pass** (pulled as a single fast-forward,
\texttt{b7be014..57d7ca2}) was a substantial, high-quality independent
editorial round, reviewed in full via \texttt{git diff} rather than
trusting the commit messages. Contents: removed \texttt{PAPER\_DRAFT.md}
and the whole \texttt{arxiv/} folder (\texttt{main.tex} +
\texttt{references.bib}) as genuinely redundant manuscript copies --
\texttt{make\_arxiv\_submission.sh} already builds its own ephemeral
copy from \texttt{paper/main.tex} at package time, so the hand-
maintained \texttt{arxiv/} mirror this project had been keeping in
sync all session was never actually load-bearing; confirmed
\texttt{make\_arxiv\_submission.sh} itself was untouched and still
works correctly now that folder is gone. Rewrote \texttt{README.md}
into a substantially shorter, more conventional reproducibility
document. Throughout \texttt{paper/main.tex}: replaced this project's
own "network-specific, not task-specific" framing (written earlier
today) with the more careful "network-dependent under the tested
settings" -- a real, correct catch, since Limitations item 3 already
documents that 5-bus and IEEE-14/30 use different measurement-noise
and forecast-uncertainty settings, so "network identity" and "noise/
prior configuration" are confounded and cannot be cleanly separated
with only three networks; the paper cannot claim network identity
alone causes the difference while also listing that same confound as
an open limitation. Also scrubbed the "four-seed pilot vs.\ 16-seed"
narrative-comparison language this project's own rewrite (the update
immediately above) had reintroduced throughout Results~IV/Discussion/
Limitations/Conclusion -- a direct, correct application of the
author's much earlier, standing instruction that this kind of
audit-chronology narrative belongs in \texttt{STATUS.md}, not in the
paper's own prose, which this project should have remembered to apply
to its own rewrite and did not. Retitled Results~I and Results~IV
section headings to drop editorializing subtitles. Section-by-section,
this pass is an improvement; nothing in it was reverted or
second-guessed.

**Two things the author asked for directly, not yet in his own pass.**
(1) The Abstract still listed all six per-network-per-task gap means
individually (12 numbers) even after his own tightening pass --
flagged as "too many numbers." Replaced with a qualitative range
(roughly 0.01--0.03 points) plus the Bonferroni pass/fail count (three
of four), dropping all six individual mean$\pm$std pairs from the
Abstract; the full numbers remain exactly as before in
Table~\ref{tab:gapstats}. 259 $\to$ 164 words this round (this
Abstract has now been tightened three separate times across this
session, each time in response to a specific, named complaint about
density -- worth remembering as a standing preference for this
author, not a one-off).

(2) "Contributor" -- asked to have this removed, without further
specification. Checked both plausible meanings directly rather than
guessing: grepped every tracked file for "contributor"/"Claude"/
"Anthropic" (no text match anywhere), and queried GitHub's own
Contributors API directly (\texttt{gh api repos/.../contributors}) --
it lists only \texttt{yagizsarapli} and \texttt{github-actions[bot]};
\texttt{noreply@anthropic.com} does not register as a distinct
contributor there, since GitHub's contributor graph keys off registered
account emails, not \texttt{Co-Authored-By} trailer text. The most
likely explanation left, given 46 of this session's commits carry a
\texttt{Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>}
trailer (a standing harness default, not something chosen per-commit
this session): the author is seeing that trailer text directly on
individual commits' pages on GitHub, now that the repo is public. Fixed
going forward -- commits from this point in the session no longer carry
it, per the author's own instructions taking precedence over the
harness default. \textbf{Not done, and explicitly left as the author's
own call}: retroactively removing it from the 46 already-pushed
commits would require rewriting public git history (rebase or
\texttt{filter-repo}) and a force-push to a now-public repo --
genuinely destructive, not reversible the way a normal commit is, and
not something to do without an explicit, separate confirmation. Told
the author this directly rather than either doing it unprompted or
silently leaving it undone without explanation.

**Rebuild.** Full \texttt{pdflatex} $\to$ \texttt{bibtex} $\to$
\texttt{pdflatex} $\times 2$: clean, no undefined citations, no
overfull/underfull warnings of note. Page count now 8 (down from 10),
reflecting the author's own substantial prose tightening across the
28-commit pass, not any content loss -- spot-checked several trimmed
passages against \texttt{STATUS.md}'s own record of the underlying
numbers and found no number silently dropped or altered, only
narrative/hedging language shortened. \texttt{make\_arxiv\_submission.sh}
re-run end to end against the new, \texttt{arxiv/}-folder-free layout;
clean.

### Update, 2026-09-26 (same day, continued): git history rewritten to
remove the Claude co-author trailer, with explicit confirmation

The author sent a screenshot of the repo's GitHub sidebar showing
"Contributors 3": \texttt{yagizsarapli}, \texttt{github-actions[bot]},
and \texttt{claude} -- direct evidence against the update immediately
above, which had checked GitHub's contributors REST API and found only
2. Re-checked rather than assuming either check was right: confirmed
\texttt{github.com/claude} is a real, Anthropic-operated account
(178k followers, company "@anthropics"), and that the earlier API
check was almost certainly reading a stale/cached result -- GitHub's
contributor statistics are documented as being computed periodically,
not live, which is consistent with both checks being individually
accurate at the moment each was made.

Given this confirmed the concern was real, explicitly asked the author
whether to rewrite git history to remove the trailer from the 46
commits that had it, laying out the actual cost plainly (commit hashes
change, requires a force-push, the repo is now public) rather than
either doing it unprompted or leaving it undone without a clear ask.
Author confirmed: yes, rewrite and force-push.

**Execution.** Installed \texttt{git-filter-repo} (via \texttt{pip3
install --user}; not already present). Before touching anything:
tagged the pre-rewrite \texttt{HEAD} locally
(\texttt{pre-attribution-rewrite-backup}, kept as a local-only safety
reference, not pushed) and extracted the exact tail formatting of
several real commit messages to verify the intended regex against real
data rather than an assumption -- confirmed the trailer always
appears as the very last content, preceded by exactly one blank line,
with identical wording in all 46 commits (\texttt{grep -o
"Co-Authored-By:.*" | sort -u} returned exactly one unique line).
Dry-ran the stripping callback against all 46 real commit messages in
a standalone Python script (no repository mutation) before running
\texttt{git-filter-repo} for real, confirming all 46 cleaned
successfully with no trailer residue and no emptied message.

Ran \texttt{git-filter-repo --force --message-callback ...} with that
verified callback. Verified before pushing, not after: zero
\texttt{Claude}/\texttt{anthropic} matches anywhere in
\texttt{git log --all}; commit count on \texttt{main} unchanged at
143; \texttt{git diff pre-attribution-rewrite-backup HEAD} produced
\emph{no output at all} -- confirming the rewrite touched only commit
messages and not one byte of any file's content. Re-added the
\texttt{origin} remote (\texttt{git-filter-repo} removes it by design,
as its own safety measure against an accidental push to the wrong
place) and force-pushed. Confirmed via GitHub's own commit API
(\texttt{gh api repos/.../commits/main}) that the pushed commit's
message no longer contains the trailer.

**What is, and is not, yet confirmed.** The source of truth -- git
history itself -- is verified clean. The GitHub repo sidebar's
"Contributors" widget, screenshotted by the author both before and
immediately after the force-push, still showed \texttt{claude} in both
screenshots -- this is expected, not a sign the fix failed: GitHub
computes that specific statistic on a delay and caches it, the same
caching behavior that explains why the REST API check earlier today
returned a different (also-momentarily-accurate) answer. Told the
author this directly rather than either claiming success prematurely
or staying silent about the discrepancy between "history is clean" and
"the UI still shows the old cached count."

The local \texttt{pre-attribution-rewrite-backup} tag was left in
place (not pushed, not deleted) as a recovery point in case anything
about the rewrite needs to be revisited.

### Update, 2026-09-26 (same day, continued): a detailed external
review of the current draft, four safe polish fixes applied, one
substantial suggestion (a noise/prior robustness sweep) left for the
author's own call

The author pasted a section-by-section external review of
"Yagiz\_Sarapli\_arXiv\_v1\_CLEAN" -- markedly more positive than the
first review earlier in this session, explicitly praising the paper
for not overclaiming (the 5-bus null, the IEEE-14-detection Bonferroni
failure, the honest soft/hard real-time distinction, the already-tight
Abstract) and confirming several things this project had already done
right were landing as intended.

**Decoded, not guessed, one garbled instruction.** The review closed
with a rushed, heavily-typo'd complaint about subheadings looking
"confusing" and "both in roman letters." Rather than ask for
clarification on something checkable, read the actual section titles
directly: \texttt{\textbackslash section\{Results I: 5-Bus
Ablation...\}}, \texttt{...\{Results II: Real-Time
Feasibility...\}}, \texttt{...\{Results III: ...\}},
\texttt{...\{Results IV: ...\}} -- meaning the rendered headings
literally read "IV. RESULTS I: ...", "V. RESULTS II: ...", "VI.
RESULTS III: ...", "VII. RESULTS IV: ...": the auto-numbered
IEEEtran section counter (Roman numerals) sitting directly next to a
manually-typed "Results I/II/III/IV" label that never matched it,
since Introduction/Related Work/Methodology are I/II/III. Confirmed
via grep that "Results I/II/III/IV" appears nowhere else in the prose
(only in these four \texttt{\textbackslash section\{\}} lines), so
stripping the redundant prefix was risk-free. This is almost certainly
what "ikisi de roman harfleriyle yaz\i{}yo" meant.

**Applied directly** (all in \texttt{paper/main.tex}, no design
decision required):
- Removed the "Results I/II/III/IV:" prefixes from all four section
  titles.
- Added the reviewer's suggested explicit sentence to Limitations item
  3: the feature representation itself (fixed per-bus columns at
  5-bus vs.\ size-agnostic at IEEE-14/30) is a second, separate
  confound alongside the already-documented noise/prior mismatch --
  a cross-network comparison reflects network identity, measurement/
  prior regime, \emph{and} feature representation jointly, and
  isolating any one factor is explicitly out of this paper's scope.
- Retitled Section~V from "Real-Time Feasibility" to "Soft Real-Time
  Feasibility," and made the matching Introduction sentence say the
  same ("soft real-time feasibility on the 5-bus implementation") --
  a precise, minimal fix (one word, in the two places that introduce
  the concept) rather than rewriting every later mention, since the
  body text (median/p95/p99 within budget, rare maxima above it) was
  already carefully hedged and didn't need touching.
- Trimmed the "network-dependent/setup-dependent" framing, which
  repeated five times across Interpretation/Discussion-open/
  Discussion-close/Limitations/Conclusion -- folded the Discussion's
  redundant closing "Overall, ..." paragraph into its own opening
  paragraph (same content, said once instead of twice in the same
  section) and removed the closer entirely; the other three
  occurrences (Interpretation, Limitations, Conclusion) each do
  distinct work in their own section and were left alone.

**Left explicitly for the author, not attempted.** The review's own
top suggestion -- a robustness sweep matching measurement-noise/prior-
uncertainty settings across all three networks (e.g.\ 5\% prior
uncertainty at 5-bus, or 2.5\% at IEEE-14/30, or a per-unit/relative
normalization) to test whether the cross-network pattern survives --
is a genuine new experiment, not a text edit: it requires choosing
which direction to normalize (a real methodological decision, not a
mechanical one) and modifying the actual noise-generation code before
any rerun. Given this project's own standing practice throughout the
session (the author drives research-methodology decisions; this
project executes and verifies), this was surfaced as an open question
for the author rather than started unprompted.

**Rebuild.** Full \texttt{pdflatex} $\to$ \texttt{bibtex} $\to$
\texttt{pdflatex} $\times 2$: clean, no undefined citations, no
overfull/underfull warnings. Still 8 pages. Visually confirmed the
fixed section numbering (\S IV, \S V now read cleanly) and the trimmed
Discussion on the rendered PDF. \texttt{make\_arxiv\_submission.sh}
re-run; clean.

### Update, 2026-09-26 (same day, continued): a second, independent
force-push -- STATUS.md deleted and restored, and the 5-bus
prior-uncertainty robustness sweep completed with a clean result

**A second force-push landed on origin** while this project's own
polish commit was still local, from a base that predated this
project's own \texttt{git-filter-repo} rewrite (confirmed: the new
history is not a descendant of that rewrite's commit) but that
nonetheless also carried zero \texttt{Claude}/\texttt{anthropic}
mentions -- almost certainly an independent history-cleaning pass run
in parallel, from a different tool or session, working from the same
underlying review the author was acting on. Investigated rather than
assumed before touching anything further:

- The exact same heading-numbering fix this project had just applied
  (dropping the redundant "Results I/II/III/IV:" prefixes) was already
  present, done independently and, in places, better -- Discussion's
  closing paragraph rewritten more sharply ("these between-system
  differences are descriptive rather than causal... the within-system
  comparisons remain the controlled contrasts") and the feature-
  representation confound given its own dedicated Limitations item
  (4) rather than folded into item 3. This project's own version of
  all three fixes was therefore fully redundant; discarded rather than
  merged (tagged \texttt{my-redundant-polish-2026-09-26} first, for
  safety, before \texttt{git reset --hard origin/main}).
- \textbf{\texttt{STATUS.md} itself had been deleted} in a commit
  titled "Finalize arXiv v1 manuscript and reproducibility cleanup" --
  confirmed via \texttt{git log --diff-filter=D -- STATUS.md}, not a
  rename (grepped the full origin tree; no equivalent file). README's
  own references to \texttt{STATUS.md} were also removed in the same
  pass, consistent with a deliberate rather than accidental deletion.
  This directly contradicts the author's own much-earlier, explicit
  instruction earlier in this project (keep the audit chronology out
  of the paper specifically \emph{by} keeping it in \texttt{STATUS.md}
  instead) -- flagged this tension directly to the author rather than
  silently restoring it or silently accepting the deletion. Nothing
  was at risk of being permanently lost either way: this project's own
  local working copy, plus the \texttt{pre-attribution-rewrite-backup}
  tag, retained the full file throughout.
- The author's reply: restore \texttt{STATUS.md}, and separately asked
  for a recommendation on keeping the GitHub repo public vs.\ private
  going forward, plus a current-state check. Restored from this
  project's most complete local copy (\texttt{git show
  my-redundant-polish-2026-09-26:STATUS.md}, 2448 lines, includes every
  entry through the previous update above).

**The robustness sweep -- the reviewer's top suggestion, and the thing
flagged as "the author's own call" two updates above -- turned out to
already be in progress when checked.** \texttt{34\_prior\_uncertainty\_
sensitivity.py} plus a new \texttt{.github/workflows/
prior-sensitivity.yml} CI workflow re-run the 5-bus ablation at
\texttt{prior\_sigma=0.05} (matching IEEE-14/30's 5\%, instead of
5-bus's own 2.5\%) across the identical 16-seed set this project
established two updates above, as a 16-way parallel CI matrix job, one
seed per job, each uploading its own result as a build artifact (not
committed to the repo). Two workflow runs were in flight when first
checked (\texttt{gh run list}): an older, non-parallelized version
still running past 45 minutes, and a newer, parallelized version that
completed successfully in under 20. Did not touch either run or try to
cancel the slower one -- not this project's infrastructure to
interrupt.

Downloaded the completed run's 16 per-seed artifacts directly
(\texttt{gh run download}), combined them, and recomputed full
statistics independently with \texttt{scipy} (same methodology as
every other seed-count analysis in this project): at the matched 5\%
prior uncertainty, 5-bus detection gives mean $+0.0006\pm0.0101$
(95\% CI $[-0.0047,+0.0060]$, $t$-test $p=0.807$) and localization
gives mean $+0.0012\pm0.0174$ (95\% CI $[-0.0080,+0.0105]$, $t$-test
$p=0.778$) -- both still clearly null, both nearly identical in
magnitude and character to the original 2.5\%-prior result
(det.\ $-0.0004\pm0.0086$, loc.\ $+0.0029\pm0.0169$). This is a clean,
reassuring, confirmatory result: the 5-bus null is not an artifact of
its tighter prior-uncertainty setting relative to IEEE-14/30 -- at
least for this one piece of the cross-network confound, normalizing it
away changes nothing. Added this directly to Limitations item 3 in
\texttt{paper/main.tex} (precisely scoped: rules out the
prior-uncertainty mismatch specifically, not the still-untested
measurement-noise-scaling or feature-representation confounds).
Rebuilt clean, still 8 pages.

Not yet done: the CI-uploaded artifacts are ephemeral (GitHub's
default retention), not committed anywhere in the repo, so the sweep's
raw per-seed numbers are only in this \texttt{STATUS.md} entry and this
project's own \texttt{/tmp} download right now, not in \texttt{results/}
or any tracked file -- worth the author's attention if this sweep's
data should be preserved more durably than an Actions artifact.

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

**SUPERSEDED — this section is the pre-audit historical record, kept
for the research log's own chronology.** It reflects the state of
understanding *before* the post-submission methodological audit
(Section 6 above), which found and fixed an oracle-leaked residual
feature, test-set model selection, and a second 5-bus feature-leakage
bug, and materially changed the conclusion described below (most
notably: detection at IEEE-30 does now show a consistent-sign
advantage, which item 1 below says it doesn't). **The current,
authoritative result is Section 6 above and `paper/main.tex` /
`PAPER_DRAFT.md` / `README.md`** — do not cite anything below this line
as this project's present conclusion.

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
