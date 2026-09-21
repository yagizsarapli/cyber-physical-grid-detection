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
audit (§5's own "second instance" writeup) — once fixed,
topology-fusion shows no advantage there either. Residual-only
detection (0.750) outright beats topology-fusion (0.737), and even the
corrected topology-free ablation (residual_plus_prior, 0.743) edges it
out; topology-fusion keeps only a narrow, single-run (not multi-seed)
localization edge (0.560 vs. 0.533-0.547). Read together with §2.5,
this is not "the advantage doesn't transfer to scale" — it's the same
null result, reached independently twice, with two unrelated
implementations that each independently needed the same fix, at two
network sizes.

One honest limit found by testing it directly: this advantage
(residual+prior fusion, not topology) is **pattern recognition of
known attack types, not zero-day generalization**. A model trained
with one cyber attack type entirely withheld catches almost none of it
at test time (0–4% recall, vs. 89–100% when that type is in training).
Framed accurately, not overclaimed — see §4.

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

| | Original (easy, n=72) | Hard (n=72) | **Hard (n=300)** | **Harder: +2x forecast noise (n=300)** |
|---|---|---|---|---|
| Best detection (bal-acc) | topology_fusion 1.000 | prior_only 0.903 | **topology_fusion 0.950** (prior_only 0.873) | **topology_fusion 0.953** (prior_only 0.83–0.90) |
| Best localization (top-1) | topology_fusion 1.000 | topology_fusion 1.000 | **topology_fusion 0.980** (prior_only 0.780) | **topology_fusion 0.980** (prior_only 0.767) |
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

**Corrected n=500 result** (best model per feature set):

| | 5-bus (n=300 test) | **IEEE 14-bus, n=500 (corrected)** |
|---|---|---|
| Detection | topology_fusion 0.950 = residual_plus_prior 0.950 | residual_only/LR **0.750** best; residual_plus_prior 0.743, topology_fusion 0.737 close behind — neither the ablation nor topology_fusion leads |
| Localization | residual_plus_prior **0.993** > topology_fusion 0.980 | topology_fusion **0.560**, narrowly ahead of residual_only/prior_only (0.547) and residual_plus_prior (0.533) — single run, no seeds/CI at this scale |

**Honest reading, updated twice now.** First pass (before §2.5):
"topology helps at 5-bus, not at 14-bus — scale-dependent." Second
pass (after §2.5, before this fix): "topology never helped at 5-bus
either — 14-bus independently confirms a null result, and wasn't
touched by the bug." Both were wrong in the same specific way: neither
checked whether the 14-bus "baselines" were actually topology-free,
and they weren't — checked now, they weren't. Correct reading: **two
separately-written pipelines each needed the identical class of fix**
(exclude relational/degree columns from what's supposed to be a
topology-free baseline), found independently, at different times, by
auditing one after finding the bug in the other. Once both are
corrected, they agree: no feature set is confidently ahead of the
topology-free ablation at either scale — detection is unambiguous at
both scales (5-bus: exact tie; 14-bus: residual\_only *and*
residual\_plus\_prior both edge out topology\_fusion), localization is
a real (5-bus, multi-seed) or narrow-and-unreplicated (14-bus,
single run) story depending on scale. That the same mistake was made
twice, independently, is itself worth noting: it suggests this is an
easy trap in "topology-aware feature" engineering generally, not a
one-off slip in one script.

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

1. **No topology-specific advantage was found at either scale tested**
   (§2.5, §5), using two unrelated feature-engineering implementations.
   Both are still limited to synthetic networks/loads; a third,
   real-topology system (e.g. IEEE 30-bus, or a real feeder) would
   further test whether this null result itself generalizes, but the
   open question is no longer "does topology help at scale" — it's
   "does this class of relational feature ever help, on any network,"
   which the evidence so far says no to, twice.
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
   it wasn't; corrected across 2 stress conditions and 4 seeds. What's
   left on scale: try a third standard system (e.g. IEEE 30-bus) to
   see whether this null result itself generalizes further, or is
   specific to the two topologies tested so far.
2. Some form of unseen-attack robustness beyond pure supervised
   classification (e.g. anomaly-based pre-filter, or bridging Phase
   14/17's richer taxonomy) — §4 shows the current approach has none,
   and that's worth addressing rather than only disclosing.
3. A direct numeric comparison against arXiv:2605.17256's own
   architectures on the same data, if their code/data is available.
4. Re-run §4's zero-day test on `residual_plus_prior` specifically
   (currently only verified for `topology_fusion`) — flagged in
   Limitations item 2, not expected to differ but not yet checked.
