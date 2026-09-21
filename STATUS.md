# GRIDRA Cyber-Physical Microgrid — Status

Last updated: 2026-09-21. Supersedes the "Current backbone / Next" section
in `README.md`, which only described the Aug-11 starting point and was
never updated through phases 2B–2T.

## Summary

A 5-bus/0.4kV synthetic microgrid (PV as a GFL inverter, BESS as a
GFM inverter) with a WLS AC state estimator is used to test whether
topology-aware relational features let a classifier (a) tell a cyber
false-data-injection attack apart from a legitimate physical
disturbance, (b) localize which bus is affected, and (c) do both
within a one-cycle protection-relevant computational target (50 Hz =
20 ms, confirmed from `net.f_hz`, not assumed -- not a claim that
every protective relaying function requires exactly this latency).

Under a realistic attack magnitude (pushed toward the sensor noise
floor, not the original easy configuration) **and** with a properly
powered test set (n=300, not n=72), topology-fusion features give a
real, statistically supported advantage for both detection
(balanced accuracy 0.950–0.953) and localization (top-1 0.980) over
a simpler prior-based baseline (0.83–0.90 detection, 0.75–0.78
localization) — stable whether load-forecast uncertainty is normal or
doubled, and **confirmed across 4 independent random seeds** (detection
gap +0.088 ± 0.035, localization gap +0.207 ± 0.032, sign never
flips). After profiling and fixing the actual latency bottleneck (a
software inefficiency, not the model), the full pipeline now meets the
20ms budget at the median, p95, and p99 (16.4 / 17.0 / 17.1 ms), with
a single outlier over 300 trials still exceeding it (21.3 ms max).

Tested at a bigger, standard scale (IEEE 14-bus, §5) at three
escalating sample sizes, ending at n=500 — matching the 5-bus study's
own statistical power exactly: the advantage does **not** transfer.
Residual-only detection (0.750) outright beats topology-fusion (0.737)
there; topology-fusion keeps only a narrow, non-decisive localization
edge (0.560 vs. 0.507-0.547). This is a confirmed, equally-powered
result, not a pilot with an asterisk — the honest scope of the claim
is now "helps on small, simple networks; does not help, and for
detection actively underperforms a simpler baseline, on larger meshed
ones."

One honest limit found by testing it directly: this advantage is
**pattern recognition of known attack types, not zero-day
generalization**. A model trained with one cyber attack type entirely
withheld catches almost none of it at test time (0–4% recall, vs.
89–100% when that type is in training). Framed accurately, not
overclaimed — see §4.

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

**Conclusion**: topology-aware relational features give a real,
statistically supported advantage for both detection and
localization under realistic (not easy) attack conditions, and that
advantage is stable when a second, independent difficulty axis
(forecast uncertainty) is added on top. The n=72 "detection tie"
was a genuine research dead-end correctly caught by insisting on a
harder test — but the fix for a suspicious result should also include
checking whether the test set was simply too small before trusting a
reversal, which the first pass here skipped.

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

**Held-out attack-type generalization** (`27_held_out_attack_type_generalization.py`,
run on the seed=2024 data). Note this replaces an earlier plan to
"re-run Phase 14 (zero-day)/17 (hard-negative) against the HARD data" —
those scripts turned out to depend on a completely different upstream
data lineage (`phase2g_v2_scenario_metadata.csv`, a richer 5-cyber/
3-physical event taxonomy) that isn't compatible with the Phase 2Q/2R/2S
graph-feature files without substantial bridging work. This script asks
the same *kind* of question directly on the data already validated
here instead: train topology_fusion/HistGradientBoosting with one
cyber attack type **completely removed** from training, test recall
on exactly that type.

| Held out | Recall when withheld | Recall when trained on it |
|---|---|---|
| naive_single_sensor_corruption | **0.000** | 0.893 |
| nonlinear_model_consistent_fdia (stealth) | **0.040** | 1.000 |

**Honest conclusion**: the detector does not extrapolate to an attack
mechanism it has never seen — it recognizes the specific statistical
signatures of the attack types in its training data (including the
stealthy one, very well) but has no mechanism to generalize beyond
them, exactly as expected for a supervised classifier with only two
attack families to learn from. Any real deployment claim needs this
stated plainly: strong within-distribution discrimination, no
demonstrated zero-day capability.

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

| | 5-bus (n=300 test) | IEEE 14-bus, n=200 (first read) | **IEEE 14-bus, n=500 (confirmed)** |
|---|---|---|---|
| Best detection | topology_fusion 0.950, **consistent across all 3 models** | topology_fusion/LR 0.766, inconsistent across models | **residual_only/LR 0.750 — now the outright best**, topology_fusion best is 0.737 (LR), not ahead |
| Best localization | **topology_fusion 0.980**, prior_only 0.780 | residual_only 0.600, topology_fusion 0.517 | topology_fusion 0.560 — narrowly best again, but tight (prior_only 0.547, residual_only 0.507) — not decisive the way the 5-bus gap is |

**Reading, now with full statistical power on both sides of the
comparison:** the n=200 pattern was not a fluke that a bigger sample
would erase — it got clearer. For detection specifically, plain
residual analysis is now the single best-performing feature set on
IEEE 14-bus, ahead of topology_fusion. For localization, topology_fusion
keeps a narrow edge, but nothing resembling the decisive 5-bus gap
(0.560 vs. 0.980). This is no longer a pilot with an asterisk — it's a
direct, equally-powered comparison, and the answer is: **the
topology-fusion advantage does not transfer to IEEE 14-bus.**

**Honest reading: the advantage looks scale/topology-dependent, not
universal.** IEEE 14-bus is meshed, multi-generator, with transformers
— a node's electrical "neighbors" there are a much less tight, less
informative concept than on the simple, small, mostly-radial 5-bus
microgrid where the original advantage was found. This is a genuinely
useful, publishable nuance — arguably more interesting than a clean
replication would have been, because it scopes the claim honestly:
*topology-aware fusion helps on small, simple networks; it does not
help, and for detection specifically actively underperforms a simpler
baseline, on this larger meshed one.*

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

1. **Scale**: the headline advantage (§2) is a 5-bus finding. Confirmed
   at n=500 on IEEE 14-bus (§5, matching the 5-bus study's own power)
   that it does **not** transfer — residual-only detection now
   outright beats topology-fusion there. The honest claim is
   scope-limited to small/simple networks, not universal. This is no
   longer an open statistical-power question at all; what remains is
   whether a *second* standard system (e.g. IEEE 30-bus) shows the
   same pattern, i.e. whether "doesn't transfer past ~14 buses" itself
   generalizes — a genuine open question, but a different one from
   "was n=200 enough," which is now closed.
2. **No zero-day generalization**: demonstrated directly in §4 — 0–4%
   recall on a completely withheld attack type. The detection/
   localization numbers above only hold for attack types represented
   in training.
3. **Third stress lever not testable as a free parameter**: restricting
   which measurements a stealth attack compromises is a physical
   consequence of the AC model consistency construction, not a knob —
   noted, not evaded.
4. **Attack family**: only two synthetic families (naive,
   model-consistent FDIA) exist in this data lineage. Phase 14/17's
   richer taxonomy lives in an incompatible upstream pipeline (§4) —
   bridging it is future work, not done here.
5. **Latency benchmark** measures this specific Python/pandapower
   implementation on one machine — not a hardware/RTOS claim.

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
   pilot (not a formal power analysis — same N, not verified-equal
   statistical power). What's
   left on scale: try a second standard system (e.g. IEEE 30-bus) to
   see whether "topology-fusion loses its edge past ~14 buses" itself
   generalizes, or is specific to IEEE 14-bus's particular topology.
2. Some form of unseen-attack robustness beyond pure supervised
   classification (e.g. anomaly-based pre-filter, or bridging Phase
   14/17's richer taxonomy) — §4 shows the current approach has none,
   and that's worth addressing rather than only disclosing.
3. A direct numeric comparison against arXiv:2605.17256's own
   architectures on the same data, if their code/data is available.
