# Cyber-Physical Grid Detection

Does giving a cyber-attack detector *topology* -- not just raw sensor
residuals, but how each meter's reading compares to its electrical
neighbors -- actually help it (a) tell a cyberattack apart from a
normal physical disturbance, (b) find where the attack is, and (c) do
both fast enough for a protective relay to act on?

Short answer: **it depends which network, not which task** -- yes at
both standard IEEE test systems (14-bus, 30-bus), on both detection
and localization; no at a small custom 5-bus microgrid, on either
task. That answer took two full rounds of auditing our own pipeline
and catching our own mistakes, then a 4x seed-count replication to
confirm the pattern was real and not four-seed noise. See below for
what that means and where to find the details.

![5-bus vs. IEEE 14-bus vs. IEEE 30-bus: detection and localization by feature set](figures/paper_fig4_scale_comparison.png)

## The headline result

We built topology-relational features (comparing each meter's reading
to its electrical neighbors) expecting them to help. A first pass
seemed to confirm it: on a small 5-bus inverter-dominated microgrid,
`topology_fusion` beat a simpler prior-based baseline (`prior_only`)
at both detecting a stealthy false-data-injection attack (balanced
accuracy 0.950 vs. 0.873) and localizing it (top-1 accuracy 0.980 vs.
0.780) -- validated across 4 independent random seeds, gap never
flipping sign.

That comparison turned out to be unfair, in a way that took two
separate rounds of auditing to fully uncover. Round one: `prior_only`
alone is missing residual information `topology_fusion` has always
had; the actual control is `residual_plus_prior` -- residual and prior
features combined, with **no** topology at all. Building that ablation
surfaced a real bug -- topology-relational columns are *named* after
the residual/innovation quantities they compare (e.g.
`neighbor_mean_residual`), so the original name-matching feature-set
split let about half of them leak into what were meant to be
topology-free baselines. Round two, later, on a deliberately
adversarial re-read of the same code: those "topology-free" 5-bus
localization baselines also secretly carried node degree and
device-role flags that reveal which buses are even attack-eligible --
a second, independent leak, found only once this project's own central
finding was audited a second time. A third fix, found in a later review
round, restricted localization *training* to cyber-scenario nodes only
(matching the convention the IEEE-14/30 pipelines already used, despite
a comment elsewhere claiming the two matched) -- evaluation was always
cyber-only, only the training-set composition changed. With **all
three** fixed, the fair comparison is a near-exact tie on **both
tasks**: detection 0.945 vs. 0.938 (4-seed mean, gap +0.008±0.012, sign
unstable) and localization 0.958 vs. 0.965 (gap -0.007±0.016, sign
unstable, including one exact tie across the 4 seeds) -- since
quadrupled to a 16-seed mean (see the Update further below): detection
0.936 vs. 0.936 (gap -0.000±0.009) and localization 0.964 vs. 0.961
(gap +0.003±0.017), the null result if anything tighter, not
overturned, at 4x the seeds. An earlier, once-corrected pass had reported
localization as a topology-free *win* (0.980 vs. 0.993, before the
degree/role leak was found); that asymmetry is gone once the
comparison is fair on both sides. **Combining residual and prior
information explains the entire advantage; explicit topology-awareness
adds nothing measurable at 5-bus, on either task.**

A separately-implemented, size-agnostic pipeline for two bigger,
standard networks (IEEE 14-bus and IEEE 30-bus, n=500 replications,
300 test scenarios each) needed the same two rounds of correction --
first an independent rediscovery of the feature-leakage bug (a cruder,
explicit-list version, not a name-substring collision), then, in the
same later, adversarial audit that found 5-bus's second leak, a
completely different and more serious problem: the "residual" feature
at both networks had been computed from the simulator's *hidden
ground-truth state* -- something no real detector could ever observe
-- instead of the actual, deployable WLS measurement residual the rest
of this project already used. The same audit found that model
selection at all three networks (5-bus included) had been picking
whichever of three candidate models scored best on the test set
itself, rather than on a held-out validation split. Fixing both and
re-running every network at 4 seeds changed the three-network picture
a second time, into something more complicated than either earlier
version:

- **Detection**: null at 5-bus and IEEE-14 (gaps of +0.008±0.012 and
  -0.003±0.013, sign unstable at both) -- but a gap positive in all 4
  tested seeds at IEEE-30 (+0.039±0.029). This specific finding is new
  to this last audit pass; no earlier version of this analysis,
  corrected or not, found a detection advantage anywhere.
- **Localization**: null at 5-bus and IEEE-30 (-0.007±0.016 and
  +0.002±0.023, both sign-unstable) -- but a gap never negative across
  the same 4 seeds at IEEE-14 (+0.023±0.021), the one finding across
  this whole project that has survived every round of correction so
  far -- including a later round that removed raw node degree
  (`node_n_neighbors`) from `topology_fusion` at both IEEE networks (it
  was already removed at 5-bus for the same reason: attacks only ever
  target load buses, so degree alone could act as a static
  target-eligibility prior rather than a genuine neighbor comparison)
  -- though with roughly double the uncertainty an earlier,
  less-audited pass reported.

**Update, 2026-09-25**: both gaps above were flagged as consistent-sign
but not statistically confirmed at 4 seeds (see the original caveat
this replaced, below) -- so they were quadrupled to 16 seeds,
identically across all three networks and both tasks, to resolve it
properly. The picture changed again, in a cleaner direction: IEEE-14
detection, previously null at 4 seeds (-0.003±0.013), now also shows a
small positive gap (+0.009±0.012); IEEE-30 localization, previously
null (+0.002±0.023), now also shows one (+0.013±0.016). The original
two findings held up and tightened: IEEE-30 detection (+0.028±0.019)
and IEEE-14 localization (+0.018±0.017). 5-bus remained null on both
tasks throughout (-0.000±0.009 detection, +0.003±0.017 localization).
The pattern is therefore network-specific, not task-specific: both
standard IEEE systems now show a small, mostly confirmed advantage on
both tasks; the custom 5-bus microgrid shows none on either.

**The current statistics, stated plainly**: at 16 seeds, four of the
six network/task gaps have a t-based 95% CI excluding zero (two-sided
t-test p<0.05): IEEE-14 detection (p=0.011), IEEE-14 localization
(p<0.001), IEEE-30 detection (p<0.001), IEEE-30 localization (p=0.007).
A distribution-free sign test agrees for three of the four (p<0.001,
p<0.001, p=0.018 respectively) but not IEEE-14 detection specifically
(p=0.059, 11 of 15 non-zero seeds positive) -- the weakest-supported of
the four. A Bonferroni correction for testing six gaps at once (needing
p<0.0083 for family-wise 95% confidence) leaves the same three
surviving; IEEE-14 detection does not clear it. Both 5-bus gaps remain
clearly null (p=0.849, p=0.500 by t-test). Sixteen seeds is a
convenience choice, not a formal power calculation, and effect sizes
remain small (0.009-0.028 points) relative to the much larger
residual+prior-vs-either-alone gain reported above.

Put plainly: the two standard IEEE systems now behave alike on both
tasks, and differently from the small custom 5-bus microgrid, which
shows no gap on either task -- not, as the 4-seed pilot suggested, a
different single task favored at each network. Three separately-written
pipelines needing the same category of feature-leakage fix, and then a
second, more serious category of mistake (oracle information reaching a
feature meant to be deployable) surviving a first full audit only to be
caught in a second, more adversarial one, remains the finding we'd want
a reader to take away alongside the numbers: neither kind of bug was
visible from results alone -- both were found only by reading
feature-computation code line by line against "could a real, deployed
detector actually compute this." Full per-seed numbers for all 16
seeds, and the complete before/after account of both audit rounds plus
the seed-count expansion, are in `STATUS.md` §6.

Two further things remain central. First, the detector does **not**
generalize to an attack type it never trained on (0-1.3% recall when
one of two attack families is withheld, under the corrected symmetric
protocol). Second, the full decision pipeline (state estimation ->
feature computation -> classification) was profiled against a
one-cycle (20 ms at 50 Hz) computational target. Two documented independent N=300 reruns on Apple arm64 / macOS
15.6.1 (Python 3.12.4, pandapower 2.14.10, NumPy 1.26.4,
pandas 2.2.2, scikit-learn 1.4.2), both with 0 convergence failures,
gave 16.155-17.015 / 16.429-18.211 / 16.952-19.190 ms across the
median/p95/p99 and 89.067-92.657 ms maxima. Thus median, p95, and p99
remain inside the 20 ms target in both reruns while rare solver-tail
events exceed it. Stage medians remain dominated by WLS state
estimation (15.855-16.664 ms) rather than residual/feature/inference
work.

**Full story, with every number and why it's trustworthy: [`STATUS.md`](STATUS.md).**

## Repository map

```
exploration/            Phases 2A-2P (01..20_*.py): microgrid model,
                       inverter models, WLS state estimation, early
                       cyber-vs-physical benchmarks -- built before
                       this session. Two files are still live: 01_
                       microgrid_topology.py (loaded by 21_*_HARD.py
                       below) and 08_multirate_operating_dataset.py
                       (run directly -- see "Reproducing this"; its
                       output is required by every script below it).
                       Everything else that was ever in this folder is
                       now under exploration/archive/ -- superseded
                       design work, not read by any script below, kept
                       as the record of how the current design was
                       arrived at.

archive/                Four early, pre-stress-test versions of the
                       21-23 pipeline (the plain, non-"_HARD" 21 and
                       22, and both a plain and a _FIXED 23) --
                       superseded once the stress-test attack
                       magnitude and the localization fixes below were
                       introduced. Not read by any script below or by
                       the paper; kept for the record, not deleted.

21_*_HARD.py           Attack generator, magnitude reduced toward the
                       sensor noise floor (this session's stress test)
22_*_HARD.py/HARDER.py  Dataset generation on top of 21_*_HARD
23_*_FIXED/HARD/HARDER.py  Detection + localization evaluation
24_realtime_latency_*.py   Phase 2T: latency measurement and fixes
25_wls_overhead_diagnostic.py   Root-cause split of the latency gap
26_multi_seed_replication.py    16-seed statistical validation
                       (quadrupled from an original 4-seed pilot,
                       2026-09-25, to resolve the cross-network pattern
                       below statistically)
27_held_out_attack_type_*.py    Zero-day generalization test
28_ieee14_scale_replication.py  The IEEE 14-bus scale study
                       (confirmed at n=500, matching the 5-bus study's
                       own 300-scenario test-set size)
29_paper_figures.py    Generates figures/paper_fig*.png from results/
30_ieee14_multi_seed_replication.py  16-seed replication of 28, same
                       pattern as 26 -- confirms the IEEE-14
                       localization result isn't single-run noise, and
                       at 16 seeds also resolves an IEEE-14 detection
                       gap the original 4-seed pass had called null
31_ieee30_scale_replication.py  A third network (IEEE 30-bus), direct
                       copy of 28 with the network swapped -- tests
                       whether IEEE-14's pattern generalizes
32_ieee30_multi_seed_replication.py  16-seed replication of 31, same
                       pattern as 30 -- re-run three times: first found
                       IEEE-30's single-run match to IEEE-14 was a seed
                       coincidence (localization null); re-run again
                       after 31's oracle-residual/model-selection audit
                       fix found a detection gap positive in all 4
                       tested seeds instead, new to that later pass; a
                       third pass (2026-09-25) quadrupled to 16 seeds
                       and additionally resolved the localization gap
                       (previously null at 4 seeds) as statistically
                       confirmed too
33_feature_redundancy_diagnostic.py  Out-of-fold Ridge/R^2 check that
                       the paper's Discussion cited but no earlier
                       script actually computed -- written to close
                       that gap, confirms the qualitative redundancy
                       finding on the corrected features

STATUS.md              Research log -- what was done, in what order,
                       every real number, every correction made along
                       the way. Start here.
RELATED_WORK.md         57 sources found; 41 read past search-summary
                       level, with two invented figures caught and
                       corrected in the process. See its own honesty
                       caveat before citing anything from it unstarred.
PAPER_DRAFT.md          The findings above, written as a paper draft
                       (Markdown).
paper/main.tex          Full IEEEtran manuscript/preprint version,
                       with a verified BibTeX bibliography
                       (paper/references.bib). A venue-specific
                       condensed version should be prepared separately.

data/, results/         Generated by the scripts above; gitignored
                       (regeneratable, kept out of the repo to stay
                       small -- see below to reproduce).
figures/                The 4 figures embedded in the paper
                       (`paper_fig1-4_*.png`), generated by
                       `29_paper_figures.py`. Per-phase diagnostic PNGs
                       are gitignored now (regeneratable byproducts,
                       not part of the paper -- rerun the relevant
                       numbered script to get them back locally).
```

The numbering is a research log, not a build order: each script is a
snapshot of one step, and a `_FIXED`/`_HARD`/`_HARDER` suffix means
"same idea, corrected or stress-tested" rather than "replaces the
original." Where the original sits alongside its corrected/stress-
tested version at the root (e.g. `24_realtime_latency_benchmark.py`
next to `_WARMSTART`/`_OPTIMIZED`/`_FINAL`), that's on purpose: each
one is a distinct, individually cited measurement in the paper's own
optimization narrative, not a discarded draft. Where a root-level
original predates a since-introduced stress test and nothing in the
current paper depends on it any more (the plain `21`/`22` and the
plain/`_FIXED` `23`), it has been moved to `archive/` instead, to keep
the root directory limited to what the current paper actually uses.
`STATUS.md` narrates the actual sequence and reasoning; the file list
alone won't make sense without it. The `exploration/` vs. root-level
split follows the same logic at folder granularity: `exploration/` is
"how we got here," root is "what the paper is," and each level's own
`archive/` holds what neither the paper nor any live script still
reads.

## Reproducing this

```bash
pip install -r requirements.txt
```

Everything reads from and writes to `data/` and `results/` (created
automatically, gitignored). A reasonable path through the numbered
scripts, in the order this session actually ran them:

```bash
python3 exploration/08_multirate_operating_dataset.py   # generates data/phase2e_7day_operating_dataset.csv, required by every script below -- run this first on a clean clone
python3 22_graph_ready_protected_prior_telemetry_HARD.py --n-rep 500
python3 23_topology_aware_cyber_physical_localization_HARD.py
python3 26_multi_seed_replication.py          # ~1h, 16 full reruns (quadrupled from an original 4-seed/~15-20min pilot), then auto-restores the primary-seed (20260812) snapshot -- 27 and 33 below both read data/phase2s_hard_graph_feature_matrix.csv directly and need that exact snapshot, not whichever seed this loop last ran
python3 27_held_out_attack_type_generalization.py
python3 24_realtime_latency_benchmark_FINAL.py   # also records results/phase2t_latency_environment.json (machine + software versions) for reproducible latency reporting
python3 28_ieee14_scale_replication.py --n-rep 500   # slower per-replication than the 5-bus scripts; this is the confirmed run (§7 of STATUS.md) -- --n-rep 200 was an earlier, superseded pass
python3 33_feature_redundancy_diagnostic.py   # must run HERE, not after 30 below -- 30's own seed loop overwrites data/phase2v_ieee14_node_feature_matrix.csv with its last seed (2024), and this script's IEEE-14 R^2 needs the primary-seed (20260921) run 28 just produced, not a multi-seed leftover (see this script's own --help). Its 5-bus side is separately safe because 26 above already restored that primary-seed snapshot.
python3 30_ieee14_multi_seed_replication.py   # ~2h, 16 full reruns of 28 (quadrupled from an original 4-seed/~30-40min pilot); does NOT auto-restore the primary-seed snapshot afterward (unlike 26 above) -- re-run 28 before 33 if you need that snapshot again later
python3 31_ieee30_scale_replication.py --n-rep 500
python3 32_ieee30_multi_seed_replication.py   # ~2h, 16 full reruns of 31 (quadrupled from an original 4-seed/~30-40min pilot); also does not auto-restore the primary-seed snapshot afterward
python3 29_paper_figures.py
```

Each script prints what it saves and where. Random seeds are fixed in
each script (see `STATUS.md` for which seeds were used where) --
re-running should reproduce the numbers in `STATUS.md` and
`PAPER_DRAFT.md` exactly, modulo floating-point/BLAS nondeterminism.

## Building the paper

`paper/main.tex` + `paper/references.bib` target `IEEEtran`
(conference mode) and compile cleanly -- `paper/main.pdf` in this repo
*is* that compiled output, checked in so you don't have to build it
just to read it.

To rebuild locally (verified working with a BasicTeX install):

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

If `pdflatex` reports a missing `IEEEtran.cls`, install it once with
`sudo tlmgr install ieeetran` (lowercase -- the CTAN package name).
If you hit `Font ... pcrr7t ... not loadable` (BasicTeX's minimal
install is missing the Courier metrics IEEEtran's typography wants),
that's already worked around in `main.tex` via
`\renewcommand{\ttdefault}{cmtt}` -- no action needed.

No local LaTeX install: drop `main.tex` + `references.bib` into a new
[Overleaf](https://overleaf.com) project (IEEEtran is built in).
Figures are referenced as `../figures/paper_fig*.png`, relative to
`paper/`.

## Status and what's not done yet

This is an active, unpublished research project -- not a finished
paper. `PAPER_DRAFT.md`'s own end-of-file checklist and `STATUS.md`'s
"Limitations" and "Next" sections are the accurate, current picture;
the short version:

- Bibliography: all 24 references actually cited in the current paper
  have been checked for bibliographic existence/status. Where a
  peer-reviewed version could be verified, the BibTeX now cites that
  version rather than an older arXiv preprint; remaining preprints are
  retained where no later published version was verified. The broader
  57-source research log in RELATED_WORK.md still contains candidate
  sources not used by the manuscript.
- Scale: three network topologies have now been tested, each to the
  same 16-seed standard (quadrupled from an original 4-seed pilot,
  2026-09-25, specifically to resolve this pattern statistically),
  each independently audited twice before that (feature-leakage, then
  oracle-residual/test-set-model-selection -- `STATUS.md` §6 has the
  full account). Corrected, 16-seed result: the pattern is
  network-specific, not task-specific. Both detection and localization
  show a small, mostly statistically confirmed positive gap at both
  standard IEEE systems -- IEEE 30-bus (`31_ieee30_scale_replication.py`
  + `32_ieee30_multi_seed_replication.py`) and IEEE 14-bus
  (`28_ieee14_scale_replication.py` + `30_ieee14_multi_seed_replication.py`)
  -- while 5-bus shows no such gap on either task. This is materially
  different from the 4-seed pilot immediately above, which made each
  IEEE network appear to favor a single, different task; that pattern
  turned out to sit within ordinary small-sample noise once the seed
  count was quadrupled. Four of the six network/task gaps now have a
  t-based 95% CI excluding zero and p<0.05 by t-test (IEEE-14 detection
  p=0.011, IEEE-14 localization p<0.001, IEEE-30 detection p<0.001,
  IEEE-30 localization p=0.007); a distribution-free sign test agrees
  for three of the four, not IEEE-14 detection (p=0.059); a Bonferroni
  correction for testing six gaps at once leaves the same three
  surviving (`STATUS.md` §6 has the full calculation). We describe
  IEEE-14 localization, IEEE-30 detection, and IEEE-30 localization as
  confirmed at a conservative, multiple-comparisons-aware threshold,
  IEEE-14 detection as directionally consistent but the
  weakest-supported of the four, and 5-bus as clearly null on both
  tasks.
- A committed, reproducible script for the out-of-fold redundancy
  diagnostic (`33_feature_redundancy_diagnostic.py`) now exists; the
  paper's earlier 0.960/0.959 R² numbers were never backed by
  committed code and have been replaced with 0.956/0.966 (a further,
  separate reproducibility bug -- the 5-bus figure was briefly
  0.950/0.966 in an intermediate pass that turned out to itself not be
  reproducible from a clean clone; see `STATUS.md` §6's latest update
  for the full account), the
  qualitative conclusion unchanged.
- Two smaller, known-and-documented gaps remain deliberately unresolved:
  cross-network measurement-noise/forecast-uncertainty is not
  standardized across the three networks (a real confound on the
  cross-network comparisons above, `STATUS.md` §6 item 7 -- so
  "network-specific" throughout this project is more precisely
  *setup-and-network-specific under the tested noise/prior regimes*;
  IEEE-14/IEEE-30 share an identical convention, so only comparisons
  that include 5-bus carry this confound), and the README's own
  clean-clone reproduction path had a real bug (fixed -- see the
  Repository map above) that was only caught while verifying this
  same audit.
- No target venue has formally accepted anything. SmartGridComm is a
  strong topical fit, but the current 10-page file is the full
  manuscript/preprint rather than a venue-compliant conference
  submission; a condensed version should be created against the exact
  page limit of the next chosen call.
