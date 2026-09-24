# Cyber-Physical Grid Detection

Does giving a cyber-attack detector *topology* -- not just raw sensor
residuals, but how each meter's reading compares to its electrical
neighbors -- actually help it (a) tell a cyberattack apart from a
normal physical disturbance, (b) find where the attack is, and (c) do
both fast enough for a protective relay to act on?

Short answer: **it depends which task and which network** -- and only
after two full rounds of auditing our own pipeline and catching our
own mistakes, twice, did that become the honest answer rather than a
cleaner-sounding wrong one. See below for what that means and where to
find the details.

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
unstable, including one exact tie across the 4 seeds). An earlier, once-corrected pass had reported
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

**A caveat on both of those, stated plainly**: 4 same-signed seeds is
consistent-sign evidence, not a statistically confirmed effect. A
standard t-based 95% CI crosses zero for both gaps (IEEE-30 detection:
+0.039±0.047, i.e. [-0.008, 0.086]; IEEE-14 localization: +0.023±0.033,
i.e. [-0.010, 0.056]). A distribution-free sign test doesn't reach
p<0.05 for either, but not by the same margin: one-sided p=0.5⁴=0.0625
for IEEE-30 (four of four seeds positive), and p=0.5³=0.125 for IEEE-14
once its one exact tie is excluded per sign-test convention (three of
three non-zero seeds positive, not four of four -- a standard sign test
drops ties rather than counting them as a positive). Both findings are
described throughout as *consistent in sign across all four tested
seeds*, not as confirmed nonzero effects. Note the sign test's own bar
is low if the pattern holds -- a fifth same-direction seed at IEEE-30
would already cross one-sided p<0.05 (0.5⁵=0.03125) -- so resolving the
gap's actual *magnitude* with a tight, stable confidence interval is the
harder goal, needing roughly 10-20 seeds or a paired/bootstrap
difference test, neither run here.

Put plainly: each of the two standard IEEE systems shows a
sign-consistent topology gap in exactly one task, and it's a
*different* task at each network; the small custom 5-bus microgrid
shows no such gap in either task, under either round of correction.
Three separately-written pipelines needing the same category of
feature-leakage fix, and then a second, more serious category of
mistake (oracle information reaching a feature meant to be deployable)
surviving a first full audit only to be caught in a second, more
adversarial one, is itself the finding we'd want a reader to take away
alongside the numbers: neither kind of bug was visible from results
alone -- both were found only by reading feature-computation code line
by line against "could a real, deployed detector actually compute
this." Full per-seed numbers, and the complete before/after account of
both audit rounds, are in `STATUS.md` §6.

Two further things remain central. First, the detector does **not**
generalize to an attack type it never trained on (0-1.3% recall when
one of two attack families is withheld, under the corrected symmetric
protocol). Second, the full decision pipeline (state estimation ->
feature computation -> classification) was profiled against a
one-cycle (20 ms at 50 Hz) computational target. A fresh documented
N=300 run on Apple arm64 / macOS 15.6.1 (Python 3.12.4,
pandapower 2.14.10, NumPy 1.26.4, pandas 2.2.2, scikit-learn 1.4.2)
gave 17.015/18.211/19.190 ms at the median/p95/p99 and 92.657 ms max,
with 0 convergence failures. Thus the median, p95, and p99 remain
inside the 20 ms target while rare solver-tail events can exceed it.
The stage medians make the bottleneck clear: WLS state estimation
16.664 ms versus 0.057 ms residual/innovation calculation, 0.088 ms
feature computation, 0.008 ms array assembly, and 0.200 ms inference.

**Full story, with every number and why it's trustworthy: [`STATUS.md`](STATUS.md).**

## Repository map

```
exploration/            Phases 2A-2P (01..20_*.py): microgrid model,
                       inverter models, WLS state estimation, early
                       cyber-vs-physical benchmarks -- built before
                       this session, kept as the record of how the
                       current design was arrived at. Nothing in the
                       paper depends on this folder except one shared
                       module (exploration/01_microgrid_topology.py,
                       loaded by 21_*.py below) -- it is not part of
                       the paper pipeline otherwise.

21_*_HARD.py           Attack generator, magnitude reduced toward the
                       sensor noise floor (this session's stress test)
22_*_HARD.py/HARDER.py  Dataset generation on top of 21_*_HARD
23_*_FIXED/HARD/HARDER.py  Detection + localization evaluation
24_realtime_latency_*.py   Phase 2T: latency measurement and fixes
25_wls_overhead_diagnostic.py   Root-cause split of the latency gap
26_multi_seed_replication.py    4-seed statistical validation
27_held_out_attack_type_*.py    Zero-day generalization test
28_ieee14_scale_replication.py  The IEEE 14-bus scale study
                       (confirmed at n=500, matching the 5-bus study's
                       own 300-scenario test-set size)
29_paper_figures.py    Generates figures/paper_fig*.png from results/
30_ieee14_multi_seed_replication.py  4-seed replication of 28, same
                       pattern as 26 -- confirms the IEEE-14
                       localization result isn't single-run noise
31_ieee30_scale_replication.py  A third network (IEEE 30-bus), direct
                       copy of 28 with the network swapped -- tests
                       whether IEEE-14's pattern generalizes
32_ieee30_multi_seed_replication.py  4-seed replication of 31, same
                       pattern as 30 -- re-run twice: first found
                       IEEE-30's single-run match to IEEE-14 was a seed
                       coincidence (localization null); re-run again
                       after 31's oracle-residual/model-selection audit
                       fix found a detection gap positive in all 4
                       tested seeds instead, new to that later pass
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
original" -- the original is kept alongside it on purpose, as the
record of what changed and why. `STATUS.md` narrates the actual
sequence and reasoning; the file list alone won't make sense without
it. The `exploration/` vs. root-level split follows the same logic at
folder granularity: `exploration/` is "how we got here," root is "what
the paper is."

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
python3 26_multi_seed_replication.py          # ~15-20 min, 4 full reruns, then auto-restores the primary-seed (20260812) snapshot -- 27 and 33 below both read data/phase2s_hard_graph_feature_matrix.csv directly and need that exact snapshot, not whichever seed this loop last ran
python3 27_held_out_attack_type_generalization.py
python3 24_realtime_latency_benchmark_FINAL.py   # also records results/phase2t_latency_environment.json (machine + software versions) for reproducible latency reporting
python3 28_ieee14_scale_replication.py --n-rep 500   # slower per-replication than the 5-bus scripts; this is the confirmed run (§7 of STATUS.md) -- --n-rep 200 was an earlier, superseded pass
python3 33_feature_redundancy_diagnostic.py   # must run HERE, not after 30 below -- 30's own seed loop overwrites data/phase2v_ieee14_node_feature_matrix.csv with its last seed (2024), and this script's IEEE-14 R^2 needs the primary-seed (20260921) run 28 just produced, not a multi-seed leftover (see this script's own --help). Its 5-bus side is separately safe because 26 above already restored that primary-seed snapshot.
python3 30_ieee14_multi_seed_replication.py   # ~30-40 min, 4 full reruns of 28
python3 31_ieee30_scale_replication.py --n-rep 500
python3 32_ieee30_multi_seed_replication.py   # ~30-40 min, 4 full reruns of 31
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
  same 4-seed standard, each independently audited twice (feature-
  leakage, then oracle-residual/test-set-model-selection --
  `STATUS.md` §6 has the full account). Corrected result: detection
  shows a gap consistently positive across all four tested seeds at
  exactly one network (IEEE 30-bus, `31_ieee30_scale_replication.py` +
  `32_ieee30_multi_seed_replication.py`); localization shows a gap
  never negative across the same four seeds at exactly one, different,
  network (IEEE 14-bus, `28_ieee14_scale_replication.py` +
  `30_ieee14_multi_seed_replication.py`); 5-bus shows no such gap in
  either task. This is a materially different finding from an earlier
  pass, which (before the oracle-residual and test-set-selection bugs
  were found) reported a uniform detection null and a localization
  advantage confined to IEEE-14 only, with IEEE-30 null on both tasks.
  Neither surviving gap is yet a statistically confirmed effect at
  n=4 seeds -- a t-based 95% CI crosses zero for both, and a
  distribution-free sign test gives p=0.0625 for IEEE-30 (4/4 seeds
  positive) and p=0.125 for IEEE-14 (its one exact tie excluded per
  sign-test convention, leaving 3/3 non-zero seeds positive, not 4/4),
  short of p<0.05 either way (`STATUS.md` §6 has the full calculation).
  We describe both as consistent in sign across every seed tested, not
  as confirmed nonzero.
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
