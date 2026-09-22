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
finding was audited a second time. With **both** fixed, the fair
comparison is a near-exact tie on **both tasks**: detection 0.938 vs.
0.935 (4-seed mean, gap +0.003±0.010, sign unstable) and localization
0.968 vs. 0.963 (gap +0.005±0.036, sign unstable, including two exact
ties across the 4 seeds). An earlier, once-corrected pass had reported
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

- **Detection**: null at 5-bus and IEEE-14 (gaps of +0.003±0.010 and
  +0.002±0.013, sign unstable at both) -- but a real, 4-seed-consistent
  *advantage* at IEEE-30 (+0.039±0.032, positive in all 4 seeds). This
  specific finding is new to this last audit pass; no earlier version
  of this analysis, corrected or not, found a detection advantage
  anywhere.
- **Localization**: null at 5-bus and IEEE-30 (+0.005±0.036 and
  +0.008±0.018, both sign-unstable) -- but a real advantage at IEEE-14
  (+0.018±0.021, positive in all 4 seeds), the one finding across this
  whole project that has survived every round of correction so far,
  though with roughly double the uncertainty an earlier, less-audited
  pass reported.

Put plainly: each of the two standard IEEE systems shows a genuine,
replicated topology advantage in exactly one task, and it's a
*different* task at each network; the small custom 5-bus microgrid
shows no advantage in either task, under either round of correction.
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

Two further things, checked and re-checked under the same audit and
essentially unchanged: the detector does **not** generalize to an
attack type it never trained on (0-1.3% recall when one of two attack
families is withheld, evaluated under a corrected, symmetric protocol
-- an earlier, asymmetric version of this same test reported 0-4%,
same conclusion), and the full decision pipeline (state estimation ->
feature computation -> classification) was profiled against a
one-cycle (20 ms at 50 Hz) protection-relevant computational target --
the original 43 ms pipeline turned out to be slow because of a fixable
software inefficiency, not the model or the estimator's math, and
fixing it brought the pipeline to 16.9/17.6/18.4 ms at the
median/p95/p99. That benchmark's own classifier and residual feature
were also found, in the same audit, to be untrained-on-dummy-data and
oracle-leaked respectively; re-measured with a classifier trained on
150 real warm-up scenarios and a deployable residual, the timing
conclusion is essentially unchanged (16.9 ms vs. the earlier 16.4 ms
median) -- computing a fixed-size result takes the same time whether
the numbers behind it are real or synthetic, so this was always
expected to hold, and it did.

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
                       fix found a real, 4-seed-consistent detection
                       advantage instead, new to that later pass
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
paper/main.tex          The same draft, formatted for IEEE
                       SmartGridComm (IEEEtran), with a verified
                       BibTeX bibliography (paper/references.bib).

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
python3 26_multi_seed_replication.py          # ~15-20 min, 4 full reruns
python3 27_held_out_attack_type_generalization.py
python3 24_realtime_latency_benchmark_FINAL.py
python3 28_ieee14_scale_replication.py --n-rep 500   # slower per-replication than the 5-bus scripts; this is the confirmed run (§7 of STATUS.md) -- --n-rep 200 was an earlier, superseded pass
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

- Bibliography: 41/57 sources verified in depth; a handful of
  IEEE Xplore/ACM-hosted sources couldn't be fetched at all (paywall)
  and need institutional access.
- Scale: three network topologies have now been tested, each to the
  same 4-seed standard, each independently audited twice (feature-
  leakage, then oracle-residual/test-set-model-selection --
  `STATUS.md` §6 has the full account). Corrected result: detection
  shows a reproducible topology advantage at exactly one network
  (IEEE 30-bus, `31_ieee30_scale_replication.py` +
  `32_ieee30_multi_seed_replication.py`); localization shows a
  reproducible advantage at exactly one, different, network (IEEE
  14-bus, `28_ieee14_scale_replication.py` +
  `30_ieee14_multi_seed_replication.py`); 5-bus shows no advantage in
  either task. This is a materially different finding from an earlier
  pass, which (before the oracle-residual and test-set-selection bugs
  were found) reported a uniform detection null and a localization
  advantage confined to IEEE-14 only, with IEEE-30 null on both tasks.
- A committed, reproducible script for the out-of-fold redundancy
  diagnostic (`33_feature_redundancy_diagnostic.py`) now exists; the
  paper's earlier 0.960/0.959 R² numbers were never backed by
  committed code and have been replaced with 0.950/0.966, the
  qualitative conclusion unchanged.
- Two smaller, known-and-documented gaps remain deliberately unresolved:
  cross-network measurement-noise/forecast-uncertainty is not
  standardized across the three networks (a real confound on the
  cross-network comparisons above, `STATUS.md` §6 item 7), and the
  README's own clean-clone reproduction path had a real bug (fixed --
  see the Repository map above) that was only caught while verifying
  this same audit.
- No target venue has formally accepted anything -- `paper/main.tex`
  is formatted for IEEE SmartGridComm as the best topical fit found,
  not a submission in progress.
