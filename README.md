# Cyber-Physical Grid Detection

Does giving a cyber-attack detector *topology* -- not just raw sensor
residuals, but how each meter's reading compares to its electrical
neighbors -- actually help it (a) tell a cyberattack apart from a
normal physical disturbance, (b) find where the attack is, and (c) do
both fast enough for a protective relay to act on?

Short answer: **no** -- once measured against a fair comparison, not
the one we first ran. See below for what that means, how we caught
our own mistake, and where to find the details.

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

That comparison turned out to be unfair. `prior_only` alone is
missing residual information `topology_fusion` has always had; the
actual control is `residual_plus_prior` -- residual and prior features
combined, with **no** topology at all. Building that ablation surfaced
a real bug: topology-relational columns are *named* after the
residual/innovation quantities they compare (e.g.
`neighbor_mean_residual`), so the original name-matching feature-set
split let about half of them leak into what were meant to be
topology-free baselines. Once fixed, the fair comparison is a
near-exact tie for detection (0.950 vs. 0.950) and, on the primary
seed, a win for the topology-free ablation at localization (0.993 vs.
0.980) -- across two independent stress conditions. The 4-seed mean
tells the more honest story: both gaps are small, 0.3 percentage
points either way (detection −0.003±0.010, sign unstable; localization
−0.003±0.007, consistently but only narrowly favoring the topology-free
side). **Combining residual and prior information explains the entire
advantage; explicit topology-awareness adds nothing measurable on this
testbed** -- but "adds nothing" here means a near-exact tie, not a
one-sided rout in either direction.

A separately-implemented, size-agnostic pipeline for a bigger,
standard, meshed network (IEEE 14-bus, n=500 replications, 300 test
scenarios) turned out to need the *same* fix -- an independent
rediscovery of the same category of mistake, in a cruder form (an
explicit list of neighbor/degree columns inside "residual_only" this
time, not a name-substring collision). Once corrected and confirmed
across 4 independent seeds, the two networks agree on **detection**
(`topology_fusion` never confidently ahead of the best non-relational
alternative -- an exact tie at 5-bus, a consistent 4-seed deficit at
IEEE-14) but **not on localization**: the topology-free ablation is
narrowly but consistently ahead at 5-bus (mean gap −0.003±0.007), while
`topology_fusion` keeps a comparably small but oppositely-signed,
4-seed-robust lead at IEEE-14 (mean gap +0.018, range +0.013 to +0.033,
sign never flipping). Neither gap is large -- what's real is the
*direction* reversing between the two networks, not a big effect at
either one -- the one place in this whole project where explicit
topology-relational features show a genuine, if modest, advantage. Two
independently-written pipelines needing the identical feature-definition
fix is itself informative -- it suggests that specific mistake
(comparing a topology-rich feature set against baselines that were
never actually topology-free) is an easy, general trap -- but, now
confirmed, it does not mean topology never helps anywhere: it means
the honest answer is task- and network-specific, not a clean yes or no.

A third network, IEEE 30-bus, looked in a single run like it
reproduced both halves of this pattern too (detection gap -0.006,
localization gap +0.020 -- both inside IEEE-14's own 4-seed range).
It doesn't hold up: re-run across the same 4 seeds used at IEEE-14,
neither gap keeps a consistent sign (detection: -0.007, +0.000,
+0.013, -0.003; localization: +0.020, +0.007, +0.013, -0.020) -- a
clean null, unlike IEEE-14's uniformly one-directional gaps. The
single run had simply landed on a seed that looked representative by
chance. Corrected picture: the localization reversal is real and
seed-confirmed at exactly one of the three tested scales (IEEE-14),
not a general property of standard/meshed networks -- see `STATUS.md`
§5.6 for the full per-seed numbers and how the earlier single-run
framing was caught and corrected.

Two further things, still true and still worth reporting plainly: the
detector does **not** generalize to an attack type it never trained on
(0-4% recall when one of two attack families is withheld), and the
full decision pipeline (state estimation -> feature computation ->
classification) was profiled against a one-cycle (20 ms at 50 Hz)
protection-relevant computational target -- the original 43 ms
pipeline turned out to be slow because of a fixable software
inefficiency, not the model or the estimator's math, and fixing it
brought the pipeline to 16.4/17.0/17.1 ms at the median/p95/p99. That
part of the finding is unaffected by the correction above.

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
                       pattern as 30 -- found that IEEE-30's single-run
                       match to IEEE-14 was a seed coincidence, not a
                       real effect

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
- Scale: three network topologies have now been tested. The IEEE
  14-bus study (`28_ieee14_scale_replication.py`) is confirmed at
  n=500 (300 test scenarios) across 4 independent seeds
  (`30_ieee14_multi_seed_replication.py`): detection agrees with
  5-bus (no measurable topology-relational advantage), but
  localization shows a real, seed-robust reversal (`topology_fusion`
  ahead by a small, sign-stable margin). IEEE 30-bus
  (`31_ieee30_scale_replication.py`, then 4-seed-replicated by
  `32_ieee30_multi_seed_replication.py`) looked like it matched
  IEEE-14 in its first single run, but the replication shows neither
  gap holds a consistent sign there -- a null result, not a second
  confirmation. The localization reversal is confirmed at exactly one
  of the three tested scales (IEEE-14); whether that makes IEEE-14
  the outlier or IEEE-30 just needs more replications to resolve is
  open (`STATUS.md` §5.6).
- No target venue has formally accepted anything -- `paper/main.tex`
  is formatted for IEEE SmartGridComm as the best topical fit found,
  not a submission in progress.
