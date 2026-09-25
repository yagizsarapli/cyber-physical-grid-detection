# Cyber-Physical Grid Detection

Reproducible experiments for the manuscript **“Evaluating the Value of Topology-Aware Features for Cyber-Physical Grid Monitoring: Detection, Localization, and Real-Time Feasibility.”**

The study evaluates whether explicit topology-relational features improve false-data-injection attack (FDIA) detection and localization beyond a matched topology-free baseline, and separately measures end-to-end real-time feasibility.

## Main results

- **Topology ablation, 16 seeds:** on the custom 5-bus microgrid, `topology_fusion` and `residual_plus_prior` are effectively tied for both detection (mean gap `-0.000 ± 0.009`) and localization (`+0.003 ± 0.017`). Small positive gaps appear at both standard IEEE systems:
  - IEEE 14-bus: `+0.009 ± 0.012` detection, `+0.018 ± 0.017` localization.
  - IEEE 30-bus: `+0.028 ± 0.019` detection, `+0.013 ± 0.016` localization.
  Three IEEE effects remain significant under a Bonferroni-adjusted threshold; IEEE-14 detection is the borderline case.
- **Latency:** pipeline optimization reduces median end-to-end latency from **42.98 ms** to **16.2–17.0 ms**. Across two independent 300-trial reruns, p99 is **17.0–19.2 ms**, while rare maxima remain above the 20 ms reference (**89.1–92.7 ms**).
- **Unseen attack types:** when an entire attack family is withheld from training, recall collapses to **0–1.3%**, versus **91–100%** when that family is represented in training.

The cross-network result is intentionally interpreted as **setup-dependent**: the 5-bus and IEEE systems do not use identical measurement-noise and prior-uncertainty settings.

## Manuscript

- Compiled paper: [`paper/main.pdf`](paper/main.pdf)
- LaTeX source: [`paper/main.tex`](paper/main.tex)
- Bibliography: [`paper/references.bib`](paper/references.bib)
- Paper figures: [`figures/`](figures/)

## Repository structure

| Path | Purpose |
|---|---|
| `21_*_HARD.py` – `23_*_HARD*.py` | 5-bus attack generation, feature construction, detection and localization |
| `24_realtime_latency_benchmark_FINAL.py` | Final end-to-end latency benchmark |
| `25_wls_overhead_diagnostic.py` | State-estimation / measurement-update profiling |
| `26_multi_seed_replication.py` | 16-seed 5-bus replication |
| `27_held_out_attack_type_generalization.py` | Withheld-attack-family generalization test |
| `28_ieee14_scale_replication.py`, `30_ieee14_multi_seed_replication.py` | IEEE 14-bus evaluation |
| `31_ieee30_scale_replication.py`, `32_ieee30_multi_seed_replication.py` | IEEE 30-bus evaluation |
| `33_feature_redundancy_diagnostic.py` | Out-of-fold relational-feature redundancy diagnostic |
| `29_paper_figures.py` | Generates the manuscript figures |
| `exploration/08_multirate_operating_dataset.py` | Generates the operating dataset required by the main experiments |
| `archive/`, `exploration/archive/` | Superseded exploratory scripts; not used for reported manuscript results |
| `RELATED_WORK.md` | Literature notes |
| `STATUS.md` | Detailed research/audit log |

Generated `data/` and `results/` directories are gitignored.

## Reproducing the reported experiments

The pinned environment is in `requirements.txt` (verified with Python 3.12.4).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run the experiments in this order on a clean clone:

```bash
python3 exploration/08_multirate_operating_dataset.py

python3 22_graph_ready_protected_prior_telemetry_HARD.py --n-rep 500
python3 23_topology_aware_cyber_physical_localization_HARD.py
python3 26_multi_seed_replication.py

python3 27_held_out_attack_type_generalization.py
python3 24_realtime_latency_benchmark_FINAL.py

python3 28_ieee14_scale_replication.py --n-rep 500
python3 33_feature_redundancy_diagnostic.py
python3 30_ieee14_multi_seed_replication.py

python3 31_ieee30_scale_replication.py --n-rep 500
python3 32_ieee30_multi_seed_replication.py

python3 29_paper_figures.py
```

**Ordering note:** run `33_feature_redundancy_diagnostic.py` immediately after `28_ieee14_scale_replication.py`. The IEEE-14 multi-seed script overwrites the primary-seed feature-matrix snapshot used by that diagnostic.

The multi-seed studies are computationally heavier than the single-run scripts. Runtime depends strongly on CPU and BLAS implementation.

## Rebuilding the paper

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The repository also contains a packaging script for the arXiv source bundle:

```bash
bash make_arxiv_submission.sh
```

It produces:

- `arxiv_preview.pdf`
- `arxiv_submission.zip`

The ZIP contains only the LaTeX source, bibliography, and four manuscript figures.

## Reproducibility notes

- Train/validation/test splitting is performed at the **replication level**, so matched scenarios from one operating point cannot leak across partitions.
- Reported residual features are deployable WLS measurement residuals; simulator ground truth is not used as a detector input.
- Non-relational baselines exclude relational columns and static node-degree/device-role metadata.
- Model selection uses validation data only; test data are reserved for final reporting.
- The latency result is hardware- and software-environment-specific. The benchmark script records the platform and package versions in `results/phase2t_latency_environment.json`.

## Scope

The manuscript evaluates three synthetic/test networks, two attack families, and one documented latency environment. The cross-network comparison is not a controlled topology-only experiment because the 5-bus and IEEE cases use different noise/prior settings. These limitations are stated explicitly in the paper.

## Contact

Yagiz Sarapli  
Department of Electrical Engineering and Information Technology  
TU Dortmund University  
Dortmund, Germany
