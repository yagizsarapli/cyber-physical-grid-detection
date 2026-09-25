from pathlib import Path
import subprocess
import sys

import pandas as pd

# ============================================================
# IEEE-14 MULTI-SEED REPLICATION
# ============================================================
# Re-run the scale study for 16 independent seeds. Intermediate
# single-seed files are overwritten intentionally; per-seed summaries
# are preserved in results/phase2w_* files.
# ============================================================
#
# The IEEE-14 result (28_ieee14_scale_replication.py) has only ever
# been run at one seed. The 5-bus study's own headline result was not
# trusted until confirmed across 4 independent seeds
# (26_multi_seed_replication.py) -- this applies the identical
# discipline here, specifically because the IEEE-14 localization
# result (topology_fusion 0.560 vs. the best non-relational
# alternative at 0.547, a 0.013 margin) is exactly the kind of small,
# single-run gap that this project's own methodology (Sec. 2 of
# STATUS.md) has already taught it not to trust without replication.
#
# Re-runs 28_ieee14_scale_replication.py (data generation + evaluation
# in one script, unlike the 5-bus 22+23 split) for several independent
# seeds, sequentially (each seed overwrites the same
# phase2v_ieee14_*.csv files -- intentional, only the extracted summary
# numbers per seed are kept), and reports mean/std/range.
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SEEDS = [20260921, 42, 777, 2024, 3, 11, 19, 37, 53, 71, 97, 131, 163, 197, 229, 251]
N_REP = 500


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def extract_headline(seed):
    det = pd.read_csv(RESULTS / "phase2v_ieee14_detection_metrics.csv")
    loc = pd.read_csv(RESULTS / "phase2v_ieee14_localization_metrics.csv")

    topo_det = det[det["feature_set"] == "topology_fusion"]["balanced_accuracy"].max()
    rpp_det = det[det["feature_set"] == "residual_plus_prior"]["balanced_accuracy"].max()
    res_det = det[det["feature_set"] == "residual_only"]["balanced_accuracy"].max()
    prior_det = det[det["feature_set"] == "prior_only"]["balanced_accuracy"].max()

    topo_loc = float(loc.loc[loc["feature_set"] == "topology_fusion", "top1_accuracy"].iloc[0])
    rpp_loc = float(loc.loc[loc["feature_set"] == "residual_plus_prior", "top1_accuracy"].iloc[0])
    res_loc = float(loc.loc[loc["feature_set"] == "residual_only", "top1_accuracy"].iloc[0])
    prior_loc = float(loc.loc[loc["feature_set"] == "prior_only", "top1_accuracy"].iloc[0])

    return {
        "seed": seed,
        "topology_fusion_det": topo_det,
        "residual_plus_prior_det": rpp_det,
        "residual_only_det": res_det,
        "prior_only_det": prior_det,
        "best_nonrelational_det": max(rpp_det, res_det, prior_det),
        "topo_vs_best_nonrelational_det_gap": topo_det - max(rpp_det, res_det, prior_det),
        "topo_vs_residual_plus_prior_det_gap": topo_det - rpp_det,
        "topology_fusion_loc": topo_loc,
        "residual_plus_prior_loc": rpp_loc,
        "residual_only_loc": res_loc,
        "prior_only_loc": prior_loc,
        "best_nonrelational_loc": max(rpp_loc, res_loc, prior_loc),
        "topo_vs_best_nonrelational_loc_gap": topo_loc - max(rpp_loc, res_loc, prior_loc),
        "topo_vs_residual_plus_prior_loc_gap": topo_loc - rpp_loc,
    }


def main():
    rows = []
    for seed in SEEDS:
        print(f"\n=== SEED {seed} ===")
        run([sys.executable, "28_ieee14_scale_replication.py",
             "--n-rep", str(N_REP), "--seed", str(seed)])
        row = extract_headline(seed)
        rows.append(row)
        print(f"  topology vs residual+prior: "
              f"detection gap={row['topo_vs_residual_plus_prior_det_gap']:+.4f} | "
              f"localization gap={row['topo_vs_residual_plus_prior_loc_gap']:+.4f}")

    df = pd.DataFrame(rows)
    print("\n=== ALL SEEDS ===")
    print(df.to_string(index=False))

    summary = df.drop(columns=["seed"]).agg(["mean", "std", "min", "max"])
    print("\n=== CROSS-SEED SUMMARY ===")
    print(summary.to_string())

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "phase2w_ieee14_multiseed_replication.csv", index=False)
    summary.to_csv(RESULTS / "phase2w_ieee14_multiseed_summary.csv")
    print("\nSaved:\n  results/phase2w_ieee14_multiseed_replication.csv"
          "\n  results/phase2w_ieee14_multiseed_summary.csv")


if __name__ == "__main__":
    main()
