from pathlib import Path
import subprocess
import sys

import pandas as pd

# ============================================================
# 5-BUS MULTI-SEED REPLICATION
# ============================================================
# Re-run data generation and evaluation for 16 independent seeds.
# Intermediate single-seed files are overwritten intentionally; the
# per-seed summary is preserved in results/phase2u_multiseed_*.csv.
# ============================================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

SEEDS = [20260812, 42, 777, 2024, 3, 11, 19, 37, 53, 71, 97, 131, 163, 197, 229, 251]
N_REP = 500


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def extract_headline(seed):
    det = pd.read_csv(RESULTS / "phase2s_hard_cyber_detection_metrics.csv")
    loc = pd.read_csv(RESULTS / "phase2s_hard_attack_localization_metrics.csv")

    topo_det = det[det["feature_set"] == "topology_fusion"]["balanced_accuracy"].max()
    rpp_det = det[det["feature_set"] == "residual_plus_prior"]["balanced_accuracy"].max()
    prior_det = det[det["feature_set"] == "prior_only"]["balanced_accuracy"].max()
    # Track every feature set used in the cross-network comparison.
    res_det = det[det["feature_set"] == "residual_only"]["balanced_accuracy"].max()
    topo_loc = loc[loc["feature_set"] == "topology_fusion"]["top1_accuracy"].max()
    rpp_loc = loc[loc["feature_set"] == "residual_plus_prior"]["top1_accuracy"].max()
    prior_loc = loc[loc["feature_set"] == "prior_only"]["top1_accuracy"].max()
    res_loc = loc[loc["feature_set"] == "residual_only"]["top1_accuracy"].max()

    return {
        "seed": seed,
        "topology_fusion_best_detection_balacc": topo_det,
        "residual_plus_prior_best_detection_balacc": rpp_det,
        "prior_only_best_detection_balacc": prior_det,
        "residual_only_best_detection_balacc": res_det,
        # Positive values indicate topology_fusion ahead of the matched
        # topology-free residual_plus_prior ablation.
        "topology_vs_residual_plus_prior_detection_gap": topo_det - rpp_det,
        "detection_gap": topo_det - prior_det,
        "topology_fusion_best_localization_top1": topo_loc,
        "residual_plus_prior_best_localization_top1": rpp_loc,
        "prior_only_best_localization_top1": prior_loc,
        "residual_only_best_localization_top1": res_loc,
        "topology_vs_residual_plus_prior_localization_gap": topo_loc - rpp_loc,
        "localization_gap": topo_loc - prior_loc,
    }


def main():
    rows = []
    for seed in SEEDS:
        print(f"\n=== SEED {seed} ===")
        run([sys.executable, "22_graph_ready_protected_prior_telemetry_HARD.py",
             "--n-rep", str(N_REP), "--seed", str(seed)])
        run([sys.executable, "23_topology_aware_cyber_physical_localization_HARD.py"])
        row = extract_headline(seed)
        rows.append(row)
        print(f"  topology vs residual+prior: "
              f"detection gap={row['topology_vs_residual_plus_prior_detection_gap']:+.4f} | "
              f"localization gap={row['topology_vs_residual_plus_prior_localization_gap']:+.4f}")

    df = pd.DataFrame(rows)
    print("\n=== ALL SEEDS ===")
    print(df.to_string(index=False))

    summary = df.drop(columns=["seed"]).agg(["mean", "std", "min", "max"])
    print("\n=== CROSS-SEED SUMMARY ===")
    print(summary.to_string())

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "phase2u_multiseed_replication.csv", index=False)
    summary.to_csv(RESULTS / "phase2u_multiseed_summary.csv")
    print("\nSaved:\n  results/phase2u_multiseed_replication.csv"
          "\n  results/phase2u_multiseed_summary.csv")

    # Restore the primary-seed snapshot expected by downstream scripts
    # after the multi-seed loop has overwritten the single-run files.
    primary_seed = SEEDS[0]
    print(f"\n=== Restoring primary-seed ({primary_seed}) snapshot for "
          f"downstream scripts (27, 33) ===")
    run([sys.executable, "22_graph_ready_protected_prior_telemetry_HARD.py",
         "--n-rep", str(N_REP), "--seed", str(primary_seed)])
    run([sys.executable, "23_topology_aware_cyber_physical_localization_HARD.py"])


if __name__ == "__main__":
    main()
