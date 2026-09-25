from pathlib import Path
import argparse
import importlib.util

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# ============================================================
# OUT-OF-FOLD LINEAR-REDUNDANCY DIAGNOSTIC
# ============================================================
#
# Reproducibility implementation for the relational-feature redundancy
# diagnostic reported in the paper.
#
# Method: for each explicit
# topology-relational column (neighbor-/incident-edge-comparison
# features), fit a 5-fold cross-validated Ridge regression predicting
# that single column from the full non-relational (residual_plus_prior)
# feature vector, and record the out-of-fold R^2. A high R^2 means a
# flexible classifier could reconstruct that relational feature from
# non-relational information alone, without ever being given it
# explicitly. Feature lists are obtained from each network's own
# feature-definition function to keep the diagnostic aligned with the
# evaluated pipeline:
#   5-bus:    feature_columns() in 23_topology_aware_cyber_physical_localization_HARD.py
#             (graph-level, one row per scenario)
#   IEEE-14:  node-level relational columns in
#             28_ieee14_scale_replication.py's per_node_rows(), pivoted
#             from long (one row per node per scenario) to wide (one
#             row per scenario, one column per node's local res/innov)
#             so "predicted from all 14 nodes' local residual/
#             innovation values" is a literal, not approximate, match
#             to the paper's own description.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_FOLDS = 5
RIDGE_ALPHA = 1.0


def load_module(filename, module_name):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def out_of_fold_r2(X, y, groups):
    # Group folds by replication so matched scenarios from the same
    # operating point never appear on opposite sides of an OOF split.
    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_pred = np.zeros(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = Ridge(alpha=RIDGE_ALPHA)
        model.fit(X[train_idx], y[train_idx])
        oof_pred[test_idx] = model.predict(X[test_idx])
    return r2_score(y, oof_pred)


def run_5bus():
    print("\n=== 5-BUS (HARD, primary seed) ===")
    phase23 = load_module(
        "23_topology_aware_cyber_physical_localization_HARD.py", "phase23_for_33"
    )
    df = pd.read_csv(DATA / "phase2s_hard_graph_feature_matrix.csv")
    feats = phase23.feature_columns(df)

    residual_plus_prior = feats["residual_plus_prior"]
    topology_fusion = feats["topology_fusion"]
    relational_only = sorted(set(topology_fusion) - set(residual_plus_prior))
    print(f"non-relational (residual_plus_prior): {len(residual_plus_prior)} columns")
    print(f"relational-only columns to predict   : {len(relational_only)} columns")

    X = df[residual_plus_prior].fillna(0.0).to_numpy(dtype=float)
    groups = df["replication"].to_numpy()

    rows = []
    for col in relational_only:
        y = df[col].fillna(0.0).to_numpy(dtype=float)
        if np.std(y) < 1e-12:
            continue
        r2 = out_of_fold_r2(X, y, groups)
        rows.append({"network": "5-bus", "relational_column": col, "r2": r2})

    out = pd.DataFrame(rows)
    print(f"columns fit: {len(out)} (skipped {len(relational_only) - len(out)} constant columns)")
    print(f"mean R^2   : {out['r2'].mean():.3f}")
    print(f"median R^2 : {out['r2'].median():.3f}")
    print(f"min R^2    : {out['r2'].min():.3f}")
    print(f"% above 0.8: {100.0 * (out['r2'] > 0.8).mean():.1f}%")
    return out


def run_ieee14(node_matrix_path, seed_label):
    print(f"\n=== IEEE-14 ({seed_label}) ===")
    node = pd.read_csv(node_matrix_path)

    local_cols = ["node_res", "node_innov"]
    relational_cols = [
        "node_res_nb_mean", "node_res_nb_max", "node_res_local_minus_nb",
        "node_innov_nb_mean", "node_innov_nb_max", "node_innov_local_minus_nb",
    ]

    wide_local = node.pivot(index="scenario_id", columns="node", values=local_cols)
    wide_local.columns = [f"{c}_{n}" for c, n in wide_local.columns]
    wide_local = wide_local.fillna(0.0)
    predictor_cols = list(wide_local.columns)
    print(f"non-relational predictors: {len(predictor_cols)} columns "
          f"(2 per node x {node['node'].nunique()} nodes)")

    X_by_scenario = wide_local.to_numpy(dtype=float)
    scenario_order = wide_local.index
    # scenario_id is f"{rep}_{case_name}" (28/31_*.py's one_replication());
    # rep is purely numeric, so splitting on the first "_" isolates it
    # regardless of how many underscores case_name itself has.
    groups = np.array([s.split("_")[0] for s in scenario_order])

    rows = []
    for target_node in sorted(node["node"].unique()):
        node_rows = node[node["node"] == target_node].set_index("scenario_id")
        node_rows = node_rows.reindex(scenario_order)
        for col in relational_cols:
            y = node_rows[col].fillna(0.0).to_numpy(dtype=float)
            if np.std(y) < 1e-12:
                continue
            r2 = out_of_fold_r2(X_by_scenario, y, groups)
            rows.append({
                "network": "IEEE-14", "node": target_node,
                "relational_column": col, "r2": r2,
            })

    out = pd.DataFrame(rows)
    print(f"node x column pairs fit: {len(out)}")
    print(f"mean R^2   : {out['r2'].mean():.3f}")
    print(f"median R^2 : {out['r2'].median():.3f}")
    print(f"min R^2    : {out['r2'].min():.3f}")
    print(f"% above 0.8: {100.0 * (out['r2'] > 0.8).mean():.1f}%")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ieee14", action="store_true",
                         help="5-bus only (IEEE-14 needs a fresh primary-seed "
                              "run of 28_ieee14_scale_replication.py first -- "
                              "the multi-seed replication script overwrites "
                              "phase2v_ieee14_node_feature_matrix.csv with "
                              "its last seed's data).")
    args = parser.parse_args()

    bus5 = run_5bus()
    bus5.to_csv(RESULTS / "phase2z_5bus_redundancy_r2.csv", index=False)

    if not args.skip_ieee14:
        ieee14 = run_ieee14(
            DATA / "phase2v_ieee14_node_feature_matrix.csv",
            "current data/phase2v_ieee14_node_feature_matrix.csv -- "
            "confirm this is the primary-seed run, not a multi-seed leftover",
        )
        ieee14.to_csv(RESULTS / "phase2z_ieee14_redundancy_r2.csv", index=False)

    print("\nSaved:\n  results/phase2z_5bus_redundancy_r2.csv" +
          ("" if args.skip_ieee14 else "\n  results/phase2z_ieee14_redundancy_r2.csv"))


if __name__ == "__main__":
    main()
