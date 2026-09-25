from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score

warnings.filterwarnings("ignore")

# ============================================================
# HELD-OUT ATTACK-TYPE GENERALIZATION
# ============================================================
# Evaluate attack-family generalization using the two cyber case types
# in the final dataset. For each family, train with that family removed
# from the training split and measure recall on held-out test rows.
# Both topology_fusion and residual_plus_prior are evaluated under the
# same protocol.
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RANDOM_STATE = 20260812


def load_module(filename, module_name):
    path = ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase23 = load_module(
    "23_topology_aware_cyber_physical_localization_HARD.py", "phase23_for_27"
)

CYBER_TYPES = ["naive_single_sensor_corruption", "nonlinear_model_consistent_fdia"]


# Evaluate both the topology-aware representation and the matched
# topology-free residual_plus_prior ablation.
FEATURE_SETS_TO_TEST = ["topology_fusion", "residual_plus_prior"]


def fit_eval(df, feat_cols, train_mask, test_mask, test_label, feature_set):
    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, "is_cyber"]
    X_test = df.loc[test_mask, feat_cols]
    y_test = df.loc[test_mask, "is_cyber"]

    # FIX (caught in post-submission audit, STATUS.md Sec. 6 item 9,
    # continued): this used to be a bare, default-hyperparameter HGB,
    # different from the tuned model 23_topology_aware_cyber_physical_
    # localization_HARD.py actually reports as "the" detector
    # (max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
    # l2_regularization=1.0). The zero-day result should describe the
    # same detector's generalization, not a different, unconfigured one.
    clf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "feature_set": feature_set,
        "condition": test_label,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_features": len(feat_cols),
        "recall_on_test": float(recall_score(y_test, pred)) if y_test.nunique() > 1 or y_test.iloc[0] == 1 else float((pred == 1).mean()),
        "balanced_accuracy_on_full_test": None,  # filled by caller where relevant
    }


def main():
    df = pd.read_csv(DATA / "phase2s_hard_graph_feature_matrix.csv")
    all_feat_sets = phase23.feature_columns(df)

    rows = []

    for feature_set in FEATURE_SETS_TO_TEST:
        feat_cols = all_feat_sets[feature_set]

        for held_out in CYBER_TYPES:
            # Held-out-type generalization: train WITHOUT this type at all.
            train_mask = (
                (df["dataset_split"] == "train")
                & (df["case"] != held_out)
            )
            # Both held-out and seen-in-training conditions are evaluated
            # on the same test split for a matched recall comparison.
            test_mask = (df["case"] == held_out) & (df["dataset_split"] == "test")

            r = fit_eval(df, feat_cols, train_mask, test_mask,
                         f"held_out:{held_out}", feature_set)
            rows.append(r)

            # Reference ceiling: same recipe, but this type WAS in training
            # (standard split), tested on its own normal test rows.
            train_mask_std = (df["dataset_split"] == "train")
            test_mask_std = (df["case"] == held_out) & (df["dataset_split"] == "test")
            r_ceiling = fit_eval(df, feat_cols, train_mask_std, test_mask_std,
                                  f"seen_in_training:{held_out}", feature_set)
            rows.append(r_ceiling)

            print(f"\n[{feature_set}] {held_out}:")
            print(f"  held-out recall     : {r['recall_on_test']:.3f} (n_test={r['n_test']})")
            print(f"  seen-in-training recall: {r_ceiling['recall_on_test']:.3f} (n_test={r_ceiling['n_test']})")
            print(f"  generalization gap  : {r_ceiling['recall_on_test'] - r['recall_on_test']:+.3f}")

    out = pd.DataFrame(rows)
    print("\n=== FULL TABLE ===")
    print(out.to_string(index=False))

    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "phase2u_held_out_attack_type.csv", index=False)
    print("\nSaved:\n  results/phase2u_held_out_attack_type.csv")


if __name__ == "__main__":
    main()
