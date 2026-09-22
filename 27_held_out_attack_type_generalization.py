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
# PHASE 2U-B -- HELD-OUT ATTACK-TYPE GENERALIZATION
# ============================================================
#
# STATUS.md's "next steps" originally said "re-run Phase 14 (zero-day)
# against the HARD/HARDER data." That turned out to be wrong to
# promise: Phase 14/17 read from a completely different upstream data
# lineage (phase2g_v2_scenario_metadata.csv / phase2i_streaming_*),
# built for a richer taxonomy of event types, and are not compatible
# with the Phase 2Q/2R/2S graph-feature files without a substantial
# bridging effort. Rather than force that fit, this script asks the
# same *kind* of question -- does the detector generalize to an
# attack TYPE it never trained on, not just a new sample of a type it
# already knows -- directly on the data already validated here.
#
# This project only has two cyber case types (naive_single_sensor_
# corruption, nonlinear_model_consistent_fdia/stealth), so this is a
# small, 2-condition test, not a claim of the same scope as Phase 14's
# 5-cyber-type taxonomy. Framed honestly as that.
#
# Method: for each cyber type, train topology_fusion/HistGradientBoosting
# with that type COMPLETELY REMOVED from train+validation (clean +
# physical + the OTHER cyber type only), then test recall specifically
# on the held-out type's test-split rows. Compare against the same
# model's recall when it WAS allowed to train on that type (the
# standard Phase 2S-hard result) as a reference ceiling.
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


# Originally topology_fusion only ("this test uses topology_fusion
# specifically, unaffected by Sec. IV's correction -- whether
# residual_plus_prior shows the same collapse was not independently
# re-tested"). Added residual_plus_prior here to close that flagged
# gap: Sec. IV shows the two feature sets are behaviorally
# near-identical in-distribution, so the mechanistic expectation is
# that the zero-day collapse -- which is about attack TYPES never
# seen at all, not about which non-relational features are used --
# should not depend on whether topology-relational columns are present.
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
            # FIX (caught in post-submission audit, STATUS.md Sec. 6
            # item 9): this used to be dataset_split != "train"
            # (validation+test combined), while the seen-in-training
            # reference below evaluates on dataset_split == "test"
            # only -- an apples-to-oranges comparison (different
            # evaluation-set composition, not just different training
            # data). Both conditions now evaluate on the same "test"
            # split; validation+test rows of the held-out type were
            # fair, unseen material either way, but comparing the two
            # recall numbers requires the denominator to match too.
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
