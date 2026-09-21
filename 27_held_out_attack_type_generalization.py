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


def fit_eval(df, feat_cols, train_mask, test_mask, test_label):
    X_train = df.loc[train_mask, feat_cols]
    y_train = df.loc[train_mask, "is_cyber"]
    X_test = df.loc[test_mask, feat_cols]
    y_test = df.loc[test_mask, "is_cyber"]

    clf = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "condition": test_label,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "recall_on_test": float(recall_score(y_test, pred)) if y_test.nunique() > 1 or y_test.iloc[0] == 1 else float((pred == 1).mean()),
        "balanced_accuracy_on_full_test": None,  # filled by caller where relevant
    }


def main():
    df = pd.read_csv(DATA / "phase2s_hard_graph_feature_matrix.csv")
    feat_cols = phase23.feature_columns(df)["topology_fusion"]

    rows = []

    for held_out in CYBER_TYPES:
        other_cyber = [c for c in CYBER_TYPES if c != held_out][0]

        # Held-out-type generalization: train WITHOUT this type at all.
        train_mask = (
            (df["dataset_split"] == "train")
            & (df["case"] != held_out)
        )
        # Test on ALL rows of the held-out type not used anywhere in training
        # (train split only excludes it, so validation+test rows of this
        # type are fair, unseen test material).
        test_mask = (df["case"] == held_out) & (df["dataset_split"] != "train")

        r = fit_eval(df, feat_cols, train_mask, test_mask,
                     f"held_out:{held_out}")
        rows.append(r)

        # Reference ceiling: same recipe, but this type WAS in training
        # (standard split), tested on its own normal test rows.
        train_mask_std = (df["dataset_split"] == "train")
        test_mask_std = (df["case"] == held_out) & (df["dataset_split"] == "test")
        r_ceiling = fit_eval(df, feat_cols, train_mask_std, test_mask_std,
                              f"seen_in_training:{held_out}")
        rows.append(r_ceiling)

        print(f"\n{held_out}:")
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
