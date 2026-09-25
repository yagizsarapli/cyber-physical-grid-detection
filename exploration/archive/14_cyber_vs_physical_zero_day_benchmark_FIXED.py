from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

STREAM_PATH = DATA / "phase2i_streaming_feature_dataset.csv"
META_PATH = DATA / "phase2g_v2_scenario_metadata.csv"

RANDOM_STATE = 20260812

# A repeated online detector needs a much lower per-window false-positive
# operating point than a one-shot classifier.
TARGET_BENIGN_WINDOW_FPR = 0.01

CYBER_FAMILIES = {"cyber", "hybrid"}

CYBER_EVENT_TYPES = [
    "cyber_fake_voltage_measurement",
    "cyber_phase_drift",
    "cyber_gfl_current_limit",
    "cyber_gfm_setpoint_manipulation",
    "hybrid_fault_masking",
]

PHYSICAL_EVENT_TYPES = [
    "physical_cloud_transient",
    "physical_load_spike",
    "physical_voltage_sag",
]

PERSISTENCE_POLICIES = [
    ("1_of_1", 1, 1),
    ("2_of_3", 2, 3),
    ("3_of_5", 3, 5),
]

INVERTER_SIGNAL_PREFIXES = [
    "v_gfl_measured_pu__",
    "theta_gfl_measured_deg__",
    "gfl_pll_freq_dev_hz__",
    "gfl_id_pu__",
    "gfl_iq_pu__",
    "gfl_current_pu__",
    "gfm_freq_dev_hz__",
    "gfm_p_pu__",
    "gfm_q_pu__",
    "gfm_current_pu__",
    "gfl_limiter__",
    "gfm_limiter__",
]

PHYSICS_PROXY_EXTRA_PREFIXES = [
    "min_bus_v_pu__",
    "max_line_loading_percent__",
    "sensor_v_residual_pu__",
    "sensor_angle_residual_deg__",
]


def model_zoo():
    return {
        "LogisticRegression": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        max_features="sqrt",
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def feature_columns(df, feature_set):
    prefixes = list(
        INVERTER_SIGNAL_PREFIXES
    )

    if feature_set == "physics_proxy":
        prefixes += (
            PHYSICS_PROXY_EXTRA_PREFIXES
        )
    elif feature_set != "inverter_only":
        raise ValueError(feature_set)

    prefixes = tuple(prefixes)

    cols = [
        c
        for c in df.columns
        if c.startswith(prefixes)
    ]

    # Explicit leakage protection.
    forbidden = (
        "scenario_id",
        "window_mode",
        "event_type",
        "event_family",
        "phase",
        "split",
        "context_",
        "event_start",
        "event_end",
        "duration",
        "severity",
        "imax_pu",
        "pref_pu",
    )

    return sorted(
        [
            c
            for c in cols
            if not any(
                token in c
                for token in forbidden
            )
        ]
    )


def add_cyber_target(df):
    d = df.copy()

    # Keep causal normal/pre-event/event windows.
    d = d[
        d["phase"].isin(
            [
                "normal",
                "pre_event",
                "event",
            ]
        )
    ].copy()

    d["target_cyber"] = (
        (d["phase"] == "event")
        & (
            d["event_family"].isin(
                CYBER_FAMILIES
            )
        )
        & (
            d["window_mode"] == "event"
        )
    ).astype(int)

    d["benign_kind"] = "normal"

    physical_mask = (
        (d["phase"] == "event")
        & (
            d["event_family"]
            == "physical"
        )
        & (
            d["window_mode"]
            == "event"
        )
    )

    d.loc[
        physical_mask,
        "benign_kind",
    ] = "physical_disturbance"

    pre_mask = (
        d["phase"] == "pre_event"
    )

    d.loc[
        pre_mask,
        "benign_kind",
    ] = "pre_event"

    return d


def positive_scores(model, X):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    return proba[:, classes.index(1)]


def threshold_from_benign_validation(
    y_val,
    scores,
):
    y_val = np.asarray(y_val)
    scores = np.asarray(scores)

    benign = scores[
        y_val == 0
    ]

    if len(benign) == 0:
        raise RuntimeError(
            "No benign validation windows."
        )

    return float(
        np.quantile(
            benign,
            1.0
            - TARGET_BENIGN_WINDOW_FPR,
            method="higher",
        )
    )


def apply_persistence(
    df,
    score_col,
    threshold,
    k,
    n,
):
    """
    Apply k-of-n persistence independently to each trajectory.

    Important:
    The input DataFrame may retain non-contiguous/global pandas indices
    after train/validation/test filtering. Therefore we must NOT use those
    labels as positional indices into a NumPy array.

    We write alarms back with .loc using the original DataFrame labels.
    """
    out = df.copy()

    out["raw_flag"] = (
        out[score_col] >= threshold
    ).astype(int)

    # Index-safe initialization.
    out["alarm"] = 0

    for (
        scenario_id,
        window_mode,
    ), idx in out.groupby(
        [
            "scenario_id",
            "window_mode",
        ],
        sort=False,
    ).groups.items():
        block = out.loc[
            list(idx)
        ].sort_values(
            "window_end_s"
        )

        flags = block[
            "raw_flag"
        ].to_numpy(
            dtype=int
        )

        persistent_flags = np.zeros(
            len(flags),
            dtype=int,
        )

        for i in range(
            len(flags)
        ):
            lo = max(
                0,
                i - n + 1
            )

            if (
                flags[
                    lo:i+1
                ].sum()
                >= k
            ):
                persistent_flags[i] = 1

        # Safe label-aligned assignment.
        out.loc[
            block.index,
            "alarm",
        ] = persistent_flags

    out["alarm"] = out[
        "alarm"
    ].astype(int)

    return out

def trajectory_metrics(
    scored,
    meta,
):
    meta_idx = meta.set_index(
        "scenario_id"
    )

    cyber_hits = []
    cyber_delays = []
    normal_false = []
    physical_false = []
    pre_event_false = []

    per_attack = []

    # Normal counterfactual trajectories.
    for (
        sid,
        mode,
    ), g in scored[
        scored[
            "window_mode"
        ]
        == "normal_counterfactual"
    ].groupby(
        [
            "scenario_id",
            "window_mode",
        ]
    ):
        normal_false.append(
            float(
                g["alarm"].any()
            )
        )

    # Event trajectories.
    for sid, g in scored[
        scored[
            "window_mode"
        ]
        == "event"
    ].groupby(
        "scenario_id"
    ):
        sid = int(sid)
        m = meta_idx.loc[sid]

        event_type = str(
            m["event_type"]
        )
        event_family = str(
            m["event_family"]
        )

        t0 = float(
            m["event_start_s"]
        )
        t1 = float(
            m["event_end_s"]
        )

        pre = g[
            g["window_end_s"] < t0
        ]

        if len(pre):
            pre_event_false.append(
                float(
                    pre["alarm"].any()
                )
            )

        during = g[
            (g["window_end_s"] >= t0)
            & (g["window_end_s"] <= t1)
        ].sort_values(
            "window_end_s"
        )

        if event_family in CYBER_FAMILIES:
            hit = during[
                during["alarm"] == 1
            ]

            detected = (
                not hit.empty
            )

            cyber_hits.append(
                float(detected)
            )

            if detected:
                cyber_delays.append(
                    1000.0
                    * (
                        float(
                            hit.iloc[0][
                                "window_end_s"
                            ]
                        )
                        - t0
                    )
                )

            per_attack.append(
                {
                    "scenario_id": sid,
                    "event_type": event_type,
                    "detected": float(
                        detected
                    ),
                }
            )

        elif event_family == "physical":
            physical_false.append(
                float(
                    during["alarm"].any()
                )
            )

    attack_df = pd.DataFrame(
        per_attack
    )

    if not attack_df.empty:
        attack_rates = (
            attack_df.groupby(
                "event_type",
                as_index=False
            )["detected"]
            .mean()
            .rename(
                columns={
                    "detected":
                    "scenario_detection_rate"
                }
            )
        )
    else:
        attack_rates = pd.DataFrame(
            columns=[
                "event_type",
                "scenario_detection_rate",
            ]
        )

    metrics = {
        "cyber_scenario_detection_rate": (
            float(
                np.mean(cyber_hits)
            )
            if cyber_hits
            else np.nan
        ),
        "median_detection_delay_ms": (
            float(
                np.median(
                    cyber_delays
                )
            )
            if cyber_delays
            else np.nan
        ),
        "p95_detection_delay_ms": (
            float(
                np.percentile(
                    cyber_delays,
                    95
                )
            )
            if cyber_delays
            else np.nan
        ),
        "normal_trajectory_false_alarm_rate": (
            float(
                np.mean(
                    normal_false
                )
            )
            if normal_false
            else np.nan
        ),
        "physical_disturbance_false_alarm_rate": (
            float(
                np.mean(
                    physical_false
                )
            )
            if physical_false
            else np.nan
        ),
        "pre_event_false_alarm_rate": (
            float(
                np.mean(
                    pre_event_false
                )
            )
            if pre_event_false
            else np.nan
        ),
    }

    return metrics, attack_rates


def window_metrics(
    y_true,
    scores,
    threshold,
):
    y = np.asarray(
        y_true,
        dtype=int
    )

    pred = (
        np.asarray(scores)
        >= threshold
    ).astype(int)

    benign = y == 0

    return {
        "roc_auc": float(
            roc_auc_score(
                y,
                scores,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y,
                scores,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y,
                pred,
            )
        ),
        "f1": float(
            f1_score(
                y,
                pred,
            )
        ),
        "cyber_window_recall": float(
            recall_score(
                y,
                pred,
            )
        ),
        "benign_window_fpr": float(
            np.mean(
                pred[
                    benign
                ] == 1
            )
        ),
    }


def validation_policy_score(
    metrics,
):
    """
    Prefer high attack detection but explicitly punish
    physical/normal false alarms.
    """
    return (
        metrics[
            "cyber_scenario_detection_rate"
        ]
        - 1.5
        * metrics[
            "normal_trajectory_false_alarm_rate"
        ]
        - 1.5
        * metrics[
            "physical_disturbance_false_alarm_rate"
        ]
    )


def train_and_select_policy(
    train,
    val,
    cols,
    model_name,
    model,
    meta,
):
    X_train = train[
        cols
    ]
    y_train = train[
        "target_cyber"
    ]

    X_val = val[
        cols
    ]
    y_val = val[
        "target_cyber"
    ]

    model.fit(
        X_train,
        y_train,
    )

    val_scores = positive_scores(
        model,
        X_val,
    )

    threshold = (
        threshold_from_benign_validation(
            y_val,
            val_scores,
        )
    )

    scored_val = val.copy()
    scored_val[
        "score"
    ] = val_scores

    policy_rows = []

    for (
        policy_name,
        k,
        n,
    ) in PERSISTENCE_POLICIES:
        p = apply_persistence(
            scored_val,
            "score",
            threshold,
            k,
            n,
        )

        metrics, _ = (
            trajectory_metrics(
                p,
                meta,
            )
        )

        score = (
            validation_policy_score(
                metrics
            )
        )

        policy_rows.append(
            {
                "policy": policy_name,
                "k": k,
                "n": n,
                "validation_utility": score,
                **metrics,
            }
        )

    policy_df = pd.DataFrame(
        policy_rows
    ).sort_values(
        [
            "validation_utility",
            "cyber_scenario_detection_rate",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = policy_df.iloc[0]

    return (
        model,
        threshold,
        str(best["policy"]),
        int(best["k"]),
        int(best["n"]),
        policy_df,
    )


def evaluate_test(
    model,
    threshold,
    k,
    n,
    test,
    cols,
    meta,
):
    scores = positive_scores(
        model,
        test[cols],
    )

    wm = window_metrics(
        test[
            "target_cyber"
        ],
        scores,
        threshold,
    )

    scored = test.copy()
    scored["score"] = scores

    scored = apply_persistence(
        scored,
        "score",
        threshold,
        k,
        n,
    )

    tm, attack_rates = (
        trajectory_metrics(
            scored,
            meta,
        )
    )

    return (
        {
            **wm,
            **tm,
        },
        attack_rates,
    )


def standard_benchmark(
    df,
    meta,
    split_col,
    regime,
    feature_set,
):
    cols = feature_columns(
        df,
        feature_set,
    )

    train = df[
        df[split_col] == "train"
    ].copy()

    val = df[
        df[split_col]
        == "validation"
    ].copy()

    test = df[
        df[split_col] == "test"
    ].copy()

    rows = []
    attack_frames = []
    validation_frames = []

    for (
        model_name,
        model,
    ) in model_zoo().items():
        (
            fitted,
            threshold,
            policy_name,
            k,
            n,
            policy_df,
        ) = train_and_select_policy(
            train,
            val,
            cols,
            model_name,
            model,
            meta,
        )

        metrics, attack_rates = (
            evaluate_test(
                fitted,
                threshold,
                k,
                n,
                test,
                cols,
                meta,
            )
        )

        rows.append(
            {
                "regime": regime,
                "feature_set": feature_set,
                "model": model_name,
                "n_features": len(cols),
                "threshold": threshold,
                "persistence_policy":
                    policy_name,
                "persistence_k": k,
                "persistence_n": n,
                **metrics,
            }
        )

        if not attack_rates.empty:
            attack_rates[
                "regime"
            ] = regime
            attack_rates[
                "feature_set"
            ] = feature_set
            attack_rates[
                "model"
            ] = model_name
            attack_frames.append(
                attack_rates
            )

        policy_df[
            "regime"
        ] = regime
        policy_df[
            "feature_set"
        ] = feature_set
        policy_df[
            "model"
        ] = model_name
        validation_frames.append(
            policy_df
        )

    return (
        pd.DataFrame(rows),
        (
            pd.concat(
                attack_frames,
                ignore_index=True,
            )
            if attack_frames
            else pd.DataFrame()
        ),
        pd.concat(
            validation_frames,
            ignore_index=True,
        ),
    )


def unseen_attack_benchmark(
    df,
    meta,
):
    """
    Leave-one-cyber-event-type-out (LOETO):
    held-out attack type is completely removed from training/validation.
    Test asks whether a detector trained on other attacks can still flag it.
    """
    cols = feature_columns(
        df,
        "inverter_only",
    )

    rows = []

    for held_out in CYBER_EVENT_TYPES:
        print(
            f"\nZero-day holdout: {held_out}"
        )

        # Completely remove held-out type from train/validation.
        train = df[
            (df["iid_split"] == "train")
            & (
                df["event_type"]
                != held_out
            )
        ].copy()

        val = df[
            (
                df["iid_split"]
                == "validation"
            )
            & (
                df["event_type"]
                != held_out
            )
        ].copy()

        # Test set:
        # - all windows from held-out type (all its scenarios; never trained)
        # - IID test benign windows from normal/physical cases.
        held = df[
            df["event_type"]
            == held_out
        ].copy()

        benign_test = df[
            (
                df["iid_split"]
                == "test"
            )
            & (
                df["target_cyber"]
                == 0
            )
            & (
                ~df[
                    "event_type"
                ].isin(
                    CYBER_EVENT_TYPES
                )
            )
        ].copy()

        test = pd.concat(
            [
                held,
                benign_test,
            ],
            ignore_index=True,
        )

        for (
            model_name,
            model,
        ) in model_zoo().items():
            (
                fitted,
                threshold,
                policy_name,
                k,
                n,
                _,
            ) = train_and_select_policy(
                train,
                val,
                cols,
                model_name,
                model,
                meta,
            )

            metrics, attack_rates = (
                evaluate_test(
                    fitted,
                    threshold,
                    k,
                    n,
                    test,
                    cols,
                    meta,
                )
            )

            held_rate = np.nan

            if not attack_rates.empty:
                x = attack_rates[
                    attack_rates[
                        "event_type"
                    ]
                    == held_out
                ]

                if not x.empty:
                    held_rate = float(
                        x.iloc[0][
                            "scenario_detection_rate"
                        ]
                    )

            rows.append(
                {
                    "held_out_attack":
                        held_out,
                    "model": model_name,
                    "threshold": threshold,
                    "persistence_policy":
                        policy_name,
                    "zero_day_scenario_detection_rate":
                        held_rate,
                    "normal_trajectory_false_alarm_rate":
                        metrics[
                            "normal_trajectory_false_alarm_rate"
                        ],
                    "physical_disturbance_false_alarm_rate":
                        metrics[
                            "physical_disturbance_false_alarm_rate"
                        ],
                    "median_detection_delay_ms":
                        metrics[
                            "median_detection_delay_ms"
                        ],
                }
            )

    return pd.DataFrame(
        rows
    )


def plot_standard_results(results):
    d = results[
        results[
            "feature_set"
        ]
        == "inverter_only"
    ].copy()

    labels = (
        d["regime"]
        + "\n"
        + d["model"]
        + "\n"
        + d["persistence_policy"]
    ).tolist()

    x = np.arange(
        len(d)
    )
    width = 0.25

    detect = (
        100
        * d[
            "cyber_scenario_detection_rate"
        ].to_numpy()
    )
    normal_fa = (
        100
        * d[
            "normal_trajectory_false_alarm_rate"
        ].to_numpy()
    )
    physical_fa = (
        100
        * d[
            "physical_disturbance_false_alarm_rate"
        ].to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    ax.bar(
        x - width,
        detect,
        width,
        label="Cyber detection",
    )
    ax.bar(
        x,
        normal_fa,
        width,
        label="Normal false alarm",
    )
    ax.bar(
        x + width,
        physical_fa,
        width,
        label="Physical-event false alarm",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
        fontsize=8.5,
    )
    ax.set_ylim(
        0,
        105
    )
    ax.set_ylabel(
        "Trajectory-level rate (%)"
    )
    ax.set_title(
        "Cyber Attack Detection vs Legitimate Physical Disturbances"
    )
    ax.legend(
        frameon=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "36_cyber_vs_physical_streaming.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_zero_day(zero_day):
    pivot = (
        zero_day.pivot(
            index="held_out_attack",
            columns="model",
            values=(
                "zero_day_scenario_detection_rate"
            ),
        )
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(11, 6.8)
    )

    x = np.arange(
        len(pivot)
    )
    models = list(
        pivot.columns
    )

    width = (
        0.75
        / max(
            1,
            len(models)
        )
    )

    for j, model in enumerate(
        models
    ):
        offset = (
            j
            - (
                len(models) - 1
            ) / 2
        ) * width

        ax.bar(
            x + offset,
            pivot[
                model
            ].to_numpy(),
            width,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        pivot.index,
        rotation=25,
        ha="right",
    )
    ax.set_ylim(
        0,
        105
    )
    ax.set_ylabel(
        "Zero-day scenario detection rate (%)"
    )
    ax.set_title(
        "Leave-One-Attack-Type-Out Generalization"
    )
    ax.legend(
        frameon=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "37_zero_day_attack_generalization.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    if not STREAM_PATH.exists():
        raise FileNotFoundError(
            "Run Phase 2I first: "
            + str(STREAM_PATH)
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            str(META_PATH)
        )

    print(
        "\n=== PHASE 2J FIXED: CYBER-vs-PHYSICAL + ZERO-DAY BENCHMARK ==="
    )
    print(
        "Persistence implementation: index-safe pandas .loc assignment"
    )

    windows = pd.read_csv(
        STREAM_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    df = add_cyber_target(
        windows
    )

    print(
        f"Usable causal windows : {len(df):,}"
    )
    print(
        f"Cyber-positive windows: "
        f"{int(df['target_cyber'].sum()):,}"
    )
    print(
        f"Benign windows        : "
        f"{int((df['target_cyber']==0).sum()):,}"
    )

    print(
        "\nBenign composition:"
    )
    print(
        df[
            df[
                "target_cyber"
            ] == 0
        ][
            "benign_kind"
        ].value_counts().to_string()
    )

    all_results = []
    all_attacks = []
    all_policy = []

    for split_col, regime in [
        (
            "iid_split",
            "IID_group_safe",
        ),
        (
            "ood_split",
            "OOD_parameter_shift",
        ),
    ]:
        for feature_set in [
            "inverter_only",
            "physics_proxy",
        ]:
            print(
                f"\nRunning {regime} / "
                f"{feature_set} ..."
            )

            (
                result,
                attacks,
                policy,
            ) = standard_benchmark(
                df,
                meta,
                split_col,
                regime,
                feature_set,
            )

            all_results.append(
                result
            )

            if not attacks.empty:
                all_attacks.append(
                    attacks
                )

            all_policy.append(
                policy
            )

            print(
                result[
                    [
                        "model",
                        "persistence_policy",
                        "roc_auc",
                        "cyber_window_recall",
                        "benign_window_fpr",
                        "cyber_scenario_detection_rate",
                        "normal_trajectory_false_alarm_rate",
                        "physical_disturbance_false_alarm_rate",
                        "median_detection_delay_ms",
                    ]
                ].to_string(
                    index=False
                )
            )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )

    attacks = (
        pd.concat(
            all_attacks,
            ignore_index=True,
        )
        if all_attacks
        else pd.DataFrame()
    )

    policy = pd.concat(
        all_policy,
        ignore_index=True,
    )

    print(
        "\nRunning leave-one-attack-type-out "
        "zero-day evaluation ..."
    )

    zero_day = unseen_attack_benchmark(
        df,
        meta,
    )

    results.to_csv(
        RESULTS
        / "phase2j_cyber_vs_physical_metrics.csv",
        index=False,
    )

    attacks.to_csv(
        RESULTS
        / "phase2j_detection_by_attack_type.csv",
        index=False,
    )

    policy.to_csv(
        RESULTS
        / "phase2j_validation_persistence_selection.csv",
        index=False,
    )

    zero_day.to_csv(
        RESULTS
        / "phase2j_zero_day_generalization.csv",
        index=False,
    )

    plot_standard_results(
        results
    )

    plot_zero_day(
        zero_day
    )

    print(
        "\n=== CYBER-vs-PHYSICAL SUMMARY ==="
    )
    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\n=== ZERO-DAY ATTACK GENERALIZATION ==="
    )
    print(
        zero_day.to_string(
            index=False
        )
    )

    if not attacks.empty:
        print(
            "\n=== DETECTION BY ATTACK TYPE ==="
        )
        print(
            attacks.to_string(
                index=False
            )
        )

    print(
        "\nSaved:"
    )
    print(
        "  results/phase2j_cyber_vs_physical_metrics.csv"
    )
    print(
        "  results/phase2j_detection_by_attack_type.csv"
    )
    print(
        "  results/phase2j_validation_persistence_selection.csv"
    )
    print(
        "  results/phase2j_zero_day_generalization.csv"
    )
    print(
        "  figures/36_cyber_vs_physical_streaming.png"
    )
    print(
        "  figures/37_zero_day_attack_generalization.png"
    )

    print(
        "\nWHY THIS IS THE REAL RESEARCH TEST:\n"
        "The positive class is now cyber/hybrid attack only. Legitimate "
        "physical cloud, load-spike and voltage-sag events are hard negatives. "
        "The detector therefore has to distinguish cyber-physical inconsistency "
        "from real grid disturbances, rather than merely detecting any abnormal "
        "transient. Persistence logic is selected on validation data to reduce "
        "repeated-window false alarms. Finally, leave-one-attack-type-out testing "
        "measures whether the detector can recognize an attack family it never "
        "saw during training. If performance drops here, that gap is precisely "
        "where WLS network states, graph topology and physics-aware models can "
        "provide a meaningful next contribution."
    )


if __name__ == "__main__":
    main()
