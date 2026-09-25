from pathlib import Path
import importlib.util
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

CHALLENGE_RAW_PATH = (
    DATA
    / "phase2m_ambiguous_challenge_timeseries.csv.gz"
)

CHALLENGE_META_PATH = (
    DATA
    / "phase2m_ambiguous_challenge_metadata.csv"
)

BASELINE_RAW_PATH = (
    DATA
    / "phase2g_v2_transient_timeseries.csv.gz"
)

BASELINE_META_PATH = (
    DATA
    / "phase2g_v2_scenario_metadata.csv"
)

RANDOM_STATE = 20260812

# Same representative monitor uncertainty used in Phase 2M.
SIGMA_PU = 0.005
GAIN_SIGMA = 0.002
COMMAND_DELAY_MS = 20

TARGET_BENIGN_FPR = 0.01

POSITIVE_CLASS = (
    "cyber_mild_gfl_current_limit"
)

NEGATIVE_CLASSES = [
    "physical_deep_sag_with_gfl_limiting",
    "physical_fast_pv_ramp",
]


def load_stress_module():
    candidates = [
        ROOT
        / "16_command_residual_stress_test.py",
    ]

    for path in candidates:
        if path.exists():
            spec = (
                importlib.util
                .spec_from_file_location(
                    "phase2l",
                    path,
                )
            )

            module = (
                importlib.util
                .module_from_spec(
                    spec
                )
            )

            spec.loader.exec_module(
                module
            )

            return module

    raise FileNotFoundError(
        "Missing "
        "16_command_residual_stress_test.py"
    )


stress = load_stress_module()


def longest_true_run_ms(
    flags,
    sample_dt_s,
):
    flags = np.asarray(
        flags,
        dtype=bool,
    )

    longest = 0
    current = 0

    for value in flags:
        if value:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return float(
        1000.0
        * longest
        * sample_dt_s
    )


def safe_slope(
    t,
    x,
):
    t = np.asarray(
        t,
        dtype=float,
    )
    x = np.asarray(
        x,
        dtype=float,
    )

    if len(t) < 3:
        return 0.0

    t0 = t - t.mean()
    denom = np.sum(
        t0 * t0
    )

    if denom <= 1e-15:
        return 0.0

    return float(
        np.sum(
            t0
            * (
                x - x.mean()
            )
        )
        / denom
    )


def baseline_residual_threshold(
    baseline_raw,
    baseline_meta,
):
    d = stress.scenario_residuals(
        baseline_raw,
        baseline_meta,
        SIGMA_PU,
        GAIN_SIGMA,
        COMMAND_DELAY_MS,
        RANDOM_STATE + 700,
    )

    return stress.calibrate_threshold(
        d
    )


def assign_splits(
    meta,
):
    """
    Stratified scenario-level split:
      12 train / 6 validation / 6 test per challenge class.

    Event and matched normal counterfactual always share scenario_id,
    so there is no pair leakage.
    """
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    split_map = {}

    for challenge, g in meta.groupby(
        "challenge_type"
    ):
        ids = g[
            "scenario_id"
        ].astype(int).to_numpy()

        ids = rng.permutation(
            ids
        )

        if len(ids) != 24:
            raise RuntimeError(
                f"Expected 24 scenarios for "
                f"{challenge}, got {len(ids)}"
            )

        for sid in ids[:12]:
            split_map[
                int(sid)
            ] = "train"

        for sid in ids[12:18]:
            split_map[
                int(sid)
            ] = "validation"

        for sid in ids[18:]:
            split_map[
                int(sid)
            ] = "test"

    return split_map


def independent_residual_trace(
    g,
    m,
    local_seed,
):
    g = g.sort_values(
        "time_s"
    )

    t = g[
        "time_s"
    ].to_numpy(
        dtype=float
    )

    p_ref = g[
        "pv_pref_pu"
    ].to_numpy(
        dtype=float
    )

    p_ref_seen = (
        stress.shift_command(
            t,
            p_ref,
            COMMAND_DELAY_MS
            / 1000.0,
        )
    )

    rng = np.random.default_rng(
        local_seed
    )

    v_obs, i_obs = (
        stress
        .independent_monitor_measurements(
            rng,
            g[
                "v_gfl_measured_pu"
            ].to_numpy(
                dtype=float
            ),
            g[
                "gfl_id_pu"
            ].to_numpy(
                dtype=float
            ),
            SIGMA_PU,
            GAIN_SIGMA,
        )
    )

    residual = np.abs(
        p_ref_seen
        - v_obs * i_obs
    )

    return (
        t,
        p_ref,
        p_ref_seen,
        residual,
    )


def trajectory_features(
    raw,
    meta,
    residual_threshold,
    split_map,
):
    meta_idx = (
        meta.set_index(
            "scenario_id"
        )
    )

    rows = []

    master = np.random.default_rng(
        RANDOM_STATE + 1700
    )

    grouped = raw.groupby(
        [
            "scenario_id",
            "window_mode",
        ],
        sort=True,
    )

    for (
        sid,
        mode,
    ), g in grouped:
        sid = int(sid)
        m = meta_idx.loc[sid]

        local_seed = int(
            master.integers(
                0,
                2**31 - 1,
            )
        )

        (
            t,
            p_ref,
            p_ref_seen,
            residual,
        ) = independent_residual_trace(
            g,
            m,
            local_seed,
        )

        t0 = float(
            m[
                "event_start_s"
            ]
        )
        t1 = float(
            m[
                "event_end_s"
            ]
        )

        mask = (
            (t >= t0)
            & (t < t1)
        )

        if not np.any(mask):
            continue

        d = g.sort_values(
            "time_s"
        ).loc[
            mask
        ]

        tt = t[
            mask
        ]
        rr = residual[
            mask
        ]
        pref = p_ref[
            mask
        ]

        if len(tt) > 1:
            sample_dt = float(
                np.median(
                    np.diff(tt)
                )
            )
        else:
            sample_dt = 0.005

        above = (
            rr
            >= residual_threshold
        )

        pref_diff = np.diff(
            pref
        )

        pref_tv = float(
            np.sum(
                np.abs(
                    pref_diff
                )
            )
        ) if len(pref_diff) else 0.0

        pref_range = float(
            np.max(pref)
            - np.min(pref)
        )

        row = {
            "scenario_id": sid,
            "window_mode": mode,
            "challenge_type": str(
                m[
                    "challenge_type"
                ]
            ),
            "dataset_split": str(
                split_map[
                    sid
                ]
            ),

            # Cyber-positive only for actual event trajectory.
            "target_cyber": int(
                mode == "event"
                and str(
                    m[
                        "challenge_type"
                    ]
                )
                == POSITIVE_CLASS
            ),

            # ----------------------------
            # Mechanism residual features
            # ----------------------------
            "cmd_res_max_pu": float(
                np.max(rr)
            ),
            "cmd_res_rms_pu": float(
                np.sqrt(
                    np.mean(
                        rr ** 2
                    )
                )
            ),
            "cmd_res_median_pu": float(
                np.median(rr)
            ),
            "cmd_res_p95_pu": float(
                np.percentile(
                    rr,
                    95,
                )
            ),
            "cmd_res_fraction_above": float(
                np.mean(
                    above
                )
            ),
            "cmd_res_longest_run_ms":
                longest_true_run_ms(
                    above,
                    sample_dt,
                ),

            # ----------------------------
            # Network-context proxy
            # IMPORTANT:
            # this currently uses simulation truth/proxy.
            # It is NOT yet a deployable WLS estimate.
            # ----------------------------
            "network_min_v_pu": float(
                d[
                    "min_bus_v_pu"
                ].min()
            ),
            "network_mean_min_v_pu":
                float(
                    d[
                        "min_bus_v_pu"
                    ].mean()
                ),
            "network_low_v_fraction_085":
                float(
                    (
                        d[
                            "min_bus_v_pu"
                        ]
                        < 0.85
                    ).mean()
                ),

            # ----------------------------
            # Trusted PV-reference dynamics
            # Helps distinguish a legitimate
            # fast resource-availability ramp
            # from a persistent actuator limit.
            # ----------------------------
            "pref_range_pu": pref_range,
            "pref_total_variation_pu":
                pref_tv,
            "pref_max_abs_slope_pu_s":
                float(
                    np.max(
                        np.abs(
                            np.gradient(
                                pref,
                                tt,
                            )
                        )
                    )
                    if len(tt) >= 3
                    else 0.0
                ),

            # Diagnostic only.
            "gfl_limiter_fraction": float(
                d[
                    "gfl_limiter"
                ].astype(float).mean()
            ),
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def metric_dict(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    positive = (
        y_true == 1
    )
    negative = (
        y_true == 0
    )

    return {
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "cyber_detection_rate": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "benign_false_alarm_rate": float(
            np.mean(
                y_pred[
                    negative
                ] == 1
            )
            if negative.any()
            else np.nan
        ),
    }


def event_only_test(
    d,
):
    """
    Primary hard-negative evaluation:
    cyber mild current-limit vs legitimate physical events.

    Normal counterfactuals are evaluated separately as an additional check.
    """
    return d[
        d[
            "window_mode"
        ] == "event"
    ].copy()


def evaluate_max_residual_baseline(
    feature_df,
    residual_threshold,
):
    test = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "test"
        ]
    )

    pred = (
        test[
            "cmd_res_max_pu"
        ].to_numpy()
        >= residual_threshold
    ).astype(int)

    metrics = metric_dict(
        test[
            "target_cyber"
        ],
        pred,
    )

    return (
        metrics,
        test.assign(
            prediction=pred,
            detector="max_residual_only",
        ),
    )


def select_interpretable_rule(
    feature_df,
):
    """
    Rule:
      persistent command inconsistency
      AND no severe system-wide low voltage
      AND no large PV-reference ramp

    Thresholds are selected ONLY on validation event trajectories.
    """
    val = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "validation"
        ]
    )

    frac_grid = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.65,
        0.80,
    ]

    voltage_grid = [
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.93,
        0.95,
    ]

    pref_grid = [
        0.005,
        0.010,
        0.020,
        0.040,
        0.080,
        0.120,
        0.200,
    ]

    rows = []

    y = val[
        "target_cyber"
    ].to_numpy(
        dtype=int
    )

    for (
        tau_frac,
        tau_v,
        tau_pref,
    ) in itertools.product(
        frac_grid,
        voltage_grid,
        pref_grid,
    ):
        pred = (
            (
                val[
                    "cmd_res_fraction_above"
                ]
                >= tau_frac
            )
            & (
                val[
                    "network_min_v_pu"
                ]
                >= tau_v
            )
            & (
                val[
                    "pref_range_pu"
                ]
                <= tau_pref
            )
        ).astype(int)

        metrics = metric_dict(
            y,
            pred,
        )

        # Strong penalty on physical false alarms.
        utility = (
            metrics[
                "cyber_detection_rate"
            ]
            - 1.25
            * metrics[
                "benign_false_alarm_rate"
            ]
        )

        rows.append(
            {
                "tau_residual_fraction":
                    tau_frac,
                "tau_network_min_v_pu":
                    tau_v,
                "tau_pref_range_pu":
                    tau_pref,
                "validation_utility":
                    utility,
                **metrics,
            }
        )

    search = pd.DataFrame(
        rows
    )

    best = (
        search.sort_values(
            [
                "validation_utility",
                "balanced_accuracy",
                "benign_false_alarm_rate",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .iloc[0]
    )

    return (
        best,
        search,
    )


def evaluate_rule(
    feature_df,
    best,
):
    test = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "test"
        ]
    )

    pred = (
        (
            test[
                "cmd_res_fraction_above"
            ]
            >= float(
                best[
                    "tau_residual_fraction"
                ]
            )
        )
        & (
            test[
                "network_min_v_pu"
            ]
            >= float(
                best[
                    "tau_network_min_v_pu"
                ]
            )
        )
        & (
            test[
                "pref_range_pu"
            ]
            <= float(
                best[
                    "tau_pref_range_pu"
                ]
            )
        )
    ).astype(int)

    metrics = metric_dict(
        test[
            "target_cyber"
        ],
        pred,
    )

    return (
        metrics,
        test.assign(
            prediction=pred,
            detector="interpretable_physics_rule",
        ),
    )


def compact_feature_sets():
    return {
        "residual_temporal": [
            "cmd_res_max_pu",
            "cmd_res_rms_pu",
            "cmd_res_fraction_above",
            "cmd_res_longest_run_ms",
        ],

        "physics_fusion": [
            "cmd_res_max_pu",
            "cmd_res_rms_pu",
            "cmd_res_fraction_above",
            "cmd_res_longest_run_ms",
            "network_min_v_pu",
            "network_low_v_fraction_085",
            "pref_range_pu",
            "pref_total_variation_pu",
            "pref_max_abs_slope_pu_s",
        ],
    }


def ml_models():
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
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
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
                        n_estimators=500,
                        max_depth=5,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_ml(
    feature_df,
):
    rows = []
    pred_frames = []

    # Train/validation are event trajectories only.
    train = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "train"
        ]
    )

    val = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "validation"
        ]
    )

    test = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "test"
        ]
    )

    for (
        feature_set,
        cols,
    ) in compact_feature_sets().items():

        for (
            model_name,
            model,
        ) in ml_models().items():

            model.fit(
                train[cols],
                train[
                    "target_cyber"
                ],
            )

            # Select probability threshold on validation,
            # rather than blindly using 0.5.
            val_score = (
                model.predict_proba(
                    val[cols]
                )[:, 1]
            )

            thresholds = np.unique(
                np.quantile(
                    val_score,
                    np.linspace(
                        0.05,
                        0.95,
                        80,
                    ),
                )
            )

            best_threshold = 0.5
            best_utility = -np.inf

            for threshold in thresholds:
                pred = (
                    val_score
                    >= threshold
                ).astype(int)

                m = metric_dict(
                    val[
                        "target_cyber"
                    ],
                    pred,
                )

                utility = (
                    m[
                        "cyber_detection_rate"
                    ]
                    - 1.25
                    * m[
                        "benign_false_alarm_rate"
                    ]
                )

                if utility > best_utility:
                    best_utility = utility
                    best_threshold = float(
                        threshold
                    )

            test_score = (
                model.predict_proba(
                    test[cols]
                )[:, 1]
            )

            test_pred = (
                test_score
                >= best_threshold
            ).astype(int)

            metrics = metric_dict(
                test[
                    "target_cyber"
                ],
                test_pred,
            )

            rows.append(
                {
                    "detector":
                        f"{feature_set}__{model_name}",
                    "feature_set":
                        feature_set,
                    "model":
                        model_name,
                    "threshold":
                        best_threshold,
                    **metrics,
                }
            )

            p = test.copy()
            p[
                "prediction"
            ] = test_pred
            p[
                "score"
            ] = test_score
            p[
                "detector"
            ] = (
                f"{feature_set}__"
                f"{model_name}"
            )

            pred_frames.append(
                p
            )

    return (
        pd.DataFrame(
            rows
        ),
        pd.concat(
            pred_frames,
            ignore_index=True,
        ),
    )


def class_specific_rates(
    predictions,
):
    d = predictions.copy()

    return (
        d.groupby(
            [
                "detector",
                "challenge_type",
            ],
            as_index=False,
        )
        .agg(
            n=(
                "scenario_id",
                "count",
            ),
            alarm_rate=(
                "prediction",
                "mean",
            ),
        )
    )


def normal_counterfactual_alarm(
    feature_df,
    detector_name,
    prediction_function,
):
    test = feature_df[
        (
            feature_df[
                "dataset_split"
            ] == "test"
        )
        & (
            feature_df[
                "window_mode"
            ]
            == "normal_counterfactual"
        )
    ].copy()

    pred = prediction_function(
        test
    )

    return float(
        np.mean(
            pred
        )
    )


def plot_detector_comparison(
    results,
):
    d = results.sort_values(
        "balanced_accuracy",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    y = np.arange(
        len(d)
    )

    ax.barh(
        y - 0.18,
        100*d[
            "cyber_detection_rate"
        ],
        height=0.34,
        label="Cyber detection",
    )

    ax.barh(
        y + 0.18,
        100*d[
            "benign_false_alarm_rate"
        ],
        height=0.34,
        label="Physical false alarm",
    )

    ax.set_yticks(
        y
    )
    ax.set_yticklabels(
        d[
            "detector"
        ]
    )
    ax.set_xlabel(
        "Test scenario rate (%)"
    )
    ax.set_xlim(
        0,
        105,
    )
    ax.set_title(
        "Ambiguous Challenge — Minimal Physics Fusion vs Residual-Only"
    )
    ax.legend(
        frameon=False
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )
    ax.spines[
        "right"
    ].set_visible(
        False
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "44_minimal_physics_fusion_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_feature_plane(
    feature_df,
    best_rule,
):
    test = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "test"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 6.8)
    )

    for challenge, g in test.groupby(
        "challenge_type"
    ):
        ax.scatter(
            g[
                "network_min_v_pu"
            ],
            g[
                "cmd_res_fraction_above"
            ],
            s=65,
            alpha=0.80,
            label=challenge,
        )

    ax.axvline(
        float(
            best_rule[
                "tau_network_min_v_pu"
            ]
        ),
        linestyle="--",
        linewidth=1.2,
    )

    ax.axhline(
        float(
            best_rule[
                "tau_residual_fraction"
            ]
        ),
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel(
        "Minimum network voltage proxy (pu)"
    )
    ax.set_ylabel(
        "Fraction of event with command residual above threshold"
    )
    ax.set_title(
        "Mechanism Residual + Network Context on Held-Out Test Scenarios"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )
    ax.spines[
        "right"
    ].set_visible(
        False
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "45_residual_vs_network_voltage_plane.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_pref_vs_residual(
    feature_df,
):
    test = event_only_test(
        feature_df[
            feature_df[
                "dataset_split"
            ] == "test"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 6.8)
    )

    for challenge, g in test.groupby(
        "challenge_type"
    ):
        ax.scatter(
            g[
                "pref_range_pu"
            ],
            g[
                "cmd_res_longest_run_ms"
            ],
            s=65,
            alpha=0.80,
            label=challenge,
        )

    ax.set_xlabel(
        "Trusted PV-reference range during event (pu)"
    )
    ax.set_ylabel(
        "Longest command-residual exceedance (ms)"
    )
    ax.set_title(
        "Temporal Separation of Fast PV Ramps and Persistent Current Limiting"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )
    ax.spines[
        "right"
    ].set_visible(
        False
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "46_pref_dynamics_vs_residual_persistence.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    for path in [
        CHALLENGE_RAW_PATH,
        CHALLENGE_META_PATH,
        BASELINE_RAW_PATH,
        BASELINE_META_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    print(
        "\n=== PHASE 2N: INTERPRETABLE PHYSICS-FUSION DIAGNOSTIC ==="
    )

    challenge_raw = pd.read_csv(
        CHALLENGE_RAW_PATH
    )

    challenge_meta = pd.read_csv(
        CHALLENGE_META_PATH
    )

    baseline_raw = pd.read_csv(
        BASELINE_RAW_PATH
    )

    baseline_meta = pd.read_csv(
        BASELINE_META_PATH
    )

    residual_threshold = (
        baseline_residual_threshold(
            baseline_raw,
            baseline_meta,
        )
    )

    split_map = assign_splits(
        challenge_meta
    )

    feature_df = trajectory_features(
        challenge_raw,
        challenge_meta,
        residual_threshold,
        split_map,
    )

    feature_df.to_csv(
        DATA
        / "phase2n_fusion_feature_dataset.csv",
        index=False,
    )

    print(
        f"Baseline command-residual threshold: "
        f"{residual_threshold:.6f} pu"
    )

    print(
        "\nScenario split:"
    )
    print(
        feature_df[
            feature_df[
                "window_mode"
            ] == "event"
        ].groupby(
            [
                "challenge_type",
                "dataset_split",
            ]
        )[
            "scenario_id"
        ].nunique().to_string()
    )

    # ---------------------------------------------------------
    # Baseline 1: max residual only
    # ---------------------------------------------------------
    (
        max_metrics,
        max_pred,
    ) = evaluate_max_residual_baseline(
        feature_df,
        residual_threshold,
    )

    # ---------------------------------------------------------
    # Baseline 2: interpretable physics fusion rule
    # ---------------------------------------------------------
    (
        best_rule,
        rule_search,
    ) = select_interpretable_rule(
        feature_df
    )

    (
        rule_metrics,
        rule_pred,
    ) = evaluate_rule(
        feature_df,
        best_rule,
    )

    # ---------------------------------------------------------
    # Compact ML baselines
    # ---------------------------------------------------------
    (
        ml_results,
        ml_predictions,
    ) = evaluate_ml(
        feature_df
    )

    result_rows = [
        {
            "detector":
                "max_residual_only",
            "feature_set":
                "single_residual",
            "model":
                "fixed_threshold",
            "threshold":
                residual_threshold,
            **max_metrics,
        },
        {
            "detector":
                "interpretable_physics_rule",
            "feature_set":
                "residual+network+pref",
            "model":
                "validation_selected_rule",
            "threshold":
                np.nan,
            **rule_metrics,
        },
    ]

    results = pd.concat(
        [
            pd.DataFrame(
                result_rows
            ),
            ml_results,
        ],
        ignore_index=True,
    )

    predictions = pd.concat(
        [
            max_pred,
            rule_pred,
            ml_predictions,
        ],
        ignore_index=True,
    )

    class_rates = (
        class_specific_rates(
            predictions
        )
    )

    # Normal counterfactual checks for the two transparent detectors.
    normal_test = feature_df[
        (
            feature_df[
                "dataset_split"
            ] == "test"
        )
        & (
            feature_df[
                "window_mode"
            ] == "normal_counterfactual"
        )
    ].copy()

    normal_max_fa = float(
        (
            normal_test[
                "cmd_res_max_pu"
            ]
            >= residual_threshold
        ).mean()
    )

    normal_rule_pred = (
        (
            normal_test[
                "cmd_res_fraction_above"
            ]
            >= float(
                best_rule[
                    "tau_residual_fraction"
                ]
            )
        )
        & (
            normal_test[
                "network_min_v_pu"
            ]
            >= float(
                best_rule[
                    "tau_network_min_v_pu"
                ]
            )
        )
        & (
            normal_test[
                "pref_range_pu"
            ]
            <= float(
                best_rule[
                    "tau_pref_range_pu"
                ]
            )
        )
    ).astype(int)

    normal_rule_fa = float(
        normal_rule_pred.mean()
    )

    transparent_extra = pd.DataFrame(
        [
            {
                "detector":
                    "max_residual_only",
                "normal_counterfactual_false_alarm_rate":
                    normal_max_fa,
            },
            {
                "detector":
                    "interpretable_physics_rule",
                "normal_counterfactual_false_alarm_rate":
                    normal_rule_fa,
            },
        ]
    )

    results = results.merge(
        transparent_extra,
        on="detector",
        how="left",
    )

    results.to_csv(
        RESULTS
        / "phase2n_fusion_test_metrics.csv",
        index=False,
    )

    class_rates.to_csv(
        RESULTS
        / "phase2n_fusion_alarm_by_challenge_type.csv",
        index=False,
    )

    rule_search.to_csv(
        RESULTS
        / "phase2n_rule_validation_search.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "tau_residual_fraction":
                    float(
                        best_rule[
                            "tau_residual_fraction"
                        ]
                    ),
                "tau_network_min_v_pu":
                    float(
                        best_rule[
                            "tau_network_min_v_pu"
                        ]
                    ),
                "tau_pref_range_pu":
                    float(
                        best_rule[
                            "tau_pref_range_pu"
                        ]
                    ),
                "validation_utility":
                    float(
                        best_rule[
                            "validation_utility"
                        ]
                    ),
            }
        ]
    ).to_csv(
        RESULTS
        / "phase2n_selected_physics_rule.csv",
        index=False,
    )

    plot_detector_comparison(
        results
    )

    plot_feature_plane(
        feature_df,
        best_rule,
    )

    plot_pref_vs_residual(
        feature_df
    )

    print(
        "\n=== SELECTED INTERPRETABLE RULE ==="
    )
    print(
        f"Residual exceedance fraction >= "
        f"{best_rule['tau_residual_fraction']:.3f}"
    )
    print(
        f"AND network min voltage >= "
        f"{best_rule['tau_network_min_v_pu']:.3f} pu"
    )
    print(
        f"AND PV-reference range <= "
        f"{best_rule['tau_pref_range_pu']:.3f} pu"
    )

    print(
        "\n=== TEST DETECTOR COMPARISON ==="
    )
    print(
        results[
            [
                "detector",
                "balanced_accuracy",
                "f1",
                "precision",
                "cyber_detection_rate",
                "benign_false_alarm_rate",
                "normal_counterfactual_false_alarm_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== TEST ALARM RATE BY CHALLENGE TYPE ==="
    )
    print(
        class_rates.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )
    print(
        "  data/phase2n_fusion_feature_dataset.csv"
    )
    print(
        "  results/phase2n_fusion_test_metrics.csv"
    )
    print(
        "  results/phase2n_fusion_alarm_by_challenge_type.csv"
    )
    print(
        "  results/phase2n_rule_validation_search.csv"
    )
    print(
        "  results/phase2n_selected_physics_rule.csv"
    )
    print(
        "  figures/44_minimal_physics_fusion_comparison.png"
    )
    print(
        "  figures/45_residual_vs_network_voltage_plane.png"
    )
    print(
        "  figures/46_pref_dynamics_vs_residual_persistence.png"
    )

    print(
        "\nINTERPRETATION RULE:\n"
        "This is still a diagnostic, not the final cyber detector. "
        "The network_min_v feature is a simulation/network-state proxy, "
        "not yet an independently estimated state. If this compact fusion "
        "substantially reduces the deep-sag and fast-PV false alarms while "
        "retaining mild current-limit detection, the next step is to replace "
        "that proxy with noisy AC WLS state-estimation outputs and then add "
        "bus/edge topology for localization. If the compact fusion still "
        "fails, a richer graph layer is justified immediately."
    )


if __name__ == "__main__":
    main()
