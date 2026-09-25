from pathlib import Path
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
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FEATURE_PATH = (
    DATA
    / "phase2o_wls_fusion_feature_dataset.csv"
)

N_REPEATS = 100
BASE_SEED = 20260812

POSITIVE_CLASS = "cyber_mild_gfl_current_limit"


FEATURE_SETS = {
    "residual_temporal": [
        "cmd_res_max_pu",
        "cmd_res_rms_pu",
        "cmd_res_fraction_above",
        "cmd_res_longest_run_ms",
        "pref_range_pu",
        "pref_total_variation_pu",
        "pref_max_abs_slope_pu_s",
    ],

    "truth_proxy_fusion": [
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

    "wls_fusion": [
        "cmd_res_max_pu",
        "cmd_res_rms_pu",
        "cmd_res_fraction_above",
        "cmd_res_longest_run_ms",
        "wls_min_v_pu",
        "wls_mean_min_v_pu",
        "wls_voltage_spread_pu",
        "wls_max_norm_residual",
        "wls_rms_norm_residual",
        "pref_range_pu",
        "pref_total_variation_pu",
        "pref_max_abs_slope_pu_s",
    ],
}


def model_zoo(seed):
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
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
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
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


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
                zero_division=0,
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
        "physical_false_alarm_rate": float(
            np.mean(
                y_pred[
                    negative
                ] == 1
            )
            if negative.any()
            else np.nan
        ),
    }


def make_split(
    event_df,
    seed,
):
    """
    24 scenarios per challenge class:
      12 train
       6 validation
       6 test

    Same split used by every model/feature set
    within a repeat, enabling paired comparisons.
    """
    rng = np.random.default_rng(
        seed
    )

    mapping = {}

    for challenge, g in event_df.groupby(
        "challenge_type",
        sort=True,
    ):
        ids = (
            g[
                "scenario_id"
            ]
            .astype(int)
            .unique()
        )

        if len(ids) != 24:
            raise RuntimeError(
                f"Expected 24 scenarios for "
                f"{challenge}; got {len(ids)}"
            )

        ids = rng.permutation(
            ids
        )

        for sid in ids[:12]:
            mapping[
                int(sid)
            ] = "train"

        for sid in ids[12:18]:
            mapping[
                int(sid)
            ] = "validation"

        for sid in ids[18:]:
            mapping[
                int(sid)
            ] = "test"

    return mapping


def select_threshold(
    y_val,
    scores,
):
    """
    Select threshold using validation only.

    Utility explicitly favors attack detection
    while penalizing legitimate-physical alarms.
    Tie-breakers prefer higher balanced accuracy,
    then lower physical FPR, then the lower threshold
    to avoid an artificially conservative small-sample
    threshold when several candidates are equivalent.
    """
    y_val = np.asarray(
        y_val,
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    candidates = np.unique(
        np.concatenate(
            [
                [
                    np.nextafter(
                        scores.min(),
                        -np.inf,
                    )
                ],
                np.quantile(
                    scores,
                    np.linspace(
                        0.01,
                        0.99,
                        150,
                    ),
                ),
                [
                    np.nextafter(
                        scores.max(),
                        np.inf,
                    )
                ],
            ]
        )
    )

    best_tuple = None
    best_threshold = None

    for threshold in candidates:
        pred = (
            scores >= threshold
        ).astype(int)

        m = metric_dict(
            y_val,
            pred,
        )

        utility = (
            m[
                "cyber_detection_rate"
            ]
            - 1.25
            * m[
                "physical_false_alarm_rate"
            ]
        )

        ranking = (
            utility,
            m[
                "balanced_accuracy"
            ],
            -m[
                "physical_false_alarm_rate"
            ],
            -float(
                threshold
            ),
        )

        if (
            best_tuple is None
            or ranking > best_tuple
        ):
            best_tuple = ranking
            best_threshold = float(
                threshold
            )

    return best_threshold


def run_repeat(
    event_df,
    repeat_id,
    seed,
):
    split_map = make_split(
        event_df,
        seed,
    )

    d = event_df.copy()

    d[
        "repeat_split"
    ] = (
        d[
            "scenario_id"
        ]
        .astype(int)
        .map(
            split_map
        )
    )

    train = d[
        d[
            "repeat_split"
        ] == "train"
    ]

    val = d[
        d[
            "repeat_split"
        ] == "validation"
    ]

    test = d[
        d[
            "repeat_split"
        ] == "test"
    ]

    rows = []
    class_rows = []

    for feature_set, cols in (
        FEATURE_SETS.items()
    ):
        missing = [
            c
            for c in cols
            if c not in d.columns
        ]

        if missing:
            raise KeyError(
                f"{feature_set} missing "
                f"{missing}"
            )

        for model_name, model in (
            model_zoo(seed).items()
        ):
            model.fit(
                train[
                    cols
                ],
                train[
                    "target_cyber"
                ],
            )

            val_scores = (
                model.predict_proba(
                    val[
                        cols
                    ]
                )[:, 1]
            )

            threshold = select_threshold(
                val[
                    "target_cyber"
                ].to_numpy(),
                val_scores,
            )

            test_scores = (
                model.predict_proba(
                    test[
                        cols
                    ]
                )[:, 1]
            )

            pred = (
                test_scores
                >= threshold
            ).astype(int)

            metrics = metric_dict(
                test[
                    "target_cyber"
                ],
                pred,
            )

            rows.append(
                {
                    "repeat_id": repeat_id,
                    "seed": seed,
                    "feature_set":
                        feature_set,
                    "model":
                        model_name,
                    "threshold":
                        threshold,
                    **metrics,
                }
            )

            tmp = test[
                [
                    "scenario_id",
                    "challenge_type",
                    "target_cyber",
                ]
            ].copy()

            tmp[
                "prediction"
            ] = pred

            tmp[
                "repeat_id"
            ] = repeat_id

            tmp[
                "feature_set"
            ] = feature_set

            tmp[
                "model"
            ] = model_name

            class_rate = (
                tmp.groupby(
                    [
                        "repeat_id",
                        "feature_set",
                        "model",
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

            class_rows.append(
                class_rate
            )

    return (
        pd.DataFrame(
            rows
        ),
        pd.concat(
            class_rows,
            ignore_index=True,
        ),
    )


def summarize(
    metrics,
):
    rows = []

    metric_cols = [
        "balanced_accuracy",
        "f1",
        "precision",
        "cyber_detection_rate",
        "physical_false_alarm_rate",
    ]

    for (
        feature_set,
        model,
    ), g in metrics.groupby(
        [
            "feature_set",
            "model",
        ]
    ):
        row = {
            "feature_set":
                feature_set,
            "model":
                model,
            "n_repeats":
                len(g),
        }

        for c in metric_cols:
            x = g[
                c
            ].to_numpy(
                dtype=float
            )

            row[
                f"{c}_mean"
            ] = float(
                np.mean(x)
            )

            row[
                f"{c}_median"
            ] = float(
                np.median(x)
            )

            row[
                f"{c}_p025"
            ] = float(
                np.percentile(
                    x,
                    2.5,
                )
            )

            row[
                f"{c}_p975"
            ] = float(
                np.percentile(
                    x,
                    97.5,
                )
            )

        perfect = (
            (
                g[
                    "cyber_detection_rate"
                ] == 1.0
            )
            & (
                g[
                    "physical_false_alarm_rate"
                ] == 0.0
            )
        )

        row[
            "perfect_repeat_fraction"
        ] = float(
            perfect.mean()
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def paired_comparison(
    metrics,
):
    rows = []

    for model in (
        metrics[
            "model"
        ].unique()
    ):
        d = metrics[
            metrics[
                "model"
            ] == model
        ]

        pivot = d.pivot(
            index="repeat_id",
            columns="feature_set",
            values=[
                "balanced_accuracy",
                "cyber_detection_rate",
                "physical_false_alarm_rate",
            ],
        )

        if (
            "wls_fusion"
            not in pivot[
                "balanced_accuracy"
            ].columns
            or
            "truth_proxy_fusion"
            not in pivot[
                "balanced_accuracy"
            ].columns
        ):
            continue

        delta_bacc = (
            pivot[
                "balanced_accuracy"
            ][
                "wls_fusion"
            ]
            - pivot[
                "balanced_accuracy"
            ][
                "truth_proxy_fusion"
            ]
        )

        delta_detect = (
            pivot[
                "cyber_detection_rate"
            ][
                "wls_fusion"
            ]
            - pivot[
                "cyber_detection_rate"
            ][
                "truth_proxy_fusion"
            ]
        )

        delta_fpr = (
            pivot[
                "physical_false_alarm_rate"
            ][
                "wls_fusion"
            ]
            - pivot[
                "physical_false_alarm_rate"
            ][
                "truth_proxy_fusion"
            ]
        )

        rows.append(
            {
                "model": model,

                "mean_delta_balanced_accuracy":
                    float(
                        delta_bacc.mean()
                    ),

                "median_delta_balanced_accuracy":
                    float(
                        delta_bacc.median()
                    ),

                "wls_bacc_better_fraction":
                    float(
                        (
                            delta_bacc > 0
                        ).mean()
                    ),

                "wls_bacc_equal_fraction":
                    float(
                        (
                            delta_bacc == 0
                        ).mean()
                    ),

                "mean_delta_detection":
                    float(
                        delta_detect.mean()
                    ),

                "mean_delta_physical_fpr":
                    float(
                        delta_fpr.mean()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def class_summary(
    class_rates,
):
    return (
        class_rates.groupby(
            [
                "feature_set",
                "model",
                "challenge_type",
            ],
            as_index=False,
        )
        .agg(
            mean_alarm_rate=(
                "alarm_rate",
                "mean",
            ),
            median_alarm_rate=(
                "alarm_rate",
                "median",
            ),
            p025_alarm_rate=(
                "alarm_rate",
                lambda x: float(
                    np.percentile(
                        x,
                        2.5,
                    )
                ),
            ),
            p975_alarm_rate=(
                "alarm_rate",
                lambda x: float(
                    np.percentile(
                        x,
                        97.5,
                    )
                ),
            ),
        )
    )


def plot_balanced_accuracy(
    metrics,
):
    order = [
        "residual_temporal",
        "truth_proxy_fusion",
        "wls_fusion",
    ]

    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )

    positions = []
    labels = []
    data = []

    pos = 0

    for feature_set in order:
        for model in [
            "LogisticRegression",
            "RandomForest",
        ]:
            g = metrics[
                (
                    metrics[
                        "feature_set"
                    ] == feature_set
                )
                & (
                    metrics[
                        "model"
                    ] == model
                )
            ]

            positions.append(
                pos
            )

            labels.append(
                feature_set
                + "\n"
                + model
            )

            data.append(
                g[
                    "balanced_accuracy"
                ].to_numpy()
            )

            pos += 1

    ax.boxplot(
        data,
        positions=positions,
        showfliers=False,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
        fontsize=8.5,
    )

    ax.set_ylim(
        0.40,
        1.02,
    )

    ax.set_ylabel(
        "Balanced accuracy"
    )

    ax.set_title(
        "100 Repeated Group-Safe Splits — Fusion Robustness"
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
        / "50_repeated_split_balanced_accuracy.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_detection_vs_false_alarm(
    summary,
):
    fig, ax = plt.subplots(
        figsize=(9.5, 6.5)
    )

    for _, row in (
        summary.iterrows()
    ):
        ax.scatter(
            100
            * row[
                "physical_false_alarm_rate_mean"
            ],
            100
            * row[
                "cyber_detection_rate_mean"
            ],
            s=90,
        )

        ax.annotate(
            row[
                "feature_set"
            ]
            + "\n"
            + row[
                "model"
            ],
            (
                100
                * row[
                    "physical_false_alarm_rate_mean"
                ],
                100
                * row[
                    "cyber_detection_rate_mean"
                ],
            ),
            xytext=(
                6,
                5,
            ),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(
        "Mean physical false-alarm rate (%)"
    )

    ax.set_ylabel(
        "Mean cyber detection rate (%)"
    )

    ax.set_xlim(
        left=-1
    )

    ax.set_ylim(
        0,
        102,
    )

    ax.set_title(
        "Repeated-Split Detection / False-Alarm Trade-Off"
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
        / "51_repeated_split_detection_vs_false_alarm.png",
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            str(
                FEATURE_PATH
            )
        )

    print(
        "\n=== PHASE 2P: REPEATED-SPLIT WLS FUSION ROBUSTNESS ==="
    )

    df = pd.read_csv(
        FEATURE_PATH
    )

    event_df = df[
        df[
            "window_mode"
        ] == "event"
    ].copy()

    # Rebuild target explicitly for auditability.
    event_df[
        "target_cyber"
    ] = (
        event_df[
            "challenge_type"
        ]
        == POSITIVE_CLASS
    ).astype(int)

    print(
        f"Event scenarios : "
        f"{event_df['scenario_id'].nunique()}"
    )

    print(
        f"Repeats         : "
        f"{N_REPEATS}"
    )

    all_metrics = []
    all_class_rates = []

    for repeat_id in range(
        N_REPEATS
    ):
        seed = (
            BASE_SEED
            + repeat_id
        )

        metrics, class_rates = (
            run_repeat(
                event_df,
                repeat_id,
                seed,
            )
        )

        all_metrics.append(
            metrics
        )

        all_class_rates.append(
            class_rates
        )

        if (
            repeat_id + 1
        ) % 10 == 0:
            print(
                f"  completed "
                f"{repeat_id + 1}"
                f"/{N_REPEATS}"
            )

    metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    class_rates = pd.concat(
        all_class_rates,
        ignore_index=True,
    )

    summary = summarize(
        metrics
    )

    paired = paired_comparison(
        metrics
    )

    class_avg = class_summary(
        class_rates
    )

    metrics.to_csv(
        RESULTS
        / "phase2p_repeated_split_metrics.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS
        / "phase2p_repeated_split_summary.csv",
        index=False,
    )

    paired.to_csv(
        RESULTS
        / "phase2p_wls_vs_truth_paired_comparison.csv",
        index=False,
    )

    class_avg.to_csv(
        RESULTS
        / "phase2p_alarm_by_challenge_summary.csv",
        index=False,
    )

    plot_balanced_accuracy(
        metrics
    )

    plot_detection_vs_false_alarm(
        summary
    )

    print(
        "\n=== 100-REPEAT SUMMARY ==="
    )

    print(
        summary[
            [
                "feature_set",
                "model",
                "balanced_accuracy_mean",
                "balanced_accuracy_p025",
                "balanced_accuracy_p975",
                "cyber_detection_rate_mean",
                "physical_false_alarm_rate_mean",
                "perfect_repeat_fraction",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n=== PAIRED WLS vs TRUTH-PROXY COMPARISON ==="
    )

    print(
        paired.to_string(
            index=False
        )
    )

    print(
        "\n=== MEAN ALARM RATE BY CHALLENGE TYPE ==="
    )

    print(
        class_avg.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "  results/phase2p_repeated_split_metrics.csv"
    )

    print(
        "  results/phase2p_repeated_split_summary.csv"
    )

    print(
        "  results/phase2p_wls_vs_truth_paired_comparison.csv"
    )

    print(
        "  results/phase2p_alarm_by_challenge_summary.csv"
    )

    print(
        "  figures/50_repeated_split_balanced_accuracy.png"
    )

    print(
        "  figures/51_repeated_split_detection_vs_false_alarm.png"
    )

    print(
        "\nINTERPRETATION RULE:\n"
        "A single 6-scenario test split can easily produce 100% or 83.3% "
        "by one-scenario differences. This repeated-split analysis is therefore "
        "the required robustness check before claiming that AC-WLS network "
        "context materially improves the detector. The decisive evidence is "
        "not one perfect split, but the distribution over 100 paired group-safe "
        "splits: mean/interval detection, physical false alarms, and how often "
        "WLS fusion equals or outperforms the truth-proxy baseline. If WLS "
        "fusion remains consistently strong, the next phase can justifiably "
        "focus on topology-aware localization and stealth/coherent FDIAs."
    )


if __name__ == "__main__":
    main()
