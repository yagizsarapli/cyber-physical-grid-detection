from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
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
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


# ============================================================
# PHASE 2H — LEAKAGE-SAFE TEMPORAL BASELINE BENCHMARK
# ============================================================
#
# Inputs
# ------
# phase2g_v2_transient_timeseries.csv.gz
# phase2g_v2_scenario_metadata.csv
#
# Goals
# -----
# 1) Convert 5-ms transient trajectories into one temporal feature vector
#    per matched window.
# 2) Respect the existing scenario-group-safe train/validation/test split.
# 3) Never train on:
#       - scenario_id
#       - window_mode
#       - event_type / event_family
#       - dataset_split
#       - attack severity parameters
#       - event duration/start time
#       - BESS SOC
# 4) Compare two feature sets:
#
#    INVERTER_ONLY
#    -------------
#    Signals available from inverter/controller telemetry and the GFL's
#    own measured voltage/angle. No independent "true" voltage channel,
#    no direct sensor residual, no direct attacked command value.
#
#    PHYSICS_PROXY
#    -------------
#    Adds simulation/digital-twin physical-state proxies and direct
#    true-vs-measured residuals. This is intentionally optimistic and is
#    treated as an upper-bound baseline, NOT the final detector.
#
# 5) Calibrate binary decision thresholds on VALIDATION normal windows
#    to approximately 5% false-positive rate, then report TEST results.
# 6) Also test:
#       - 4-way family classification:
#           normal / physical / cyber / hybrid
#       - 9-way event classification:
#           normal + 8 event types
#
# Counterfactual pairs are used only for evaluation (pair-ranking score).
# Event-minus-normal differences are NEVER used as model inputs.
# ============================================================


RAW_PATH = DATA / "phase2g_v2_transient_timeseries.csv.gz"
META_PATH = DATA / "phase2g_v2_scenario_metadata.csv"

PRE_WINDOW_S = 0.40
PRE_GUARD_S = 0.05
POST_WINDOW_S = 0.35

TARGET_FPR = 0.05
RANDOM_STATE = 20260812


INVERTER_SIGNALS = [
    "v_gfl_measured_pu",
    "theta_gfl_measured_deg",
    "gfl_pll_freq_dev_hz",
    "gfl_id_pu",
    "gfl_iq_pu",
    "gfl_current_pu",
    "gfm_freq_dev_hz",
    "gfm_p_pu",
    "gfm_q_pu",
    "gfm_current_pu",
]

PHYSICS_PROXY_EXTRA_SIGNALS = [
    "min_bus_v_pu",
    "max_line_loading_percent",
    "sensor_v_residual_pu",
    "sensor_angle_residual_deg",
]

BOOLEAN_SIGNALS = [
    "gfl_limiter",
    "gfm_limiter",
]

ANGLE_SIGNALS = {
    "theta_gfl_measured_deg",
    "sensor_angle_residual_deg",
}


def wrap_deg(x):
    return (x + 180.0) % 360.0 - 180.0


def safe_slope(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    if len(t) < 3:
        return 0.0

    t0 = t - t.mean()
    denom = np.sum(t0 * t0)

    if denom <= 1e-15:
        return 0.0

    return float(
        np.sum(t0 * (y - y.mean()))
        / denom
    )


def segment_stats(
    signal_name,
    pre,
    event,
    post,
):
    """
    Features are defined relative to the window's own pre-event baseline.
    This is deployable in principle: no matched counterfactual is needed.
    """
    pre_x = pre[signal_name].to_numpy(dtype=float)
    event_x = event[signal_name].to_numpy(dtype=float)
    post_x = post[signal_name].to_numpy(dtype=float)

    if len(pre_x) == 0 or len(event_x) == 0:
        return {}

    pre_mean = float(np.nanmean(pre_x))
    pre_std = float(np.nanstd(pre_x))

    if signal_name in ANGLE_SIGNALS:
        event_dev = wrap_deg(event_x - pre_mean)
        post_dev = wrap_deg(post_x - pre_mean)
    else:
        event_dev = event_x - pre_mean
        post_dev = post_x - pre_mean

    abs_event_dev = np.abs(event_dev)

    out = {
        f"{signal_name}__pre_mean": pre_mean,
        f"{signal_name}__pre_std": pre_std,

        f"{signal_name}__event_mean_delta": float(
            np.nanmean(event_dev)
        ),
        f"{signal_name}__event_std": float(
            np.nanstd(event_x)
        ),
        f"{signal_name}__event_abs_peak_delta": float(
            np.nanmax(abs_event_dev)
        ),
        f"{signal_name}__event_rms_delta": float(
            np.sqrt(np.nanmean(event_dev ** 2))
        ),
        f"{signal_name}__event_p95_abs_delta": float(
            np.nanpercentile(abs_event_dev, 95)
        ),
        f"{signal_name}__event_range": float(
            np.nanmax(event_x) - np.nanmin(event_x)
        ),
        f"{signal_name}__event_slope": safe_slope(
            event["time_s"].to_numpy(dtype=float),
            event_dev,
        ),
    }

    if len(post_x) > 0:
        out[
            f"{signal_name}__post_rms_delta"
        ] = float(
            np.sqrt(
                np.nanmean(post_dev ** 2)
            )
        )
        out[
            f"{signal_name}__post_mean_abs_delta"
        ] = float(
            np.nanmean(
                np.abs(post_dev)
            )
        )
    else:
        out[
            f"{signal_name}__post_rms_delta"
        ] = np.nan
        out[
            f"{signal_name}__post_mean_abs_delta"
        ] = np.nan

    return out


def extract_window_features(
    group,
    meta_row,
):
    t_start = float(
        meta_row["event_start_s"]
    )
    t_end = float(
        meta_row["event_end_s"]
    )

    pre_start = max(
        0.0,
        t_start - PRE_WINDOW_S
    )
    pre_end = max(
        pre_start,
        t_start - PRE_GUARD_S
    )

    post_end = min(
        float(group["time_s"].max()),
        t_end + POST_WINDOW_S
    )

    pre = group[
        (group["time_s"] >= pre_start)
        & (group["time_s"] < pre_end)
    ]

    event = group[
        (group["time_s"] >= t_start)
        & (group["time_s"] < t_end)
    ]

    post = group[
        (group["time_s"] >= t_end)
        & (group["time_s"] <= post_end)
    ]

    if len(pre) < 10:
        raise RuntimeError(
            f"Insufficient pre-event samples for scenario "
            f"{int(meta_row['scenario_id'])}"
        )

    if len(event) < 10:
        raise RuntimeError(
            f"Insufficient event samples for scenario "
            f"{int(meta_row['scenario_id'])}"
        )

    row = {
        "scenario_id": int(
            meta_row["scenario_id"]
        ),
        "window_mode": str(
            group.iloc[0]["window_mode"]
        ),
        "event_type": str(
            meta_row["event_type"]
        ),
        "event_family": str(
            meta_row["event_family"]
        ),
        "dataset_split": str(
            meta_row["dataset_split"]
        ),

        # Metadata retained ONLY for reporting/audit.
        # SOC is intentionally not used as a model feature.
        "context_pv_p_mw": float(
            meta_row["context_pv_p_mw"]
        ),
        "context_load_total_p_mw": float(
            meta_row["context_load_total_p_mw"]
        ),
        "context_bess_soc": float(
            meta_row["context_bess_soc"]
        ),
    }

    all_signals = (
        INVERTER_SIGNALS
        + PHYSICS_PROXY_EXTRA_SIGNALS
    )

    for signal in all_signals:
        if signal not in group.columns:
            raise KeyError(
                f"Missing raw signal: {signal}"
            )

        row.update(
            segment_stats(
                signal,
                pre,
                event,
                post,
            )
        )

    for signal in BOOLEAN_SIGNALS:
        if signal not in group.columns:
            raise KeyError(
                f"Missing boolean signal: {signal}"
            )

        row[
            f"{signal}__pre_fraction"
        ] = float(
            pre[signal].astype(float).mean()
        )

        row[
            f"{signal}__event_fraction"
        ] = float(
            event[signal].astype(float).mean()
        )

        row[
            f"{signal}__post_fraction"
        ] = float(
            post[signal].astype(float).mean()
            if len(post) else 0.0
        )

    # Labels.
    is_event = (
        row["window_mode"] == "event"
    )

    row["target_binary"] = int(
        is_event
    )

    if not is_event:
        row["target_family"] = "normal"
        row["target_event"] = "normal"
    else:
        row["target_family"] = row[
            "event_family"
        ]
        row["target_event"] = row[
            "event_type"
        ]

    return row


def build_feature_dataset(raw, meta):
    meta_idx = meta.set_index(
        "scenario_id"
    )

    rows = []

    grouped = raw.groupby(
        [
            "scenario_id",
            "window_mode",
        ],
        sort=True,
    )

    print(
        f"Extracting temporal features from "
        f"{len(grouped)} windows ..."
    )

    for (
        scenario_id,
        window_mode,
    ), group in grouped:
        scenario_id = int(scenario_id)

        if scenario_id not in meta_idx.index:
            raise KeyError(
                f"scenario_id {scenario_id} "
                f"missing from metadata"
            )

        m = meta_idx.loc[scenario_id].copy()

        # scenario_id became the DataFrame index after set_index(),
        # so explicitly restore it as a field for downstream code.
        m["scenario_id"] = scenario_id

        group = group.sort_values(
            "time_s"
        )

        rows.append(
            extract_window_features(
                group,
                m,
            )
        )

    return pd.DataFrame(rows)


def get_feature_columns(
    feature_df,
    feature_set,
):
    if feature_set == "inverter_only":
        allowed_prefixes = tuple(
            s + "__"
            for s in INVERTER_SIGNALS
        ) + tuple(
            s + "__"
            for s in BOOLEAN_SIGNALS
        )

    elif feature_set == "physics_proxy":
        allowed_prefixes = tuple(
            s + "__"
            for s in (
                INVERTER_SIGNALS
                + PHYSICS_PROXY_EXTRA_SIGNALS
                + BOOLEAN_SIGNALS
            )
        )

    else:
        raise ValueError(feature_set)

    cols = [
        c
        for c in feature_df.columns
        if c.startswith(
            allowed_prefixes
        )
    ]

    # Explicitly exclude any direct command/attack metadata.
    forbidden_tokens = (
        "pref",
        "imax",
        "severity",
        "duration",
        "start_s",
        "end_s",
        "context_bess_soc",
    )

    cols = [
        c for c in cols
        if not any(
            token in c
            for token in forbidden_tokens
        )
    ]

    return sorted(cols)


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
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=2,
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
                        learning_rate=0.06,
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=0.5,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def split_xy(
    df,
    feature_cols,
    target_col,
):
    out = {}

    for split in (
        "train",
        "validation",
        "test",
    ):
        d = df[
            df["dataset_split"] == split
        ].copy()

        out[split] = (
            d[feature_cols],
            d[target_col],
            d,
        )

    return out


def get_positive_scores(
    model,
    X,
):
    if hasattr(
        model,
        "predict_proba",
    ):
        proba = model.predict_proba(X)
        classes = list(model.classes_)

        pos_idx = classes.index(1)
        return proba[:, pos_idx]

    if hasattr(
        model,
        "decision_function",
    ):
        score = model.decision_function(X)
        return 1.0 / (
            1.0 + np.exp(-score)
        )

    raise RuntimeError(
        "Model has no probability/decision score."
    )


def threshold_at_target_fpr(
    y_val,
    score_val,
    target_fpr=TARGET_FPR,
):
    y_val = np.asarray(y_val)
    score_val = np.asarray(score_val)

    normal_scores = score_val[
        y_val == 0
    ]

    if len(normal_scores) == 0:
        return 0.5

    # Strictly above the empirical 95th percentile is approximately
    # a 5% validation false-positive-rate operating point.
    return float(
        np.quantile(
            normal_scores,
            1.0 - target_fpr,
            method="higher",
        )
    )


def pair_ranking_accuracy(
    test_df,
    scores,
):
    d = test_df[
        [
            "scenario_id",
            "window_mode",
        ]
    ].copy()

    d["score"] = scores

    hits = []

    for sid, g in d.groupby(
        "scenario_id"
    ):
        normal = g[
            g["window_mode"]
            == "normal_counterfactual"
        ]
        event = g[
            g["window_mode"]
            == "event"
        ]

        if normal.empty or event.empty:
            continue

        hits.append(
            float(
                event.iloc[0]["score"]
                > normal.iloc[0]["score"]
            )
        )

    return (
        float(np.mean(hits))
        if hits
        else np.nan
    )


def binary_metrics(
    y_true,
    scores,
    threshold,
):
    pred = (
        scores >= threshold
    ).astype(int)

    normal_mask = (
        np.asarray(y_true) == 0
    )

    if normal_mask.any():
        fpr = float(
            np.mean(
                pred[normal_mask] == 1
            )
        )
    else:
        fpr = np.nan

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                scores
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                pred
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                pred
            )
        ),
        "detection_rate": float(
            recall_score(
                y_true,
                pred
            )
        ),
        "false_positive_rate": fpr,
    }


def bootstrap_binary_ci(
    y_true,
    scores,
    threshold,
    n_boot=500,
):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    rng = np.random.default_rng(
        RANDOM_STATE + 123
    )

    f1s = []
    bals = []

    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yb = y_true[idx]
        sb = scores[idx]

        if len(np.unique(yb)) < 2:
            continue

        pred = (
            sb >= threshold
        ).astype(int)

        f1s.append(
            f1_score(
                yb,
                pred
            )
        )

        bals.append(
            balanced_accuracy_score(
                yb,
                pred
            )
        )

    def ci(x):
        if not x:
            return (
                np.nan,
                np.nan,
            )

        return (
            float(
                np.percentile(
                    x,
                    2.5
                )
            ),
            float(
                np.percentile(
                    x,
                    97.5
                )
            ),
        )

    f1_lo, f1_hi = ci(f1s)
    bal_lo, bal_hi = ci(bals)

    return {
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
        "balanced_accuracy_ci_low": bal_lo,
        "balanced_accuracy_ci_high": bal_hi,
    }


def run_binary_benchmark(
    feature_df,
    feature_set,
):
    feature_cols = get_feature_columns(
        feature_df,
        feature_set,
    )

    split = split_xy(
        feature_df,
        feature_cols,
        "target_binary",
    )

    X_train, y_train, train_df = split[
        "train"
    ]
    X_val, y_val, val_df = split[
        "validation"
    ]
    X_test, y_test, test_df = split[
        "test"
    ]

    rows = []
    curves = {}

    for name, model in model_zoo().items():
        model.fit(
            X_train,
            y_train,
        )

        val_scores = get_positive_scores(
            model,
            X_val,
        )

        threshold = threshold_at_target_fpr(
            y_val,
            val_scores,
            TARGET_FPR,
        )

        test_scores = get_positive_scores(
            model,
            X_test,
        )

        metrics = binary_metrics(
            y_test,
            test_scores,
            threshold,
        )

        ci = bootstrap_binary_ci(
            y_test,
            test_scores,
            threshold,
        )

        pair_acc = pair_ranking_accuracy(
            test_df,
            test_scores,
        )

        row = {
            "feature_set": feature_set,
            "model": name,
            "n_features": len(
                feature_cols
            ),
            "validation_threshold": threshold,
            "pair_ranking_accuracy": pair_acc,
            **metrics,
            **ci,
        }

        rows.append(row)

        fpr, tpr, _ = roc_curve(
            y_test,
            test_scores,
        )

        curves[name] = (
            fpr,
            tpr,
            metrics["roc_auc"],
        )

    return (
        pd.DataFrame(rows),
        curves,
        feature_cols,
    )


def multiclass_benchmark(
    feature_df,
    feature_set,
    target_col,
):
    feature_cols = get_feature_columns(
        feature_df,
        feature_set,
    )

    split = split_xy(
        feature_df,
        feature_cols,
        target_col,
    )

    X_train, y_train, _ = split[
        "train"
    ]
    X_val, y_val, _ = split[
        "validation"
    ]
    X_test, y_test, test_df = split[
        "test"
    ]

    rows = []
    fitted = {}

    for name, model in model_zoo().items():
        model.fit(
            X_train,
            y_train,
        )

        val_pred = model.predict(
            X_val
        )
        test_pred = model.predict(
            X_test
        )

        row = {
            "feature_set": feature_set,
            "target": target_col,
            "model": name,
            "validation_macro_f1": float(
                f1_score(
                    y_val,
                    val_pred,
                    average="macro",
                )
            ),
            "test_macro_f1": float(
                f1_score(
                    y_test,
                    test_pred,
                    average="macro",
                )
            ),
            "test_balanced_accuracy": float(
                balanced_accuracy_score(
                    y_test,
                    test_pred,
                )
            ),
        }

        rows.append(row)
        fitted[name] = (
            model,
            test_pred,
            test_df,
            y_test,
        )

    result = pd.DataFrame(rows)

    best_name = str(
        result.sort_values(
            "validation_macro_f1",
            ascending=False,
        ).iloc[0]["model"]
    )

    return (
        result,
        fitted[best_name],
        best_name,
        feature_cols,
    )


def plot_binary_roc(
    all_curves,
):
    fig, ax = plt.subplots(
        figsize=(9.5, 6.6)
    )

    for (
        feature_set,
        model_name,
    ), (
        fpr,
        tpr,
        auc,
    ) in all_curves.items():
        label = (
            f"{feature_set} / "
            f"{model_name} "
            f"(AUC={auc:.3f})"
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=1.8,
            label=label,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(
        "False-positive rate"
    )
    ax.set_ylabel(
        "True-positive rate"
    )
    ax.set_title(
        "Binary Event Detection — Leakage-Safe Test ROC"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "28_binary_detection_roc.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_detection_operating_point(
    binary_results,
):
    fig, ax = plt.subplots(
        figsize=(10.5, 6.2)
    )

    d = binary_results.copy()

    labels = (
        d["feature_set"]
        + "\n"
        + d["model"]
    ).tolist()

    x = np.arange(
        len(d)
    )

    detect = (
        100
        * d["detection_rate"]
        .to_numpy()
    )
    fpr = (
        100
        * d[
            "false_positive_rate"
        ].to_numpy()
    )

    width = 0.36

    ax.bar(
        x - width/2,
        detect,
        width,
        label="Detection rate",
    )

    ax.bar(
        x + width/2,
        fpr,
        width,
        label="False-positive rate",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Test rate (%)"
    )
    ax.set_ylim(
        0,
        105
    )
    ax.set_title(
        "Binary Detection at Validation-Calibrated ~5% False-Alarm Operating Point"
    )
    ax.legend(
        frameon=False
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "29_binary_detection_at_5pct_fpr.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_confusion(
    y_true,
    y_pred,
    title,
    filename,
):
    labels = sorted(
        pd.unique(
            pd.concat(
                [
                    pd.Series(y_true),
                    pd.Series(y_pred),
                ],
                ignore_index=True,
            )
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true",
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )

    disp.plot(
        ax=ax,
        cmap=None,
        values_format=".2f",
        colorbar=False,
    )

    ax.set_title(title)
    plt.xticks(
        rotation=35,
        ha="right",
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES / filename,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def random_forest_importance(
    feature_df,
    feature_set,
):
    feature_cols = get_feature_columns(
        feature_df,
        feature_set,
    )

    train = feature_df[
        feature_df["dataset_split"]
        == "train"
    ]

    test = feature_df[
        feature_df["dataset_split"]
        == "test"
    ]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        train[feature_cols]
    )
    X_test = imputer.transform(
        test[feature_cols]
    )

    y_train = train[
        "target_binary"
    ].to_numpy()

    y_test = test[
        "target_binary"
    ].to_numpy()

    rf = RandomForestClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf.fit(
        X_train,
        y_train,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": (
                rf.feature_importances_
            ),
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    return importance


def plot_feature_importance(
    importance,
):
    d = importance.head(
        20
    ).sort_values(
        "importance",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        d["feature"],
        d["importance"],
    )

    ax.set_xlabel(
        "Random-forest impurity importance"
    )
    ax.set_title(
        "Top Temporal Features — Diagnostic Baseline Only"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "32_random_forest_top_features.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            "Missing raw transient dataset: "
            + str(RAW_PATH)
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            "Missing scenario metadata: "
            + str(META_PATH)
        )

    print(
        "\n=== PHASE 2H: LEAKAGE-SAFE TEMPORAL BASELINES ==="
    )

    print(
        "Loading compressed transient dataset ..."
    )

    raw = pd.read_csv(
        RAW_PATH
    )

    meta = pd.read_csv(
        META_PATH,
        parse_dates=[
            "context_timestamp"
        ],
    )

    print(
        f"Raw samples      : {len(raw):,}"
    )
    print(
        f"Scenarios        : "
        f"{meta['scenario_id'].nunique()}"
    )

    features = build_feature_dataset(
        raw,
        meta,
    )

    features.to_csv(
        DATA
        / "phase2h_temporal_feature_dataset.csv",
        index=False,
    )

    print(
        f"Feature windows  : {len(features)}"
    )
    print(
        f"Unique scenarios : {features['scenario_id'].nunique()}"
    )
    print(
        "Expected windows : "
        f"{2*meta['scenario_id'].nunique()}"
    )

    # --------------------------------------------
    # Binary detection benchmark
    # --------------------------------------------
    binary_frames = []
    all_curves = {}

    binary_feature_sets = [
        "inverter_only",
        "physics_proxy",
    ]

    for feature_set in binary_feature_sets:
        print(
            f"\nBinary benchmark: {feature_set}"
        )

        result, curves, feature_cols = (
            run_binary_benchmark(
                features,
                feature_set,
            )
        )

        binary_frames.append(
            result
        )

        for model_name, curve in curves.items():
            all_curves[
                (
                    feature_set,
                    model_name,
                )
            ] = curve

        print(
            result[
                [
                    "model",
                    "n_features",
                    "roc_auc",
                    "average_precision",
                    "detection_rate",
                    "false_positive_rate",
                    "balanced_accuracy",
                    "f1",
                    "pair_ranking_accuracy",
                ]
            ].to_string(
                index=False
            )
        )

    binary_results = pd.concat(
        binary_frames,
        ignore_index=True,
    )

    binary_results.to_csv(
        RESULTS
        / "phase2h_binary_baseline_metrics.csv",
        index=False,
    )

    plot_binary_roc(
        all_curves
    )

    plot_detection_operating_point(
        binary_results
    )

    # --------------------------------------------
    # Family classification
    # --------------------------------------------
    family_frames = []
    family_best_objects = {}

    for feature_set in binary_feature_sets:
        result, best, best_name, _ = (
            multiclass_benchmark(
                features,
                feature_set,
                "target_family",
            )
        )

        family_frames.append(
            result
        )
        family_best_objects[
            feature_set
        ] = (
            best,
            best_name,
        )

    family_results = pd.concat(
        family_frames,
        ignore_index=True,
    )

    family_results.to_csv(
        RESULTS
        / "phase2h_family_classification_metrics.csv",
        index=False,
    )

    # Choose best family model by validation macro-F1.
    best_family_row = family_results.sort_values(
        "validation_macro_f1",
        ascending=False,
    ).iloc[0]

    best_family_fs = str(
        best_family_row[
            "feature_set"
        ]
    )

    (
        family_best,
        family_best_name,
    ) = family_best_objects[
        best_family_fs
    ]

    (
        family_model,
        family_pred,
        family_test_df,
        family_y_test,
    ) = family_best

    plot_confusion(
        family_y_test,
        family_pred,
        (
            "4-Way Event-Family Classification — "
            f"{best_family_fs} / {family_best_name}"
        ),
        "30_family_confusion_best_model.png",
    )

    # --------------------------------------------
    # Fine event-type classification
    # --------------------------------------------
    event_frames = []
    event_best_objects = {}

    for feature_set in binary_feature_sets:
        result, best, best_name, _ = (
            multiclass_benchmark(
                features,
                feature_set,
                "target_event",
            )
        )

        event_frames.append(
            result
        )
        event_best_objects[
            feature_set
        ] = (
            best,
            best_name,
        )

    event_results = pd.concat(
        event_frames,
        ignore_index=True,
    )

    event_results.to_csv(
        RESULTS
        / "phase2h_eventtype_classification_metrics.csv",
        index=False,
    )

    best_event_row = event_results.sort_values(
        "validation_macro_f1",
        ascending=False,
    ).iloc[0]

    best_event_fs = str(
        best_event_row[
            "feature_set"
        ]
    )

    (
        event_best,
        event_best_name,
    ) = event_best_objects[
        best_event_fs
    ]

    (
        event_model,
        event_pred,
        event_test_df,
        event_y_test,
    ) = event_best

    plot_confusion(
        event_y_test,
        event_pred,
        (
            "9-Way Event-Type Classification — "
            f"{best_event_fs} / {event_best_name}"
        ),
        "31_eventtype_confusion_best_model.png",
    )

    # --------------------------------------------
    # Diagnostic feature importance.
    # Use inverter-only features so the ranking
    # is not dominated by oracle residuals.
    # --------------------------------------------
    importance = random_forest_importance(
        features,
        "inverter_only",
    )

    importance.to_csv(
        RESULTS
        / "phase2h_inverter_only_rf_feature_importance.csv",
        index=False,
    )

    plot_feature_importance(
        importance
    )

    print(
        "\n=== FAMILY CLASSIFICATION ==="
    )
    print(
        family_results.to_string(
            index=False
        )
    )

    print(
        "\n=== EVENT-TYPE CLASSIFICATION ==="
    )
    print(
        event_results.to_string(
            index=False
        )
    )

    print(
        "\nTop inverter-only diagnostic features:"
    )
    print(
        importance.head(15).to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )
    print(
        "  data/phase2h_temporal_feature_dataset.csv"
    )
    print(
        "  results/phase2h_binary_baseline_metrics.csv"
    )
    print(
        "  results/phase2h_family_classification_metrics.csv"
    )
    print(
        "  results/phase2h_eventtype_classification_metrics.csv"
    )
    print(
        "  results/phase2h_inverter_only_rf_feature_importance.csv"
    )
    print(
        "  figures/28_binary_detection_roc.png"
    )
    print(
        "  figures/29_binary_detection_at_5pct_fpr.png"
    )
    print(
        "  figures/30_family_confusion_best_model.png"
    )
    print(
        "  figures/31_eventtype_confusion_best_model.png"
    )
    print(
        "  figures/32_random_forest_top_features.png"
    )

    print(
        "\nINTERPRETATION POLICY:\n"
        "Do NOT celebrate a very high Physics-Proxy score as the final result. "
        "That feature set contains simulation/digital-twin truth residuals and "
        "is an optimistic upper bound. The scientifically important baseline "
        "is Inverter-Only. The next phase should add WLS-estimated network "
        "states and graph/node/edge telemetry, then test whether a physics- "
        "and topology-aware model improves over this leakage-safe baseline."
    )


if __name__ == "__main__":
    main()
