from pathlib import Path
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
    roc_curve,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RAW_PATH = DATA / "phase2g_v2_transient_timeseries.csv.gz"
META_PATH = DATA / "phase2g_v2_scenario_metadata.csv"

RANDOM_STATE = 20260812
TARGET_FPR = 0.05

# Streaming design:
# every 50 ms, detector sees only the most recent 200 ms
# plus the immediately preceding 200 ms as a causal reference window.
CURRENT_WINDOW_S = 0.20
REFERENCE_WINDOW_S = 0.20
STRIDE_S = 0.05

# Exclude the first few milliseconds after onset from "positive" scoring:
# the detector should be allowed a finite response time.
MIN_EVENT_CONTENT_S = 0.05

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


def causal_signal_features(
    signal,
    ref,
    cur,
):
    ref_x = ref[signal].to_numpy(dtype=float)
    cur_x = cur[signal].to_numpy(dtype=float)

    if len(ref_x) < 3 or len(cur_x) < 3:
        return {}

    ref_mean = float(np.nanmean(ref_x))
    ref_std = float(np.nanstd(ref_x))

    if signal in ANGLE_SIGNALS:
        cur_delta = wrap_deg(cur_x - ref_mean)
    else:
        cur_delta = cur_x - ref_mean

    abs_delta = np.abs(cur_delta)

    return {
        f"{signal}__ref_mean": ref_mean,
        f"{signal}__ref_std": ref_std,
        f"{signal}__cur_mean_delta": float(
            np.nanmean(cur_delta)
        ),
        f"{signal}__cur_std": float(
            np.nanstd(cur_x)
        ),
        f"{signal}__cur_abs_peak_delta": float(
            np.nanmax(abs_delta)
        ),
        f"{signal}__cur_rms_delta": float(
            np.sqrt(np.nanmean(cur_delta ** 2))
        ),
        f"{signal}__cur_p95_abs_delta": float(
            np.nanpercentile(abs_delta, 95)
        ),
        f"{signal}__cur_range": float(
            np.nanmax(cur_x) - np.nanmin(cur_x)
        ),
        f"{signal}__cur_slope": safe_slope(
            cur["time_s"].to_numpy(dtype=float),
            cur_delta,
        ),
        f"{signal}__cur_last_delta": float(
            cur_delta[-1]
        ),
    }


def phase_label(
    window_start,
    window_end,
    event_start,
    event_end,
    is_event_trajectory,
):
    if not is_event_trajectory:
        return "normal"

    overlap = max(
        0.0,
        min(window_end, event_end)
        - max(window_start, event_start)
    )

    if overlap >= MIN_EVENT_CONTENT_S:
        return "event"

    if window_end <= event_start:
        return "pre_event"

    if window_start >= event_end:
        return "recovery"

    return "transition"


def build_streaming_feature_dataset(
    raw,
    meta,
):
    meta_idx = meta.set_index("scenario_id")
    rows = []

    grouped = raw.groupby(
        ["scenario_id", "window_mode"],
        sort=True,
    )

    print(
        f"Building causal sliding-window features "
        f"from {len(grouped)} trajectories ..."
    )

    for (
        scenario_id,
        window_mode,
    ), group in grouped:
        scenario_id = int(scenario_id)

        m = meta_idx.loc[scenario_id]
        g = group.sort_values("time_s")

        event_start = float(m["event_start_s"])
        event_end = float(m["event_end_s"])

        t_min = float(g["time_s"].min())
        t_max = float(g["time_s"].max())

        first_end = (
            t_min
            + REFERENCE_WINDOW_S
            + CURRENT_WINDOW_S
        )

        ends = np.arange(
            first_end,
            t_max + 1e-9,
            STRIDE_S,
        )

        is_event_trajectory = (
            window_mode == "event"
        )

        for window_end in ends:
            window_start = (
                window_end - CURRENT_WINDOW_S
            )

            ref_end = window_start
            ref_start = (
                ref_end - REFERENCE_WINDOW_S
            )

            ref = g[
                (g["time_s"] >= ref_start)
                & (g["time_s"] < ref_end)
            ]

            cur = g[
                (g["time_s"] >= window_start)
                & (g["time_s"] <= window_end)
            ]

            if len(ref) < 10 or len(cur) < 10:
                continue

            phase = phase_label(
                window_start,
                window_end,
                event_start,
                event_end,
                is_event_trajectory,
            )

            row = {
                "scenario_id": scenario_id,
                "window_mode": window_mode,
                "event_type": str(m["event_type"]),
                "event_family": str(m["event_family"]),
                "iid_split": str(m["dataset_split"]),
                "window_start_s": float(window_start),
                "window_end_s": float(window_end),
                "phase": phase,

                # audit only — never model inputs
                "context_pv_p_mw": float(
                    m["context_pv_p_mw"]
                ),
                "context_load_total_p_mw": float(
                    m["context_load_total_p_mw"]
                ),
                "context_bess_soc": float(
                    m["context_bess_soc"]
                ),
            }

            for signal in (
                INVERTER_SIGNALS
                + PHYSICS_PROXY_EXTRA_SIGNALS
            ):
                row.update(
                    causal_signal_features(
                        signal,
                        ref,
                        cur,
                    )
                )

            for signal in BOOLEAN_SIGNALS:
                row[
                    f"{signal}__ref_fraction"
                ] = float(
                    ref[signal]
                    .astype(float)
                    .mean()
                )

                row[
                    f"{signal}__cur_fraction"
                ] = float(
                    cur[signal]
                    .astype(float)
                    .mean()
                )

            # Only active-event windows are positive.
            row["target_binary"] = int(
                phase == "event"
            )

            rows.append(row)

    return pd.DataFrame(rows)


def parameter_shift_score(meta):
    """
    OOD score based only on plant/controller parameter deviation,
    not event label/severity.
    """
    cols = [
        "line_r_scale",
        "line_x_scale",
        "gfl_kp_scale",
        "gfl_ki_scale",
        "gfm_m_scale",
        "gfm_d_scale",
    ]

    score = np.zeros(len(meta), dtype=float)

    for c in cols:
        x = meta[c].to_numpy(dtype=float)
        score += np.abs(x - 1.0)

    return score


def assign_ood_splits(meta):
    m = meta.copy()
    m["parameter_shift_score"] = (
        parameter_shift_score(m)
    )

    split_map = {}

    for event_type, g in m.groupby(
        "event_type"
    ):
        g = g.sort_values(
            "parameter_shift_score"
        )

        ids = g[
            "scenario_id"
        ].astype(int).to_numpy()

        n = len(ids)

        n_train = max(
            1,
            int(np.floor(0.70 * n))
        )
        n_val = max(
            1,
            int(np.floor(0.15 * n))
        )

        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        for sid in ids[:n_train]:
            split_map[int(sid)] = "train"

        for sid in ids[
            n_train:n_train+n_val
        ]:
            split_map[int(sid)] = "validation"

        for sid in ids[
            n_train+n_val:
        ]:
            split_map[int(sid)] = "test"

    return split_map


def feature_columns(
    df,
    feature_set,
):
    if feature_set == "inverter_only":
        allowed = (
            INVERTER_SIGNALS
            + BOOLEAN_SIGNALS
        )

    elif feature_set == "physics_proxy":
        allowed = (
            INVERTER_SIGNALS
            + PHYSICS_PROXY_EXTRA_SIGNALS
            + BOOLEAN_SIGNALS
        )

    else:
        raise ValueError(feature_set)

    prefixes = tuple(
        s + "__"
        for s in allowed
    )

    cols = [
        c
        for c in df.columns
        if c.startswith(prefixes)
    ]

    forbidden_tokens = (
        "context_bess_soc",
        "scenario",
        "event_type",
        "event_family",
        "window_mode",
        "split",
    )

    cols = [
        c for c in cols
        if not any(
            tok in c
            for tok in forbidden_tokens
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
                        max_iter=250,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def positive_scores(model, X):
    p = model.predict_proba(X)
    classes = list(model.classes_)
    return p[:, classes.index(1)]


def validation_threshold(
    y_val,
    scores,
    target_fpr=TARGET_FPR,
):
    y_val = np.asarray(y_val)
    scores = np.asarray(scores)

    normal = scores[
        y_val == 0
    ]

    return float(
        np.quantile(
            normal,
            1.0 - target_fpr,
            method="higher",
        )
    )


def prepare_split(
    windows,
    split_col,
):
    """
    Exclude recovery/transition windows from model fitting/evaluation.
    They remain saved in the dataset for future sequential work.
    """
    usable = windows[
        windows["phase"].isin(
            ["normal", "pre_event", "event"]
        )
    ].copy()

    return {
        s: usable[
            usable[split_col] == s
        ].copy()
        for s in (
            "train",
            "validation",
            "test",
        )
    }


def window_metrics(
    y_true,
    scores,
    threshold,
):
    pred = (
        scores >= threshold
    ).astype(int)

    y_true = np.asarray(y_true)

    normal = y_true == 0

    fpr = (
        float(
            np.mean(
                pred[normal] == 1
            )
        )
        if normal.any()
        else np.nan
    )

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
        "event_window_recall": float(
            recall_score(
                y_true,
                pred
            )
        ),
        "normal_window_fpr": fpr,
    }


def scenario_detection_metrics(
    test_full,
    test_scores,
    threshold,
    meta,
):
    d = test_full.copy()
    d["score"] = test_scores
    d["flag"] = (
        d["score"] >= threshold
    )

    meta_idx = meta.set_index(
        "scenario_id"
    )

    event_hits = []
    delays_ms = []
    pre_event_false_alarm = []

    # Use only true event trajectories for detection-delay metrics.
    for sid, g in d[
        d["window_mode"] == "event"
    ].groupby(
        "scenario_id"
    ):
        sid = int(sid)
        m = meta_idx.loc[sid]

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
            pre_event_false_alarm.append(
                float(pre["flag"].any())
            )

        candidate = g[
            (g["window_end_s"] >= t0)
            & (g["window_end_s"] <= t1)
        ].sort_values(
            "window_end_s"
        )

        hit = candidate[
            candidate["flag"]
        ]

        if hit.empty:
            event_hits.append(0.0)
        else:
            event_hits.append(1.0)

            first_t = float(
                hit.iloc[0][
                    "window_end_s"
                ]
            )

            delays_ms.append(
                1000.0 * (
                    first_t - t0
                )
            )

    # Counterfactual false alarm at trajectory level.
    normal_traj_false = []

    for sid, g in d[
        d["window_mode"]
        == "normal_counterfactual"
    ].groupby(
        "scenario_id"
    ):
        normal_traj_false.append(
            float(g["flag"].any())
        )

    return {
        "scenario_detection_rate": float(
            np.mean(event_hits)
        ) if event_hits else np.nan,

        "median_detection_delay_ms": float(
            np.median(delays_ms)
        ) if delays_ms else np.nan,

        "p95_detection_delay_ms": float(
            np.percentile(
                delays_ms,
                95
            )
        ) if delays_ms else np.nan,

        "pre_event_false_alarm_rate": float(
            np.mean(
                pre_event_false_alarm
            )
        ) if pre_event_false_alarm else np.nan,

        "normal_trajectory_false_alarm_rate": float(
            np.mean(
                normal_traj_false
            )
        ) if normal_traj_false else np.nan,
    }


def run_regime(
    windows,
    meta,
    split_col,
    regime_name,
    feature_set,
):
    cols = feature_columns(
        windows,
        feature_set,
    )

    split = prepare_split(
        windows,
        split_col,
    )

    train = split["train"]
    val = split["validation"]
    test = split["test"]

    X_train = train[cols]
    y_train = train[
        "target_binary"
    ]

    X_val = val[cols]
    y_val = val[
        "target_binary"
    ]

    X_test = test[cols]
    y_test = test[
        "target_binary"
    ]

    rows = []
    curves = {}

    for name, model in model_zoo().items():
        model.fit(
            X_train,
            y_train,
        )

        val_scores = positive_scores(
            model,
            X_val,
        )

        threshold = validation_threshold(
            y_val,
            val_scores,
        )

        test_scores = positive_scores(
            model,
            X_test,
        )

        wm = window_metrics(
            y_test,
            test_scores,
            threshold,
        )

        sm = scenario_detection_metrics(
            test,
            test_scores,
            threshold,
            meta,
        )

        row = {
            "regime": regime_name,
            "feature_set": feature_set,
            "model": name,
            "n_features": len(cols),
            "threshold": threshold,
            **wm,
            **sm,
        }

        rows.append(row)

        fpr, tpr, _ = roc_curve(
            y_test,
            test_scores,
        )

        curves[name] = (
            fpr,
            tpr,
            wm["roc_auc"],
        )

    return pd.DataFrame(rows), curves


def plot_auc_comparison(results):
    d = results.copy()

    labels = (
        d["regime"]
        + "\n"
        + d["feature_set"]
        + "\n"
        + d["model"]
    ).tolist()

    x = np.arange(len(d))

    fig, ax = plt.subplots(
        figsize=(13, 6.5)
    )

    ax.bar(
        x,
        d["roc_auc"].to_numpy()
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
        fontsize=8,
    )

    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel(
        "Test ROC-AUC"
    )
    ax.set_title(
        "Causal Streaming Detection — IID vs Parameter-Shift OOD"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "33_streaming_iid_vs_ood_auc.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_delay(results):
    d = results[
        results[
            "feature_set"
        ] == "inverter_only"
    ].copy()

    labels = (
        d["regime"]
        + "\n"
        + d["model"]
    ).tolist()

    x = np.arange(len(d))

    fig, ax = plt.subplots(
        figsize=(10.5, 6)
    )

    ax.bar(
        x,
        d[
            "median_detection_delay_ms"
        ].to_numpy()
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel(
        "Median detection delay (ms)"
    )
    ax.set_title(
        "Streaming Event Detection Delay — Inverter-Only Features"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "34_streaming_detection_delay.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_scenario_detection(results):
    d = results[
        results[
            "feature_set"
        ] == "inverter_only"
    ].copy()

    labels = (
        d["regime"]
        + "\n"
        + d["model"]
    ).tolist()

    x = np.arange(len(d))
    detect = (
        100
        * d[
            "scenario_detection_rate"
        ].to_numpy()
    )
    false_alarm = (
        100
        * d[
            "normal_trajectory_false_alarm_rate"
        ].to_numpy()
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 6.2)
    )

    ax.bar(
        x - width/2,
        detect,
        width,
        label="Scenario detection",
    )

    ax.bar(
        x + width/2,
        false_alarm,
        width,
        label="Normal-trajectory false alarm",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Streaming Scenario Detection vs False Alarm"
    )
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "35_streaming_scenario_detection.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            str(RAW_PATH)
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            str(META_PATH)
        )

    print(
        "\n=== PHASE 2I: CAUSAL STREAMING + OOD BENCHMARK ==="
    )

    print(
        "Loading raw transient dataset ..."
    )

    raw = pd.read_csv(
        RAW_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    print(
        f"Raw samples : {len(raw):,}"
    )
    print(
        f"Scenarios   : "
        f"{meta['scenario_id'].nunique()}"
    )

    ood_map = assign_ood_splits(
        meta
    )

    meta["ood_split"] = (
        meta["scenario_id"]
        .map(ood_map)
    )

    windows = build_streaming_feature_dataset(
        raw,
        meta,
    )

    windows["ood_split"] = (
        windows["scenario_id"]
        .map(ood_map)
    )

    windows.to_csv(
        DATA
        / "phase2i_streaming_feature_dataset.csv",
        index=False,
    )

    print(
        f"Streaming windows : {len(windows):,}"
    )

    print(
        "\nPhase counts:"
    )
    print(
        windows[
            "phase"
        ].value_counts().to_string()
    )

    result_frames = []

    for (
        split_col,
        regime_name,
    ) in [
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
                f"\nRunning "
                f"{regime_name} / "
                f"{feature_set} ..."
            )

            result, _ = run_regime(
                windows,
                meta,
                split_col,
                regime_name,
                feature_set,
            )

            result_frames.append(
                result
            )

            print(
                result[
                    [
                        "model",
                        "roc_auc",
                        "average_precision",
                        "event_window_recall",
                        "normal_window_fpr",
                        "scenario_detection_rate",
                        "median_detection_delay_ms",
                        "normal_trajectory_false_alarm_rate",
                    ]
                ].to_string(
                    index=False
                )
            )

    results = pd.concat(
        result_frames,
        ignore_index=True,
    )

    results.to_csv(
        RESULTS
        / "phase2i_streaming_generalization_metrics.csv",
        index=False,
    )

    meta[
        [
            "scenario_id",
            "event_type",
            "event_family",
            "dataset_split",
            "ood_split",
        ]
    ].to_csv(
        DATA
        / "phase2i_iid_and_ood_split_map.csv",
        index=False,
    )

    plot_auc_comparison(
        results
    )

    plot_delay(
        results
    )

    plot_scenario_detection(
        results
    )

    print(
        "\n=== FINAL STREAMING SUMMARY ==="
    )
    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )
    print(
        "  data/phase2i_streaming_feature_dataset.csv"
    )
    print(
        "  data/phase2i_iid_and_ood_split_map.csv"
    )
    print(
        "  results/phase2i_streaming_generalization_metrics.csv"
    )
    print(
        "  figures/33_streaming_iid_vs_ood_auc.png"
    )
    print(
        "  figures/34_streaming_detection_delay.png"
    )
    print(
        "  figures/35_streaming_scenario_detection.png"
    )

    print(
        "\nWHY THIS PHASE MATTERS:\n"
        "The previous event-aware benchmark knew the true event boundaries. "
        "This benchmark does not. Every decision uses only the previous "
        "400 ms of causal history and is issued every 50 ms. It also measures "
        "generalization to the largest controller/network parameter shifts. "
        "If performance now drops materially, that is useful evidence that "
        "the earlier near-perfect scores were partly caused by an easy "
        "synthetic benchmark rather than a universally solved detection problem. "
        "Only after this harder baseline is understood should WLS graph states "
        "and topology-aware models be added."
    )


if __name__ == "__main__":
    main()
