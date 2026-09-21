from pathlib import Path
import importlib.util
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

STREAM_PATH = DATA / "phase2i_streaming_feature_dataset.csv"
RAW_PATH = DATA / "phase2g_v2_transient_timeseries.csv.gz"
META_PATH = DATA / "phase2g_v2_scenario_metadata.csv"

TARGET_ATTACK = "cyber_gfl_current_limit"
RANDOM_STATE = 20260812


def load_phase2j():
    candidates = [
        ROOT / "14_cyber_vs_physical_zero_day_benchmark_FIXED.py",
        ROOT / "14_cyber_vs_physical_zero_day_benchmark.py",
    ]

    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(
                "phase2j",
                path,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Could not find Phase 2J benchmark script."
    )


phase2j = load_phase2j()


def event_rows(df):
    return df[
        (df["phase"] == "event")
        & (df["window_mode"] == "event")
    ].copy()


def build_iid_rf_detector(df, meta):
    """
    Refit the strongest simple inverter-only IID cyber detector and
    return the validation-selected threshold/persistence policy.
    """
    cols = phase2j.feature_columns(
        df,
        "inverter_only",
    )

    train = df[
        df["iid_split"] == "train"
    ].copy()

    val = df[
        df["iid_split"] == "validation"
    ].copy()

    model = RandomForestClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    (
        fitted,
        threshold,
        policy_name,
        k,
        n,
        policy_df,
    ) = phase2j.train_and_select_policy(
        train,
        val,
        cols,
        "RandomForest",
        model,
        meta,
    )

    return (
        fitted,
        threshold,
        policy_name,
        k,
        n,
        cols,
        policy_df,
    )


def physical_false_alarm_by_type(
    df,
    meta,
    model,
    threshold,
    k,
    n,
    cols,
):
    test = df[
        df["iid_split"] == "test"
    ].copy()

    scores = phase2j.positive_scores(
        model,
        test[cols],
    )

    scored = test.copy()
    scored["score"] = scores

    scored = phase2j.apply_persistence(
        scored,
        "score",
        threshold,
        k,
        n,
    )

    meta_idx = meta.set_index("scenario_id")
    rows = []

    for sid, g in scored[
        scored["window_mode"] == "event"
    ].groupby("scenario_id"):
        sid = int(sid)
        m = meta_idx.loc[sid]

        if str(m["event_family"]) != "physical":
            continue

        t0 = float(m["event_start_s"])
        t1 = float(m["event_end_s"])

        during = g[
            (g["window_end_s"] >= t0)
            & (g["window_end_s"] <= t1)
        ]

        rows.append(
            {
                "scenario_id": sid,
                "event_type": str(m["event_type"]),
                "false_alarm": float(
                    during["alarm"].any()
                ),
                "max_score": float(
                    during["score"].max()
                    if len(during)
                    else np.nan
                ),
            }
        )

    d = pd.DataFrame(rows)

    summary = (
        d.groupby(
            "event_type",
            as_index=False,
        )
        .agg(
            n=("scenario_id", "count"),
            trajectory_false_alarm_rate=(
                "false_alarm",
                "mean",
            ),
            median_max_score=(
                "max_score",
                "median",
            ),
            max_score=(
                "max_score",
                "max",
            ),
        )
    )

    return d, summary


def standardized_centroid_distances(
    df,
):
    """
    Diagnose where unseen GFL-current-limit event windows lie in the
    inverter-only feature space relative to the known classes.
    """
    cols = phase2j.feature_columns(
        df,
        "inverter_only",
    )

    train = df[
        df["iid_split"] == "train"
    ].copy()

    train_for_fit = train[
        train["event_type"] != TARGET_ATTACK
    ].copy()

    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_fit = imp.fit_transform(
        train_for_fit[cols]
    )
    X_fit = scaler.fit_transform(X_fit)

    target = event_rows(df)
    target = target[
        target["event_type"] == TARGET_ATTACK
    ].copy()

    X_target = scaler.transform(
        imp.transform(
            target[cols]
        )
    )

    target_centroid = np.mean(
        X_target,
        axis=0,
    )

    reference_groups = []

    # Normal / pre-event benign reference.
    normal_ref = train_for_fit[
        train_for_fit["phase"].isin(
            ["normal", "pre_event"]
        )
    ].copy()

    reference_groups.append(
        ("benign_normal_pre", normal_ref)
    )

    # Event-level reference classes.
    for label, g in event_rows(
        train_for_fit
    ).groupby("event_type"):
        reference_groups.append(
            (str(label), g.copy())
        )

    rows = []

    for label, g in reference_groups:
        if g.empty:
            continue

        X = scaler.transform(
            imp.transform(
                g[cols]
            )
        )

        centroid = np.mean(
            X,
            axis=0,
        )

        distance = float(
            np.linalg.norm(
                target_centroid
                - centroid
            )
        )

        rows.append(
            {
                "reference_class": label,
                "euclidean_centroid_distance": distance,
                "n_windows": len(g),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "euclidean_centroid_distance"
        ),
        cols,
        imp,
        scaler,
    )


def pca_failure_map(
    df,
    cols,
    imp,
    scaler,
):
    """
    2-D diagnostic only. PCA is not used as a detector.
    """
    e = event_rows(df).copy()

    # Keep known train events + all target events + normal/pre IID test
    use = e[
        (
            e["iid_split"]
            == "train"
        )
        | (
            e["event_type"]
            == TARGET_ATTACK
        )
    ].copy()

    benign = df[
        (df["iid_split"] == "test")
        & (
            df["phase"].isin(
                ["normal", "pre_event"]
            )
        )
    ].sample(
        n=min(
            1200,
            len(
                df[
                    (df["iid_split"] == "test")
                    & (
                        df["phase"].isin(
                            ["normal", "pre_event"]
                        )
                    )
                ]
            ),
        ),
        random_state=RANDOM_STATE,
    ).copy()

    benign["plot_class"] = "benign_normal_pre"

    use["plot_class"] = use[
        "event_type"
    ]

    plot_df = pd.concat(
        [use, benign],
        ignore_index=True,
    )

    X = scaler.transform(
        imp.transform(
            plot_df[cols]
        )
    )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )
    Z = pca.fit_transform(X)

    plot_df["PC1"] = Z[:, 0]
    plot_df["PC2"] = Z[:, 1]

    fig, ax = plt.subplots(
        figsize=(11, 7.5)
    )

    classes = list(
        plot_df["plot_class"].unique()
    )

    for label in classes:
        g = plot_df[
            plot_df["plot_class"] == label
        ]

        if label == TARGET_ATTACK:
            ax.scatter(
                g["PC1"],
                g["PC2"],
                s=55,
                marker="x",
                linewidths=1.4,
                label=label + " (held-out focus)",
            )
        else:
            ax.scatter(
                g["PC1"],
                g["PC2"],
                s=22,
                alpha=0.42,
                label=label,
            )

    ax.set_xlabel(
        f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}% variance)"
    )
    ax.set_ylabel(
        f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}% variance)"
    )
    ax.set_title(
        "Why the Zero-Day GFL Current-Limit Attack Is Missed — "
        "Inverter-Only Feature Space"
    )

    ax.legend(
        frameon=False,
        fontsize=7.5,
        ncol=2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "38_zero_day_gfl_current_limit_pca.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    return pca.explained_variance_ratio_


def raw_tracking_residual_analysis(
    raw,
    meta,
):
    """
    Test a deployable physics-consistency idea:

        expected active-power reference = trusted pv_pref_pu
        reconstructed actual active power ~ V_meas * i_d

        r_P = |P_ref,trusted - V_meas * i_d|

    This does NOT use the attacked Imax value itself.

    It assumes pv_pref_pu is available from a trusted supervisory/control
    record. That assumption must be stated explicitly in any paper.
    """
    meta_idx = meta.set_index("scenario_id")
    rows = []

    for (
        sid,
        mode,
    ), g in raw.groupby(
        [
            "scenario_id",
            "window_mode",
        ],
        sort=False,
    ):
        sid = int(sid)
        m = meta_idx.loc[sid]

        t0 = float(m["event_start_s"])
        t1 = float(m["event_end_s"])

        during = g[
            (g["time_s"] >= t0)
            & (g["time_s"] < t1)
        ].copy()

        if during.empty:
            continue

        p_actual_reconstructed = (
            during["v_gfl_measured_pu"]
            * during["gfl_id_pu"]
        )

        residual = np.abs(
            during["pv_pref_pu"]
            - p_actual_reconstructed
        )

        rows.append(
            {
                "scenario_id": sid,
                "window_mode": mode,
                "event_type": str(m["event_type"]),
                "event_family": str(m["event_family"]),
                "max_p_tracking_residual_pu": float(
                    residual.max()
                ),
                "rms_p_tracking_residual_pu": float(
                    np.sqrt(
                        np.mean(
                            residual.to_numpy()
                            ** 2
                        )
                    )
                ),
                "mean_p_tracking_residual_pu": float(
                    residual.mean()
                ),
                "max_gfl_current_pu": float(
                    during["gfl_current_pu"].max()
                ),
                "limiter_fraction": float(
                    during["gfl_limiter"]
                    .astype(float)
                    .mean()
                ),
            }
        )

    d = pd.DataFrame(rows)

    event_d = d[
        d["window_mode"] == "event"
    ].copy()

    summary = (
        event_d.groupby(
            [
                "event_type",
                "event_family",
            ],
            as_index=False,
        )
        .agg(
            n=("scenario_id", "count"),
            median_max_tracking_residual=(
                "max_p_tracking_residual_pu",
                "median",
            ),
            p95_max_tracking_residual=(
                "max_p_tracking_residual_pu",
                lambda x: float(
                    np.percentile(
                        x,
                        95,
                    )
                ),
            ),
            median_rms_tracking_residual=(
                "rms_p_tracking_residual_pu",
                "median",
            ),
            median_limiter_fraction=(
                "limiter_fraction",
                "median",
            ),
        )
        .sort_values(
            "median_max_tracking_residual",
            ascending=False,
        )
    )

    # Compare matched target attack against its normal counterfactual.
    target = d[
        d["event_type"] == TARGET_ATTACK
    ].copy()

    pivot = target.pivot(
        index="scenario_id",
        columns="window_mode",
        values="max_p_tracking_residual_pu",
    ).dropna()

    if (
        "event" in pivot.columns
        and "normal_counterfactual"
        in pivot.columns
    ):
        pivot["matched_delta"] = (
            pivot["event"]
            - pivot[
                "normal_counterfactual"
            ]
        )
    else:
        pivot["matched_delta"] = np.nan

    return d, summary, pivot.reset_index()


def plot_tracking_residual(summary):
    d = summary.sort_values(
        "median_max_tracking_residual",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 7)
    )

    ax.barh(
        d["event_type"],
        d["median_max_tracking_residual"],
    )

    ax.set_xlabel(
        "Median max |P_ref,trusted − V_meas·i_d| (pu)"
    )
    ax.set_title(
        "Trusted-Command Active-Power Consistency Residual by Event Type"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "39_trusted_command_tracking_residual.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_physical_false_alarm(summary):
    d = summary.sort_values(
        "trajectory_false_alarm_rate",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    ax.barh(
        d["event_type"],
        100*d["trajectory_false_alarm_rate"],
    )

    ax.set_xlabel(
        "Trajectory false-alarm rate (%)"
    )
    ax.set_title(
        "Which Legitimate Physical Disturbance Confuses the IID RF Detector?"
    )
    ax.set_xlim(0, 105)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        FIGURES
        / "40_physical_false_alarm_by_event.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    for p in (
        STREAM_PATH,
        RAW_PATH,
        META_PATH,
    ):
        if not p.exists():
            raise FileNotFoundError(str(p))

    print(
        "\n=== PHASE 2K: ZERO-DAY FAILURE DIAGNOSTIC ==="
    )

    windows = pd.read_csv(
        STREAM_PATH
    )
    raw = pd.read_csv(
        RAW_PATH
    )
    meta = pd.read_csv(
        META_PATH
    )

    df = phase2j.add_cyber_target(
        windows
    )

    (
        rf,
        threshold,
        policy_name,
        k,
        n,
        cols,
        policy_df,
    ) = build_iid_rf_detector(
        df,
        meta,
    )

    print(
        f"\nReference detector: IID inverter-only RandomForest"
    )
    print(
        f"Threshold          : {threshold:.6f}"
    )
    print(
        f"Persistence        : {policy_name}"
    )

    (
        physical_rows,
        physical_summary,
    ) = physical_false_alarm_by_type(
        df,
        meta,
        rf,
        threshold,
        k,
        n,
        cols,
    )

    (
        centroid,
        cols,
        imp,
        scaler,
    ) = standardized_centroid_distances(
        df
    )

    pca_var = pca_failure_map(
        df,
        cols,
        imp,
        scaler,
    )

    (
        tracking_rows,
        tracking_summary,
        tracking_matched,
    ) = raw_tracking_residual_analysis(
        raw,
        meta,
    )

    physical_summary.to_csv(
        RESULTS
        / "phase2k_physical_false_alarm_by_type.csv",
        index=False,
    )

    centroid.to_csv(
        RESULTS
        / "phase2k_gfl_current_limit_centroid_distances.csv",
        index=False,
    )

    tracking_summary.to_csv(
        RESULTS
        / "phase2k_tracking_residual_by_event.csv",
        index=False,
    )

    tracking_matched.to_csv(
        RESULTS
        / "phase2k_gfl_current_limit_matched_tracking_residual.csv",
        index=False,
    )

    plot_tracking_residual(
        tracking_summary
    )

    plot_physical_false_alarm(
        physical_summary
    )

    print(
        "\n=== PHYSICAL FALSE ALARM DIAGNOSTIC ==="
    )
    print(
        physical_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== GFL CURRENT-LIMIT FEATURE-SPACE NEIGHBORS ==="
    )
    print(
        centroid.head(12).to_string(
            index=False
        )
    )

    print(
        "\nPCA diagnostic explained variance:"
    )
    print(
        f"  PC1={100*pca_var[0]:.2f}% | "
        f"PC2={100*pca_var[1]:.2f}%"
    )

    print(
        "\n=== TRUSTED-COMMAND PHYSICS RESIDUAL ==="
    )
    print(
        tracking_summary.to_string(
            index=False
        )
    )

    if not tracking_matched.empty:
        print(
            "\nGFL current-limit matched tracking residual:"
        )
        print(
            f"  median event residual      = "
            f"{tracking_matched['event'].median():.5f} pu"
        )
        print(
            f"  median normal residual     = "
            f"{tracking_matched['normal_counterfactual'].median():.5f} pu"
        )
        print(
            f"  median matched increase    = "
            f"{tracking_matched['matched_delta'].median():.5f} pu"
        )
        print(
            f"  positive matched increase  = "
            f"{100*(tracking_matched['matched_delta'] > 0).mean():.1f}%"
        )

    print(
        "\nSaved:"
    )
    print(
        "  results/phase2k_physical_false_alarm_by_type.csv"
    )
    print(
        "  results/phase2k_gfl_current_limit_centroid_distances.csv"
    )
    print(
        "  results/phase2k_tracking_residual_by_event.csv"
    )
    print(
        "  results/phase2k_gfl_current_limit_matched_tracking_residual.csv"
    )
    print(
        "  figures/38_zero_day_gfl_current_limit_pca.png"
    )
    print(
        "  figures/39_trusted_command_tracking_residual.png"
    )
    print(
        "  figures/40_physical_false_alarm_by_event.png"
    )

    print(
        "\nNEXT DECISION:\n"
        "If the held-out GFL current-limit windows sit closer to benign/"
        "physical behavior than to the other cyber classes, the zero-day "
        "failure is structurally understandable rather than a classifier bug. "
        "If the trusted-command active-power tracking residual separates this "
        "attack from its matched normal case, the next detector should be "
        "multi-layer: inverter temporal fingerprints + trusted-command/model "
        "consistency + WLS/topology residuals. That is more defensible than "
        "adding a GNN blindly."
    )


if __name__ == "__main__":
    main()
