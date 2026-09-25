from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RAW_PATH = DATA / "phase2g_v2_transient_timeseries.csv.gz"
META_PATH = DATA / "phase2g_v2_scenario_metadata.csv"

TARGET_ATTACK = "cyber_gfl_current_limit"
RANDOM_STATE = 20260812

# Independent monitor uncertainty levels.
NOISE_LEVELS = [
    0.0025,   # 0.25%
    0.0050,   # 0.50%
    0.0100,   # 1.00%
    0.0200,   # 2.00%
]

# Independent multiplicative gain uncertainty.
GAIN_SIGMAS = [
    0.000,
    0.002,
    0.005,
]

# Trusted-command transport / timestamp delay.
COMMAND_DELAYS_MS = [
    0,
    20,
    50,
]

TARGET_BENIGN_FPR = 0.01

S_BASE_MVA = 0.10


def event_mask(t, start, end):
    return (
        (t >= start)
        & (t < end)
    )


def shift_command(
    time_s,
    command,
    delay_s,
):
    """
    Causal monitor-side command delay:
    at time t, the monitor sees the command value from t-delay.
    """
    t = np.asarray(
        time_s,
        dtype=float,
    )
    u = np.asarray(
        command,
        dtype=float,
    )

    if delay_s <= 0:
        return u.copy()

    query = t - delay_s

    return np.interp(
        query,
        t,
        u,
        left=u[0],
        right=u[-1],
    )


def independent_monitor_measurements(
    rng,
    v,
    i_d,
    sigma,
    gain_sigma,
):
    """
    Emulate a monitor that does NOT reuse the exact internal controller
    quantities algebraically. Voltage/current telemetry receive independent
    additive and gain uncertainty.
    """
    v = np.asarray(v, dtype=float)
    i_d = np.asarray(i_d, dtype=float)

    gain_v = rng.normal(
        0.0,
        gain_sigma,
    )
    gain_i = rng.normal(
        0.0,
        gain_sigma,
    )

    v_obs = (
        v * (1.0 + gain_v)
        + rng.normal(
            0.0,
            sigma,
            len(v),
        )
    )

    i_obs = (
        i_d * (1.0 + gain_i)
        + rng.normal(
            0.0,
            sigma,
            len(i_d),
        )
    )

    return v_obs, i_obs


def scenario_residuals(
    raw,
    meta,
    sigma,
    gain_sigma,
    command_delay_ms,
    mc_seed,
):
    meta_idx = meta.set_index(
        "scenario_id"
    )

    rows = []

    master = np.random.default_rng(
        mc_seed
    )

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

        p_ref_seen = shift_command(
            t,
            p_ref,
            command_delay_ms
            / 1000.0,
        )

        # Reproducible but independent monitor noise per trajectory.
        local_seed = int(
            master.integers(
                0,
                2**31 - 1,
            )
        )
        rng = np.random.default_rng(
            local_seed
        )

        v_obs, i_obs = (
            independent_monitor_measurements(
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
                sigma,
                gain_sigma,
            )
        )

        residual = np.abs(
            p_ref_seen
            - v_obs * i_obs
        )

        t0 = float(
            m["event_start_s"]
        )
        t1 = float(
            m["event_end_s"]
        )

        emask = event_mask(
            t,
            t0,
            t1,
        )

        # Pre-event calibration interval.
        premask = (
            t < t0
        )

        if not np.any(emask):
            continue

        event_res = residual[
            emask
        ]

        pre_res = residual[
            premask
        ]

        event_limiter = g.loc[
            emask,
            "gfl_limiter",
        ].astype(float)

        rows.append(
            {
                "scenario_id": sid,
                "window_mode": mode,
                "event_type": str(
                    m["event_type"]
                ),
                "event_family": str(
                    m["event_family"]
                ),
                "dataset_split": str(
                    m["dataset_split"]
                ),

                "sigma_pu": sigma,
                "gain_sigma": gain_sigma,
                "command_delay_ms":
                    command_delay_ms,

                "event_max_residual_pu": float(
                    np.max(
                        event_res
                    )
                ),
                "event_rms_residual_pu": float(
                    np.sqrt(
                        np.mean(
                            event_res ** 2
                        )
                    )
                ),
                "pre_p99_residual_pu": float(
                    np.percentile(
                        pre_res,
                        99,
                    )
                    if len(pre_res)
                    else np.nan
                ),
                "gfl_limiter_fraction": float(
                    event_limiter.mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def calibrate_threshold(
    d,
):
    """
    Calibrate only on TRAIN benign/physical trajectories.

    Cyber events are excluded from threshold selection.
    """
    train = d[
        d[
            "dataset_split"
        ] == "train"
    ].copy()

    benign = train[
        (
            train[
                "event_family"
            ] == "physical"
        )
        | (
            train[
                "window_mode"
            ]
            == "normal_counterfactual"
        )
    ]

    values = benign[
        "event_max_residual_pu"
    ].to_numpy(
        dtype=float
    )

    if len(values) == 0:
        raise RuntimeError(
            "No benign training residuals."
        )

    return float(
        np.quantile(
            values,
            1.0
            - TARGET_BENIGN_FPR,
            method="higher",
        )
    )


def evaluate_setting(
    d,
    threshold,
):
    test = d[
        d[
            "dataset_split"
        ] == "test"
    ].copy()

    # Only event trajectories for actual event-family evaluation.
    event = test[
        test[
            "window_mode"
        ] == "event"
    ].copy()

    event[
        "alarm"
    ] = (
        event[
            "event_max_residual_pu"
        ]
        >= threshold
    )

    target = event[
        event[
            "event_type"
        ] == TARGET_ATTACK
    ]

    physical = event[
        event[
            "event_family"
        ] == "physical"
    ]

    other_cyber = event[
        (
            event[
                "event_family"
            ].isin(
                [
                    "cyber",
                    "hybrid",
                ]
            )
        )
        & (
            event[
                "event_type"
            ]
            != TARGET_ATTACK
        )
    ]

    normal = test[
        test[
            "window_mode"
        ]
        == "normal_counterfactual"
    ].copy()

    normal[
        "alarm"
    ] = (
        normal[
            "event_max_residual_pu"
        ]
        >= threshold
    )

    return {
        "threshold_pu": threshold,

        "gfl_current_limit_detection_rate": float(
            target[
                "alarm"
            ].mean()
            if len(target)
            else np.nan
        ),

        "physical_false_alarm_rate": float(
            physical[
                "alarm"
            ].mean()
            if len(physical)
            else np.nan
        ),

        "normal_counterfactual_false_alarm_rate": float(
            normal[
                "alarm"
            ].mean()
            if len(normal)
            else np.nan
        ),

        # This residual is mechanism-specific, so low detection on other
        # cyber types is not a failure.
        "other_cyber_alarm_rate": float(
            other_cyber[
                "alarm"
            ].mean()
            if len(other_cyber)
            else np.nan
        ),
    }


def severity_audit(
    d,
    meta,
    threshold,
):
    meta_cols = [
        "scenario_id",
        "gfl_attack_imax_pu",
        "estimated_pre_event_gfl_current_pu",
    ]

    target_meta = meta[
        meta[
            "event_type"
        ] == TARGET_ATTACK
    ][
        meta_cols
    ].copy()

    x = d[
        (
            d[
                "event_type"
            ] == TARGET_ATTACK
        )
        & (
            d[
                "window_mode"
            ] == "event"
        )
    ].merge(
        target_meta,
        on="scenario_id",
        how="left",
    )

    x[
        "limit_ratio"
    ] = (
        x[
            "gfl_attack_imax_pu"
        ]
        / x[
            "estimated_pre_event_gfl_current_pu"
        ]
    )

    x[
        "alarm"
    ] = (
        x[
            "event_max_residual_pu"
        ]
        >= threshold
    )

    x[
        "severity_bin"
    ] = pd.cut(
        x[
            "limit_ratio"
        ],
        bins=[
            0.0,
            0.45,
            0.55,
            0.65,
            0.75,
            1.0,
        ],
        include_lowest=True,
    )

    return (
        x.groupby(
            "severity_bin",
            observed=True,
            as_index=False,
        )
        .agg(
            n=(
                "scenario_id",
                "count",
            ),
            median_limit_ratio=(
                "limit_ratio",
                "median",
            ),
            detection_rate=(
                "alarm",
                "mean",
            ),
            median_max_residual=(
                "event_max_residual_pu",
                "median",
            ),
        )
    )


def physical_limiter_audit(
    raw,
    meta,
):
    meta_idx = meta.set_index(
        "scenario_id"
    )
    rows = []

    for sid, g in raw[
        raw[
            "window_mode"
        ] == "event"
    ].groupby(
        "scenario_id"
    ):
        sid = int(sid)
        m = meta_idx.loc[sid]

        if str(
            m[
                "event_family"
            ]
        ) != "physical":
            continue

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

        d = g[
            (
                g[
                    "time_s"
                ] >= t0
            )
            & (
                g[
                    "time_s"
                ] < t1
            )
        ]

        rows.append(
            {
                "scenario_id": sid,
                "event_type": str(
                    m[
                        "event_type"
                    ]
                ),
                "max_gfl_current_pu": float(
                    d[
                        "gfl_current_pu"
                    ].max()
                ),
                "limiter_fraction": float(
                    d[
                        "gfl_limiter"
                    ].astype(
                        float
                    ).mean()
                ),
                "limiter_ever": bool(
                    d[
                        "gfl_limiter"
                    ].any()
                ),
            }
        )

    detail = pd.DataFrame(
        rows
    )

    summary = (
        detail.groupby(
            "event_type",
            as_index=False,
        )
        .agg(
            n=(
                "scenario_id",
                "count",
            ),
            limiter_scenario_rate=(
                "limiter_ever",
                "mean",
            ),
            median_limiter_fraction=(
                "limiter_fraction",
                "median",
            ),
            max_observed_current_pu=(
                "max_gfl_current_pu",
                "max",
            ),
        )
    )

    return detail, summary


def plot_noise_robustness(
    results,
):
    d = results[
        (
            results[
                "gain_sigma"
            ] == 0.002
        )
        & (
            results[
                "command_delay_ms"
            ] == 20
        )
    ].sort_values(
        "sigma_pu"
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 6)
    )

    ax.plot(
        100*d[
            "sigma_pu"
        ],
        100*d[
            "gfl_current_limit_detection_rate"
        ],
        marker="o",
        label="GFL current-limit detection",
    )

    ax.plot(
        100*d[
            "sigma_pu"
        ],
        100*d[
            "physical_false_alarm_rate"
        ],
        marker="o",
        label="Physical false alarm",
    )

    ax.plot(
        100*d[
            "sigma_pu"
        ],
        100*d[
            "normal_counterfactual_false_alarm_rate"
        ],
        marker="o",
        label="Normal false alarm",
    )

    ax.set_xlabel(
        "Independent monitor noise σ (% pu)"
    )
    ax.set_ylabel(
        "Scenario rate (%)"
    )
    ax.set_ylim(
        -2,
        105,
    )
    ax.set_title(
        "Trusted-Command Residual Robustness Under Independent Telemetry Noise"
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
        / "41_command_residual_noise_robustness.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_delay_effect(
    results,
):
    d = results[
        (
            results[
                "sigma_pu"
            ] == 0.005
        )
        & (
            results[
                "gain_sigma"
            ] == 0.002
        )
    ].sort_values(
        "command_delay_ms"
    )

    fig, ax = plt.subplots(
        figsize=(9.5, 6)
    )

    ax.plot(
        d[
            "command_delay_ms"
        ],
        100*d[
            "gfl_current_limit_detection_rate"
        ],
        marker="o",
        label="GFL current-limit detection",
    )

    ax.plot(
        d[
            "command_delay_ms"
        ],
        100*d[
            "physical_false_alarm_rate"
        ],
        marker="o",
        label="Physical false alarm",
    )

    ax.set_xlabel(
        "Trusted-command delay (ms)"
    )
    ax.set_ylabel(
        "Scenario rate (%)"
    )
    ax.set_ylim(
        -2,
        105,
    )
    ax.set_title(
        "Command-Residual Sensitivity to Supervisory-Channel Delay"
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
        / "42_command_residual_delay_sensitivity.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            str(
                RAW_PATH
            )
        )

    if not META_PATH.exists():
        raise FileNotFoundError(
            str(
                META_PATH
            )
        )

    print(
        "\n=== PHASE 2L: COMMAND-RESIDUAL STRESS TEST ==="
    )

    raw = pd.read_csv(
        RAW_PATH
    )

    meta = pd.read_csv(
        META_PATH
    )

    (
        physical_detail,
        physical_summary,
    ) = physical_limiter_audit(
        raw,
        meta,
    )

    print(
        "\n=== LEGITIMATE PHYSICAL LIMITER AUDIT ==="
    )
    print(
        physical_summary.to_string(
            index=False
        )
    )

    all_results = []
    representative_detail = None

    setting_id = 0

    for sigma in NOISE_LEVELS:
        for gain_sigma in GAIN_SIGMAS:
            for delay_ms in COMMAND_DELAYS_MS:
                setting_id += 1

                d = scenario_residuals(
                    raw,
                    meta,
                    sigma,
                    gain_sigma,
                    delay_ms,
                    RANDOM_STATE
                    + setting_id,
                )

                threshold = (
                    calibrate_threshold(
                        d
                    )
                )

                metrics = (
                    evaluate_setting(
                        d,
                        threshold,
                    )
                )

                all_results.append(
                    {
                        "sigma_pu": sigma,
                        "gain_sigma": gain_sigma,
                        "command_delay_ms":
                            delay_ms,
                        **metrics,
                    }
                )

                if (
                    abs(
                        sigma
                        - 0.005
                    )
                    < 1e-12
                    and abs(
                        gain_sigma
                        - 0.002
                    )
                    < 1e-12
                    and delay_ms
                    == 20
                ):
                    representative_detail = (
                        d.copy()
                    )

    results = pd.DataFrame(
        all_results
    )

    results.to_csv(
        RESULTS
        / "phase2l_command_residual_stress_metrics.csv",
        index=False,
    )

    physical_summary.to_csv(
        RESULTS
        / "phase2l_legitimate_physical_limiter_audit.csv",
        index=False,
    )

    plot_noise_robustness(
        results
    )

    plot_delay_effect(
        results
    )

    print(
        "\n=== COMMAND-RESIDUAL STRESS RESULTS ==="
    )

    show = results[
        (
            results[
                "gain_sigma"
            ] == 0.002
        )
    ][
        [
            "sigma_pu",
            "gain_sigma",
            "command_delay_ms",
            "threshold_pu",
            "gfl_current_limit_detection_rate",
            "physical_false_alarm_rate",
            "normal_counterfactual_false_alarm_rate",
            "other_cyber_alarm_rate",
        ]
    ]

    print(
        show.to_string(
            index=False
        )
    )

    if (
        representative_detail
        is not None
    ):
        threshold = (
            calibrate_threshold(
                representative_detail
            )
        )

        severity = (
            severity_audit(
                representative_detail,
                meta,
                threshold,
            )
        )

        severity.to_csv(
            RESULTS
            / "phase2l_current_limit_detection_by_severity.csv",
            index=False,
        )

        print(
            "\n=== CURRENT-LIMIT DETECTION BY SEVERITY ==="
        )
        print(
            severity.to_string(
                index=False
            )
        )

    print(
        "\nSaved:"
    )
    print(
        "  results/phase2l_command_residual_stress_metrics.csv"
    )
    print(
        "  results/phase2l_legitimate_physical_limiter_audit.csv"
    )
    print(
        "  results/phase2l_current_limit_detection_by_severity.csv"
    )
    print(
        "  figures/41_command_residual_noise_robustness.png"
    )
    print(
        "  figures/42_command_residual_delay_sensitivity.png"
    )

    print(
        "\nINTERPRETATION RULE:\n"
        "The ideal-model residual from Phase 2K was nearly algebraically "
        "zero outside current limiting, so it must not be treated as independent "
        "validation. This stress test breaks that identity with independent "
        "monitor noise, gain error and supervisory-command delay. If detection "
        "remains high with low physical/normal false-alarm rates, the residual "
        "is a credible mechanism-aware channel. If physical disturbances begin "
        "to trigger the GFL limiter or command delay causes many false alarms, "
        "the next challenge dataset must explicitly include those ambiguous "
        "conditions before any final hybrid detector claim is made."
    )


if __name__ == "__main__":
    main()
