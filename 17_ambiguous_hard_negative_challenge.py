from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

N_PER_CLASS = 24
SEED = 20260812

RAW_STEP_MS = 5.0
CONTROL_DT = 1e-3

S_BASE_MVA = 0.10


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / filename,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_module(
    "10_randomized_matched_benchmark.py",
    "phase2g_base",
)

stress = load_module(
    "16_command_residual_stress_test.py",
    "phase2l_stress",
)


CHALLENGES = [
    "physical_deep_sag_with_gfl_limiting",
    "physical_fast_pv_ramp",
    "cyber_mild_gfl_current_limit",
]


def choose_context(
    contexts,
    rng,
    challenge,
):
    c = contexts.copy()

    if challenge == "physical_deep_sag_with_gfl_limiting":
        c = c[
            c["pv_p_mw"] >= 0.028
        ]

    elif challenge == "physical_fast_pv_ramp":
        c = c[
            c["pv_p_mw"] >= 0.018
        ]

    elif challenge == "cyber_mild_gfl_current_limit":
        c = c[
            c["pv_p_mw"] >= 0.020
        ]

    if c.empty:
        raise RuntimeError(
            f"No eligible contexts for {challenge}"
        )

    idx = int(
        rng.integers(
            0,
            len(c),
        )
    )

    return c.iloc[idx]


def make_params(
    rng,
    challenge,
    context,
):
    if challenge == "physical_deep_sag_with_gfl_limiting":
        underlying = "physical_voltage_sag"

    elif challenge == "physical_fast_pv_ramp":
        underlying = "physical_cloud_transient"

    elif challenge == "cyber_mild_gfl_current_limit":
        underlying = "cyber_gfl_current_limit"

    else:
        raise ValueError(challenge)

    p = base.sample_scenario_params(
        rng,
        underlying,
    )

    # Narrow challenge-specific ranges.
    p["event_start_s"] = float(
        rng.uniform(
            0.75,
            0.90,
        )
    )

    p["event_duration_s"] = float(
        rng.uniform(
            0.35,
            0.65,
        )
    )

    p["event_end_s"] = (
        p["event_start_s"]
        + p["event_duration_s"]
    )

    p["sim_end_s"] = min(
        2.60,
        p["event_end_s"]
        + 0.90,
    )

    if challenge == "physical_deep_sag_with_gfl_limiting":
        # Deliberately extend the physical envelope below the original
        # dataset so legitimate disturbances can also trigger GFL limiting.
        p["physical_sag_vm_pu"] = float(
            rng.uniform(
                0.25,
                0.48,
            )
        )

        p["physical_angle_jump_deg"] = float(
            rng.uniform(
                3.0,
                12.0,
            )
        )

    elif challenge == "physical_fast_pv_ramp":
        # Legitimate fast PV-availability changes.
        p["cloud_min_factor"] = float(
            rng.uniform(
                0.25,
                0.75,
            )
        )

        p["cloud_ramp_s"] = float(
            rng.uniform(
                0.035,
                0.120,
            )
        )

    elif challenge == "cyber_mild_gfl_current_limit":
        # Mild manipulation: limit only slightly below expected operating
        # current. This is substantially stealthier than the previous
        # 0.35–0.72 expected-current fractions.
        v_context = float(
            context.get(
                "bus_1_vm_pu",
                0.98,
            )
        )

        expected_i = (
            float(
                context["pv_p_mw"]
            )
            / S_BASE_MVA
            / max(
                v_context,
                0.90,
            )
        )

        limit_ratio = float(
            rng.uniform(
                0.75,
                0.95,
            )
        )

        p[
            "gfl_attack_imax_pu"
        ] = float(
            np.clip(
                limit_ratio
                * expected_i,
                0.05,
                0.80,
            )
        )

        p[
            "estimated_pre_event_gfl_current_pu"
        ] = expected_i

        p[
            "challenge_limit_ratio"
        ] = limit_ratio

    return underlying, p


def run_pair(
    scenario_id,
    challenge,
    underlying,
    context,
    p,
    rng,
):
    n_steps = int(
        np.floor(
            p["sim_end_s"]
            / CONTROL_DT
        )
    ) + 1

    # Same stochastic sensor realization in event/counterfactual.
    noise_v = rng.normal(
        0.0,
        p["v_noise_sigma_pu"],
        n_steps,
    )

    noise_theta = np.deg2rad(
        rng.normal(
            0.0,
            p[
                "angle_noise_sigma_deg"
            ],
            n_steps,
        )
    )

    frames = []

    for mode in [
        "normal_counterfactual",
        "event",
    ]:
        df = base.run_window(
            scenario_id,
            underlying,
            context,
            p,
            mode,
            noise_v,
            noise_theta,
        )

        df[
            "challenge_type"
        ] = challenge

        frames.append(df)

    return frames


def event_limiter_fraction(
    df,
    p,
):
    d = df[
        (df["time_s"] >= p["event_start_s"])
        & (df["time_s"] < p["event_end_s"])
    ]

    if d.empty:
        return 0.0

    return float(
        d[
            "gfl_limiter"
        ].astype(float).mean()
    )


def build_challenge_dataset(
    contexts,
):
    master = np.random.default_rng(
        SEED
    )

    metadata = []
    raw_frames = []

    scenario_id = 0

    for challenge in CHALLENGES:
        accepted = 0
        attempts = 0

        print(
            f"\n--- {challenge} ---"
        )

        while accepted < N_PER_CLASS:
            attempts += 1

            if attempts > 10 * N_PER_CLASS:
                raise RuntimeError(
                    f"Could not generate enough accepted "
                    f"{challenge} cases."
                )

            seed = int(
                master.integers(
                    0,
                    2**31 - 1,
                )
            )

            rng = np.random.default_rng(
                seed
            )

            context = choose_context(
                contexts,
                rng,
                challenge,
            )

            underlying, p = make_params(
                rng,
                challenge,
                context,
            )

            frames = run_pair(
                scenario_id,
                challenge,
                underlying,
                context,
                p,
                rng,
            )

            event_df = frames[1]

            limiter_fraction = (
                event_limiter_fraction(
                    event_df,
                    p,
                )
            )

            # Quality gates.
            accept = True

            if (
                challenge
                == "physical_deep_sag_with_gfl_limiting"
            ):
                accept = (
                    limiter_fraction
                    >= 0.05
                )

            elif (
                challenge
                == "cyber_mild_gfl_current_limit"
            ):
                accept = (
                    limiter_fraction
                    >= 0.05
                )

            if not accept:
                continue

            raw_frames.extend(
                frames
            )

            metadata.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_seed": seed,
                    "challenge_type": challenge,
                    "underlying_event_type":
                        underlying,

                    "context_timestamp":
                        context["timestamp"],
                    "context_pv_p_mw":
                        float(
                            context[
                                "pv_p_mw"
                            ]
                        ),
                    "context_load_total_p_mw":
                        float(
                            context[
                                "load_total_p_mw"
                            ]
                        ),
                    "context_bess_soc":
                        float(
                            context[
                                "bess_soc"
                            ]
                        ),

                    "event_start_s":
                        p[
                            "event_start_s"
                        ],
                    "event_end_s":
                        p[
                            "event_end_s"
                        ],
                    "event_duration_s":
                        p[
                            "event_duration_s"
                        ],

                    "physical_sag_vm_pu":
                        p.get(
                            "physical_sag_vm_pu",
                            np.nan,
                        ),
                    "cloud_min_factor":
                        p.get(
                            "cloud_min_factor",
                            np.nan,
                        ),
                    "cloud_ramp_s":
                        p.get(
                            "cloud_ramp_s",
                            np.nan,
                        ),
                    "gfl_attack_imax_pu":
                        p.get(
                            "gfl_attack_imax_pu",
                            np.nan,
                        ),
                    "estimated_pre_event_gfl_current_pu":
                        p.get(
                            "estimated_pre_event_gfl_current_pu",
                            np.nan,
                        ),
                    "challenge_limit_ratio":
                        p.get(
                            "challenge_limit_ratio",
                            np.nan,
                        ),

                    "gfl_limiter_fraction":
                        limiter_fraction,
                }
            )

            print(
                f"  {accepted+1:02d}/{N_PER_CLASS} "
                f"id={scenario_id} | "
                f"PV={1000*context['pv_p_mw']:.1f} kW | "
                f"Load={1000*context['load_total_p_mw']:.1f} kW | "
                f"SOC={100*context['bess_soc']:.1f}% | "
                f"GFL-limit={100*limiter_fraction:.1f}%"
            )

            scenario_id += 1
            accepted += 1

    raw = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    meta = pd.DataFrame(
        metadata
    )

    raw_stride = max(
        1,
        int(
            round(
                RAW_STEP_MS
                / (
                    CONTROL_DT
                    * 1000.0
                )
            )
        ),
    )

    raw_saved = raw.iloc[
        ::raw_stride
    ].copy()

    return (
        raw_saved,
        meta,
    )


def add_monitor_residuals(
    raw,
    meta,
    sigma_pu,
    gain_sigma,
    delay_ms,
    seed,
):
    meta_idx = meta.set_index(
        "scenario_id"
    )

    master = np.random.default_rng(
        seed
    )

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

        p_ref_seen = stress.shift_command(
            t,
            p_ref,
            delay_ms / 1000.0,
        )

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
            stress.independent_monitor_measurements(
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
                sigma_pu,
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

        mask = (
            (t >= t0)
            & (t < t1)
        )

        r = residual[
            mask
        ]

        rows.append(
            {
                "scenario_id": sid,
                "window_mode": mode,
                "challenge_type":
                    m[
                        "challenge_type"
                    ],
                "event_max_residual_pu":
                    float(
                        np.max(r)
                    ),
                "event_rms_residual_pu":
                    float(
                        np.sqrt(
                            np.mean(
                                r ** 2
                            )
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def baseline_threshold(
    baseline_raw,
    baseline_meta,
    sigma_pu,
    gain_sigma,
    delay_ms,
):
    d = stress.scenario_residuals(
        baseline_raw,
        baseline_meta,
        sigma_pu,
        gain_sigma,
        delay_ms,
        SEED + 900,
    )

    return stress.calibrate_threshold(
        d
    )


def challenge_evaluation(
    residuals,
    threshold,
):
    d = residuals[
        residuals[
            "window_mode"
        ] == "event"
    ].copy()

    d[
        "alarm"
    ] = (
        d[
            "event_max_residual_pu"
        ]
        >= threshold
    )

    return (
        d.groupby(
            "challenge_type",
            as_index=False,
        )
        .agg(
            n=(
                "scenario_id",
                "count",
            ),
            alarm_rate=(
                "alarm",
                "mean",
            ),
            median_max_residual=(
                "event_max_residual_pu",
                "median",
            ),
            p95_max_residual=(
                "event_max_residual_pu",
                lambda x: float(
                    np.percentile(
                        x,
                        95,
                    )
                ),
            ),
        )
    )


def plot_challenge(
    summary,
):
    d = summary.sort_values(
        "alarm_rate",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        d[
            "challenge_type"
        ],
        100*d[
            "alarm_rate"
        ],
    )

    ax.set_xlabel(
        "Residual alarm rate (%)"
    )
    ax.set_xlim(
        0,
        105,
    )
    ax.set_title(
        "Mechanism-Aware Residual Under Ambiguous Hard-Negative Conditions"
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
        / "43_ambiguous_challenge_residual_alarm_rate.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    slow_path = (
        DATA
        / "phase2e_7day_operating_dataset.csv"
    )

    baseline_raw_path = (
        DATA
        / "phase2g_v2_transient_timeseries.csv.gz"
    )

    baseline_meta_path = (
        DATA
        / "phase2g_v2_scenario_metadata.csv"
    )

    for p in [
        slow_path,
        baseline_raw_path,
        baseline_meta_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                str(p)
            )

    print(
        "\n=== PHASE 2M: AMBIGUOUS HARD-NEGATIVE CHALLENGE ==="
    )

    slow = pd.read_csv(
        slow_path,
        parse_dates=[
            "timestamp"
        ],
    )

    contexts = (
        slow[
            slow[
                "event_label"
            ] == "normal"
        ]
        .reset_index(
            drop=True
        )
    )

    raw, meta = (
        build_challenge_dataset(
            contexts
        )
    )

    raw.to_csv(
        DATA
        / "phase2m_ambiguous_challenge_timeseries.csv.gz",
        index=False,
        compression="gzip",
    )

    meta.to_csv(
        DATA
        / "phase2m_ambiguous_challenge_metadata.csv",
        index=False,
    )

    print(
        f"\nChallenge scenarios : {len(meta)}"
    )
    print(
        f"Stored raw rows      : {len(raw):,}"
    )

    print(
        "\n=== CHALLENGE LIMITER AUDIT ==="
    )

    audit = (
        meta.groupby(
            "challenge_type",
            as_index=False,
        )
        .agg(
            n=(
                "scenario_id",
                "count",
            ),
            median_limiter_fraction=(
                "gfl_limiter_fraction",
                "median",
            ),
            min_limiter_fraction=(
                "gfl_limiter_fraction",
                "min",
            ),
            median_limit_ratio=(
                "challenge_limit_ratio",
                "median",
            ),
        )
    )

    print(
        audit.to_string(
            index=False
        )
    )

    # Representative monitor uncertainty:
    # 0.5% additive noise, 0.2% gain sigma, 20 ms command delay.
    sigma = 0.005
    gain_sigma = 0.002
    delay_ms = 20

    baseline_raw = pd.read_csv(
        baseline_raw_path
    )

    baseline_meta = pd.read_csv(
        baseline_meta_path
    )

    threshold = baseline_threshold(
        baseline_raw,
        baseline_meta,
        sigma,
        gain_sigma,
        delay_ms,
    )

    residuals = add_monitor_residuals(
        raw,
        meta,
        sigma,
        gain_sigma,
        delay_ms,
        SEED + 1000,
    )

    challenge_summary = (
        challenge_evaluation(
            residuals,
            threshold,
        )
    )

    challenge_summary[
        "baseline_threshold_pu"
    ] = threshold

    challenge_summary.to_csv(
        RESULTS
        / "phase2m_ambiguous_challenge_residual_summary.csv",
        index=False,
    )

    audit.to_csv(
        RESULTS
        / "phase2m_ambiguous_challenge_limiter_audit.csv",
        index=False,
    )

    plot_challenge(
        challenge_summary
    )

    print(
        "\n=== RESIDUAL CHALLENGE RESULT ==="
    )
    print(
        f"Baseline-calibrated threshold: "
        f"{threshold:.6f} pu"
    )
    print(
        challenge_summary.to_string(
            index=False
        )
    )

    if (
        "cyber_mild_gfl_current_limit"
        in set(
            meta[
                "challenge_type"
            ]
        )
    ):
        x = meta[
            meta[
                "challenge_type"
            ]
            == "cyber_mild_gfl_current_limit"
        ]

        print(
            "\nMild current-limit ratio coverage:"
        )
        print(
            f"  min    = "
            f"{x['challenge_limit_ratio'].min():.3f}"
        )
        print(
            f"  median = "
            f"{x['challenge_limit_ratio'].median():.3f}"
        )
        print(
            f"  max    = "
            f"{x['challenge_limit_ratio'].max():.3f}"
        )

    print(
        "\nSaved:"
    )
    print(
        "  data/phase2m_ambiguous_challenge_timeseries.csv.gz"
    )
    print(
        "  data/phase2m_ambiguous_challenge_metadata.csv"
    )
    print(
        "  results/phase2m_ambiguous_challenge_residual_summary.csv"
    )
    print(
        "  results/phase2m_ambiguous_challenge_limiter_audit.csv"
    )
    print(
        "  figures/43_ambiguous_challenge_residual_alarm_rate.png"
    )

    print(
        "\nDECISION RULE:\n"
        "A credible mechanism-aware residual should still detect a useful "
        "fraction of mild current-limit attacks while avoiding legitimate "
        "deep-sag current limiting and fast PV ramps. If legitimate limiter "
        "events also trigger the residual strongly, the residual cannot be "
        "used as a standalone cyber indicator; it must be fused with network "
        "state/topology evidence. That outcome would directly motivate the "
        "next WLS + graph-aware layer."
    )


if __name__ == "__main__":
    main()
