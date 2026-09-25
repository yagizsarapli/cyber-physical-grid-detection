from pathlib import Path
import argparse
import importlib.util
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp

from pandapower.estimation import estimate
from pandapower.pypower.makeYbus import makeYbus
from scipy.stats import chi2

warnings.filterwarnings("ignore")


# ============================================================
# PHASE 2Q
# NONLINEAR MODEL-CONSISTENT FDIA vs WLS / CHI-SQUARE BDD
# ============================================================
#
# Defensive research purpose
# --------------------------
# This script runs only on the local synthetic GRIDRA 5-bus testbed.
#
# It compares:
#
#   1) clean noisy measurements
#   2) legitimate physical load disturbance
#   3) naive single-sensor corruption
#   4) nonlinear model-consistent localized FDIA
#
# Core construction for the stealth case:
#
#       z_attack = z + h(x + c) - h(x)
#
# where:
#   z       : noisy SCADA measurement vector
#   h(x)    : nonlinear AC measurement model
#   c       : local state displacement at a target load bus
#
# Because the attacked measurements remain mutually consistent with an
# alternative AC state, residual-only WLS bad-data detection can become
# nearly blind even while the estimated state is biased.
#
# This is deliberately a defensive benchmark. It does not connect to,
# target, or provide operational instructions for any real grid.
# ============================================================


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SLOW_PATH = DATA / "phase2e_7day_operating_dataset.csv"

S_BASE_MVA = 0.10
PF_LOAD = 0.95

STD_V_PU = 0.0050
STD_P_MW = 0.0010
STD_Q_MVAR = 0.0008
STD_LINE_P_MW = 0.0012
STD_LINE_Q_MVAR = 0.0010

ALPHA_BDD = 0.05
RANDOM_SEED = 20260812

CASE_ORDER = [
    "clean_noisy",
    "physical_load_disturbance",
    "naive_single_sensor_corruption",
    "nonlinear_model_consistent_fdia",
]

CASE_LABELS = {
    "clean_noisy": "Clean noisy",
    "physical_load_disturbance": "Physical disturbance",
    "naive_single_sensor_corruption": "Naive corruption",
    "nonlinear_model_consistent_fdia": "Model-consistent FDIA",
}


def load_module(filename, module_name):
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required project file not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology = load_module(
    "exploration/01_microgrid_topology.py",
    "topology",
)


def reactive_from_pf(p_mw, pf=PF_LOAD):
    phi = np.arccos(pf)
    return float(
        p_mw * np.tan(phi)
    )


def wrap_deg(x):
    return (
        (np.asarray(x) + 180.0) % 360.0
        - 180.0
    )


def configure_context(
    net,
    context,
    physical_multiplier=None,
    target_bus=None,
):
    """
    Apply one slow operating point to the 5-bus network.

    Optional physical_multiplier changes the real load itself,
    so the resulting measurements remain physically consistent.
    """
    load_a = float(
        context["load_a_p_mw"]
    )
    load_b = float(
        context["load_b_p_mw"]
    )

    bus_load_a = int(
        net.gridra["bus_load_a"]
    )
    bus_load_b = int(
        net.gridra["bus_load_b"]
    )

    if (
        physical_multiplier is not None
        and target_bus is not None
    ):
        if int(target_bus) == bus_load_a:
            load_a *= float(
                physical_multiplier
            )
        elif int(target_bus) == bus_load_b:
            load_b *= float(
                physical_multiplier
            )

    net.load.at[
        net.load.index[0],
        "p_mw",
    ] = load_a
    net.load.at[
        net.load.index[0],
        "q_mvar",
    ] = reactive_from_pf(
        load_a
    )

    net.load.at[
        net.load.index[1],
        "p_mw",
    ] = load_b
    net.load.at[
        net.load.index[1],
        "q_mvar",
    ] = reactive_from_pf(
        load_b
    )

    net.sgen.at[
        net.gridra["pv_sgen"],
        "p_mw",
    ] = float(
        context["pv_p_mw"]
    )
    net.sgen.at[
        net.gridra["pv_sgen"],
        "q_mvar",
    ] = 0.0

    net.sgen.at[
        net.gridra["bess_sgen"],
        "p_mw",
    ] = float(
        context["bess_p_mw"]
    )
    net.sgen.at[
        net.gridra["bess_sgen"],
        "q_mvar",
    ] = 0.0


def solve_truth(net):
    pp.runpp(
        net,
        algorithm="nr",
        calculate_voltage_angles=True,
        init="flat",
        tolerance_mva=1e-9,
        max_iteration=40,
    )

    if not net.converged:
        raise RuntimeError(
            "Truth power flow did not converge."
        )


def build_measurement_model(net):
    """
    Build an AC measurement-function representation from the same
    network topology used by the estimator.

    The measurement vector order is:
      5 bus voltage magnitudes
      5 x (P,Q bus injection)
      4 x (P,Q line from-side flow)

    Total = 23 measurements.
    """
    ppc = net._ppc

    base_mva = float(
        ppc["baseMVA"]
    )

    bus = np.array(
        ppc["bus"],
        copy=True,
    )
    branch = np.array(
        ppc["branch"],
        copy=True,
    )

    (
        ybus,
        yf,
        yt,
    ) = makeYbus(
        base_mva,
        bus,
        branch,
    )

    from_bus = (
        branch[:, 0]
        .astype(int)
    )

    return {
        "base_mva": base_mva,
        "ybus": ybus,
        "yf": yf,
        "from_bus": from_bus,
        "n_bus": len(net.bus),
        "n_line": len(net.line),
    }


def truth_state(net):
    vm = (
        net.res_bus[
            "vm_pu"
        ]
        .to_numpy(
            dtype=float
        )
    )

    va_deg = (
        net.res_bus[
            "va_degree"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return vm, va_deg


def estimated_state(net):
    vm = (
        net.res_bus_est[
            "vm_pu"
        ]
        .to_numpy(
            dtype=float
        )
    )

    va_deg = (
        net.res_bus_est[
            "va_degree"
        ]
        .to_numpy(
            dtype=float
        )
    )

    return vm, va_deg


def h_ac(
    model,
    vm,
    va_deg,
):
    """
    Nonlinear AC measurement function h(x).

    pandapower res_bus uses the consumer sign convention:
    positive P/Q means net consumption.

    V * conj(Ybus V) is network injection, so bus P/Q values are
    negated to match the measurement convention used in Phase 2C.
    """
    vm = np.asarray(
        vm,
        dtype=float,
    )
    va_rad = np.deg2rad(
        np.asarray(
            va_deg,
            dtype=float,
        )
    )

    v = (
        vm
        * np.exp(
            1j * va_rad
        )
    )

    sbus_injection = (
        v
        * np.conj(
            model["ybus"].dot(v)
        )
        * model["base_mva"]
    )

    sf = (
        v[
            model["from_bus"]
        ]
        * np.conj(
            model["yf"].dot(v)
        )
        * model["base_mva"]
    )

    values = []

    # V at all buses.
    for bus in range(
        model["n_bus"]
    ):
        values.append(
            float(vm[bus])
        )

    # P,Q injection at all buses.
    for bus in range(
        model["n_bus"]
    ):
        values.append(
            float(
                -sbus_injection[
                    bus
                ].real
            )
        )
        values.append(
            float(
                -sbus_injection[
                    bus
                ].imag
            )
        )

    # P,Q from-side line flow.
    for line in range(
        model["n_line"]
    ):
        values.append(
            float(
                sf[line].real
            )
        )
        values.append(
            float(
                sf[line].imag
            )
        )

    return np.asarray(
        values,
        dtype=float,
    )


def measurement_schema(net):
    names = []
    stds = []

    for bus in net.bus.index:
        names.append(
            f"V_bus_{int(bus)}"
        )
        stds.append(
            STD_V_PU
        )

    for bus in net.bus.index:
        names.extend(
            [
                f"P_bus_{int(bus)}",
                f"Q_bus_{int(bus)}",
            ]
        )
        stds.extend(
            [
                STD_P_MW,
                STD_Q_MVAR,
            ]
        )

    for line in net.line.index:
        names.extend(
            [
                f"P_line_{int(line)}_from",
                f"Q_line_{int(line)}_from",
            ]
        )
        stds.extend(
            [
                STD_LINE_P_MW,
                STD_LINE_Q_MVAR,
            ]
        )

    return (
        names,
        np.asarray(
            stds,
            dtype=float,
        ),
    )


def add_measurement_vector(
    net,
    z,
    stds,
):
    """
    Create pandapower WLS measurements in exactly the same order
    as h_ac().
    """
    if len(net.measurement):
        net.measurement.drop(
            net.measurement.index,
            inplace=True,
        )

    k = 0

    for bus in net.bus.index:
        pp.create_measurement(
            net,
            meas_type="v",
            element_type="bus",
            value=float(z[k]),
            std_dev=float(stds[k]),
            element=int(bus),
            name=f"V_bus_{int(bus)}",
        )
        k += 1

    for bus in net.bus.index:
        pp.create_measurement(
            net,
            meas_type="p",
            element_type="bus",
            value=float(z[k]),
            std_dev=float(stds[k]),
            element=int(bus),
            name=f"P_bus_{int(bus)}",
        )
        k += 1

        pp.create_measurement(
            net,
            meas_type="q",
            element_type="bus",
            value=float(z[k]),
            std_dev=float(stds[k]),
            element=int(bus),
            name=f"Q_bus_{int(bus)}",
        )
        k += 1

    for line in net.line.index:
        pp.create_measurement(
            net,
            meas_type="p",
            element_type="line",
            value=float(z[k]),
            std_dev=float(stds[k]),
            element=int(line),
            side="from",
            name=f"P_line_{int(line)}_from",
        )
        k += 1

        pp.create_measurement(
            net,
            meas_type="q",
            element_type="line",
            value=float(z[k]),
            std_dev=float(stds[k]),
            element=int(line),
            side="from",
            name=f"Q_line_{int(line)}_from",
        )
        k += 1

    if k != len(z):
        raise RuntimeError(
            "Measurement-vector length mismatch."
        )


def run_wls(
    net,
    model,
    z,
    stds,
):
    add_measurement_vector(
        net,
        z,
        stds,
    )

    success = bool(
        estimate(
            net,
            algorithm="wls",
            init="flat",
            tolerance=1e-7,
            maximum_iterations=40,
            calculate_voltage_angles=True,
        )
    )

    if not success:
        return None

    vm_hat, va_hat = (
        estimated_state(net)
    )

    h_hat = h_ac(
        model,
        vm_hat,
        va_hat,
    )

    residual = (
        np.asarray(
            z,
            dtype=float,
        )
        - h_hat
    )

    norm_residual = (
        residual
        / stds
    )

    j_stat = float(
        np.sum(
            norm_residual ** 2
        )
    )

    n_measurements = len(z)

    # 5 voltage magnitudes + 4 non-reference angles.
    n_states = (
        2
        * model["n_bus"]
        - 1
    )

    dof = (
        n_measurements
        - n_states
    )

    threshold = float(
        chi2.ppf(
            1.0 - ALPHA_BDD,
            dof,
        )
    )

    p_value = float(
        1.0
        - chi2.cdf(
            j_stat,
            dof,
        )
    )

    return {
        "vm_hat": vm_hat,
        "va_hat": va_hat,
        "h_hat": h_hat,
        "residual": residual,
        "norm_residual": (
            norm_residual
        ),
        "j_stat": j_stat,
        "chi2_threshold": threshold,
        "chi2_p_value": p_value,
        "bdd_flag": bool(
            j_stat > threshold
        ),
        "max_abs_norm_residual": float(
            np.max(
                np.abs(
                    norm_residual
                )
            )
        ),
        "rms_norm_residual": float(
            np.sqrt(
                np.mean(
                    norm_residual ** 2
                )
            )
        ),
    }


def apply_naive_attack(
    z,
    names,
    target_bus,
    rng,
):
    z_attack = np.array(
        z,
        copy=True,
    )

    name = (
        f"V_bus_{int(target_bus)}"
    )

    idx = names.index(
        name
    )

    sign = float(
        rng.choice(
            [-1.0, 1.0]
        )
    )

    magnitude = float(
        rng.uniform(
            0.035,
            0.060,
        )
    )

    delta = (
        sign
        * magnitude
    )

    z_attack[
        idx
    ] += delta

    attack_vector = np.zeros(
        len(z),
        dtype=float,
    )
    attack_vector[
        idx
    ] = delta

    return (
        z_attack,
        attack_vector,
        {
            "naive_sensor":
                name,
            "attack_dv_pu":
                delta,
            "attack_da_deg":
                0.0,
        },
    )


def apply_model_consistent_attack(
    z,
    h_true,
    model,
    vm_true,
    va_true,
    target_bus,
    rng,
):
    """
    Exact nonlinear AC model-consistent attack:

        a = h(x_attack) - h(x_true)
        z_attack = z + a

    Only measurements whose nonlinear value changes are effectively
    compromised. For a leaf-bus state perturbation this remains local
    to that bus, its adjacent branch, and neighboring injection.
    """
    vm_attack = np.array(
        vm_true,
        copy=True,
    )
    va_attack = np.array(
        va_true,
        copy=True,
    )

    dv = -float(
        rng.uniform(
            0.012,
            0.035,
        )
    )

    da = float(
        rng.choice(
            [-1.0, 1.0]
        )
        * rng.uniform(
            0.40,
            2.20,
        )
    )

    vm_attack[
        target_bus
    ] = max(
        0.90,
        vm_attack[
            target_bus
        ]
        + dv,
    )

    va_attack[
        target_bus
    ] += da

    h_attack = h_ac(
        model,
        vm_attack,
        va_attack,
    )

    attack_vector = (
        h_attack
        - h_true
    )

    z_attack = (
        np.asarray(
            z,
            dtype=float,
        )
        + attack_vector
    )

    return (
        z_attack,
        attack_vector,
        {
            "attack_dv_pu":
                float(
                    vm_attack[
                        target_bus
                    ]
                    - vm_true[
                        target_bus
                    ]
                ),
            "attack_da_deg":
                float(da),
            "target_vm_fictitious_pu":
                float(
                    vm_attack[
                        target_bus
                    ]
                ),
            "target_va_fictitious_deg":
                float(
                    va_attack[
                        target_bus
                    ]
                ),
        },
    )


def build_case_network(
    context,
    case_name,
    target_bus,
    physical_multiplier,
):
    net = (
        topology
        .build_microgrid()
    )

    if (
        case_name
        == "physical_load_disturbance"
    ):
        configure_context(
            net,
            context,
            physical_multiplier=(
                physical_multiplier
            ),
            target_bus=target_bus,
        )
    else:
        configure_context(
            net,
            context,
        )

    solve_truth(
        net
    )

    return net


def one_case(
    replication,
    case_name,
    context,
    target_bus,
    physical_multiplier,
    standardized_noise,
    rng,
):
    net = build_case_network(
        context,
        case_name,
        target_bus,
        physical_multiplier,
    )

    model = (
        build_measurement_model(
            net
        )
    )

    vm_true, va_true = (
        truth_state(
            net
        )
    )

    h_true = h_ac(
        model,
        vm_true,
        va_true,
    )

    names, stds = (
        measurement_schema(
            net
        )
    )

    if len(
        standardized_noise
    ) != len(stds):
        raise RuntimeError(
            "Noise vector length mismatch."
        )

    z_clean = (
        h_true
        + stds
        * standardized_noise
    )

    attack_vector = np.zeros(
        len(z_clean),
        dtype=float,
    )

    attack_meta = {
        "attack_dv_pu": 0.0,
        "attack_da_deg": 0.0,
    }

    z_used = np.array(
        z_clean,
        copy=True,
    )

    if (
        case_name
        == "naive_single_sensor_corruption"
    ):
        (
            z_used,
            attack_vector,
            attack_meta,
        ) = apply_naive_attack(
            z_clean,
            names,
            target_bus,
            rng,
        )

    elif (
        case_name
        == "nonlinear_model_consistent_fdia"
    ):
        (
            z_used,
            attack_vector,
            attack_meta,
        ) = (
            apply_model_consistent_attack(
                z_clean,
                h_true,
                model,
                vm_true,
                va_true,
                target_bus,
                rng,
            )
        )

    result = run_wls(
        net,
        model,
        z_used,
        stds,
    )

    if result is None:
        return None

    vm_hat = result[
        "vm_hat"
    ]
    va_hat = result[
        "va_hat"
    ]

    v_error = (
        vm_hat
        - vm_true
    )

    a_error = wrap_deg(
        va_hat
        - va_true
    )

    attack_sigma = (
        np.abs(
            attack_vector
        )
        / stds
    )

    # Count channels materially altered by the constructed attack.
    compromised = (
        attack_sigma
        > 1e-3
    )

    row = {
        "replication":
            int(replication),
        "case":
            case_name,
        "case_label":
            CASE_LABELS[
                case_name
            ],
        "context_timestamp":
            context["timestamp"],
        "context_pv_kw":
            1000.0
            * float(
                context[
                    "pv_p_mw"
                ]
            ),
        "context_load_kw":
            1000.0
            * float(
                context[
                    "load_total_p_mw"
                ]
            ),
        "context_soc_pct":
            100.0
            * float(
                context[
                    "bess_soc"
                ]
            ),
        "target_bus":
            int(target_bus),
        "physical_load_multiplier":
            (
                float(
                    physical_multiplier
                )
                if (
                    case_name
                    == "physical_load_disturbance"
                )
                else 1.0
            ),
        "wls_converged":
            True,
        "chi2_j":
            result[
                "j_stat"
            ],
        "chi2_threshold":
            result[
                "chi2_threshold"
            ],
        "chi2_p_value":
            result[
                "chi2_p_value"
            ],
        "bdd_flag":
            result[
                "bdd_flag"
            ],
        "max_abs_norm_residual":
            result[
                "max_abs_norm_residual"
            ],
        "rms_norm_residual":
            result[
                "rms_norm_residual"
            ],
        "voltage_rmse_pu":
            float(
                np.sqrt(
                    np.mean(
                        v_error ** 2
                    )
                )
            ),
        "angle_rmse_deg":
            float(
                np.sqrt(
                    np.mean(
                        a_error ** 2
                    )
                )
            ),
        "target_true_vm_pu":
            float(
                vm_true[
                    target_bus
                ]
            ),
        "target_est_vm_pu":
            float(
                vm_hat[
                    target_bus
                ]
            ),
        "target_voltage_bias_pu":
            float(
                vm_hat[
                    target_bus
                ]
                - vm_true[
                    target_bus
                ]
            ),
        "target_true_va_deg":
            float(
                va_true[
                    target_bus
                ]
            ),
        "target_est_va_deg":
            float(
                va_hat[
                    target_bus
                ]
            ),
        "target_angle_bias_deg":
            float(
                wrap_deg(
                    vm_hat[
                        target_bus
                    ]
                    * 0.0
                    + va_hat[
                        target_bus
                    ]
                    - va_true[
                        target_bus
                    ]
                )
            ),
        "n_materially_compromised_measurements":
            int(
                np.sum(
                    compromised
                )
            ),
        "max_attack_sigma":
            float(
                np.max(
                    attack_sigma
                )
            ),
        "rms_attack_sigma":
            float(
                np.sqrt(
                    np.mean(
                        attack_sigma ** 2
                    )
                )
            ),
    }

    row.update(
        attack_meta
    )

    return row


def summarize(df):
    rows = []

    for case_name in CASE_ORDER:
        g = df[
            df[
                "case"
            ]
            == case_name
        ]

        if g.empty:
            continue

        rows.append(
            {
                "case":
                    case_name,
                "case_label":
                    CASE_LABELS[
                        case_name
                    ],
                "n":
                    int(
                        len(g)
                    ),
                "bdd_flag_rate":
                    float(
                        g[
                            "bdd_flag"
                        ].mean()
                    ),
                "bdd_pass_rate":
                    float(
                        1.0
                        - g[
                            "bdd_flag"
                        ].mean()
                    ),
                "median_chi2_j":
                    float(
                        g[
                            "chi2_j"
                        ].median()
                    ),
                "p95_chi2_j":
                    float(
                        g[
                            "chi2_j"
                        ].quantile(
                            0.95
                        )
                    ),
                "median_max_norm_residual":
                    float(
                        g[
                            "max_abs_norm_residual"
                        ].median()
                    ),
                "median_voltage_rmse_pu":
                    float(
                        g[
                            "voltage_rmse_pu"
                        ].median()
                    ),
                "median_angle_rmse_deg":
                    float(
                        g[
                            "angle_rmse_deg"
                        ].median()
                    ),
                "median_abs_target_voltage_bias_pu":
                    float(
                        g[
                            "target_voltage_bias_pu"
                        ]
                        .abs()
                        .median()
                    ),
                "median_abs_target_angle_bias_deg":
                    float(
                        g[
                            "target_angle_bias_deg"
                        ]
                        .abs()
                        .median()
                    ),
                "median_compromised_measurements":
                    float(
                        g[
                            "n_materially_compromised_measurements"
                        ]
                        .median()
                    ),
                "median_max_attack_sigma":
                    float(
                        g[
                            "max_attack_sigma"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def paired_stealth_analysis(df):
    """
    Directly compare clean vs stealth cases within the same context/noise
    replication. This is the cleanest evidence of BDD blindness.
    """
    cols = [
        "replication",
        "chi2_j",
        "bdd_flag",
        "voltage_rmse_pu",
        "angle_rmse_deg",
        "target_voltage_bias_pu",
        "target_angle_bias_deg",
    ]

    clean = (
        df[
            df[
                "case"
            ]
            == "clean_noisy"
        ][cols]
        .copy()
    )

    stealth = (
        df[
            df[
                "case"
            ]
            == "nonlinear_model_consistent_fdia"
        ][cols]
        .copy()
    )

    clean = clean.rename(
        columns={
            c: f"clean_{c}"
            for c in cols
            if c != "replication"
        }
    )

    stealth = stealth.rename(
        columns={
            c: f"stealth_{c}"
            for c in cols
            if c != "replication"
        }
    )

    paired = clean.merge(
        stealth,
        on="replication",
        how="inner",
    )

    paired[
        "delta_chi2_j"
    ] = (
        paired[
            "stealth_chi2_j"
        ]
        - paired[
            "clean_chi2_j"
        ]
    )

    paired[
        "stealth_bdd_passes_but_state_biased"
    ] = (
        ~paired[
            "stealth_bdd_flag"
        ].astype(bool)
        & (
            paired[
                "stealth_target_voltage_bias_pu"
            ].abs()
            > 0.008
        )
    )

    return paired


def plot_bdd_rates(summary):
    ordered = (
        summary
        .set_index(
            "case"
        )
        .loc[
            CASE_ORDER
        ]
        .reset_index()
    )

    x = np.arange(
        len(ordered)
    )

    rates = (
        100.0
        * ordered[
            "bdd_flag_rate"
        ].to_numpy()
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6.2)
    )

    ax.bar(
        x,
        rates,
    )

    ax.axhline(
        100.0 * ALPHA_BDD,
        linestyle="--",
        linewidth=1.3,
        label=(
            "Nominal 5% false-alarm level"
        ),
    )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        ordered[
            "case_label"
        ],
        rotation=20,
        ha="right",
    )
    ax.set_ylabel(
        "WLS χ² BDD flag rate (%)"
    )
    ax.set_title(
        "Residual-Based BDD: Obvious Corruption vs Model-Consistent FDIA"
    )
    ax.set_ylim(
        0,
        105,
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
        / "45_wls_bdd_stealth_gap.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_j_distributions(df):
    fig, ax = plt.subplots(
        figsize=(11, 6.4)
    )

    positions = (
        np.arange(
            len(
                CASE_ORDER
            )
        )
    )

    data = [
        df[
            df[
                "case"
            ]
            == case
        ][
            "chi2_j"
        ].to_numpy()
        for case in CASE_ORDER
    ]

    ax.boxplot(
        data,
        positions=positions,
        showfliers=False,
    )

    threshold = float(
        df[
            "chi2_threshold"
        ].iloc[0]
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.3,
        label=(
            "χ² decision threshold"
        ),
    )

    ax.set_xticks(
        positions
    )
    ax.set_xticklabels(
        [
            CASE_LABELS[
                c
            ]
            for c in CASE_ORDER
        ],
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Weighted residual statistic J"
    )
    ax.set_title(
        "Model-Consistent FDIA Preserves the Residual Distribution"
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
        / "46_chi2_residual_distribution.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_state_bias_vs_bdd(df):
    fig, ax = plt.subplots(
        figsize=(9.8, 6.3)
    )

    for case in [
        "clean_noisy",
        "naive_single_sensor_corruption",
        "nonlinear_model_consistent_fdia",
    ]:
        g = df[
            df[
                "case"
            ]
            == case
        ]

        ax.scatter(
            100.0
            * g[
                "target_voltage_bias_pu"
            ].abs(),
            g[
                "chi2_j"
            ],
            s=28,
            alpha=0.65,
            label=CASE_LABELS[
                case
            ],
        )

    threshold = float(
        df[
            "chi2_threshold"
        ].iloc[0]
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.3,
        label="χ² threshold",
    )

    ax.set_xlabel(
        "Absolute target-bus estimation bias (% pu)"
    )
    ax.set_ylabel(
        "Weighted residual statistic J"
    )
    ax.set_title(
        "State Bias Can Grow While Residual BDD Remains Below Threshold"
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
        / "47_state_bias_vs_bdd_statistic.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def print_interpretation(
    summary,
    paired,
):
    print(
        "\n=== INTERPRETATION ==="
    )

    s = (
        summary
        .set_index(
            "case"
        )
    )

    for case in CASE_ORDER:
        if case not in s.index:
            continue

        row = s.loc[
            case
        ]

        print(
            f"{CASE_LABELS[case]:28s} | "
            f"BDD flag={100*row['bdd_flag_rate']:.1f}% | "
            f"median J={row['median_chi2_j']:.2f} | "
            f"median |Vtarget bias|="
            f"{100*row['median_abs_target_voltage_bias_pu']:.2f}% pu"
        )

    if not paired.empty:
        blindness = float(
            paired[
                "stealth_bdd_passes_but_state_biased"
            ].mean()
        )

        print(
            "\nMatched clean-vs-stealth evidence:"
        )
        print(
            "  Median stealth-minus-clean J change : "
            f"{paired['delta_chi2_j'].median():.3f}"
        )
        print(
            "  BDD passes despite >0.8% target-state bias: "
            f"{100*blindness:.1f}% of matched stealth cases"
        )

    print(
        "\nSCIENTIFIC DECISION RULE:"
    )
    print(
        "If naive corruption is detected strongly while the nonlinear "
        "model-consistent FDIA produces a BDD flag rate close to clean "
        "noise despite clear state bias, we have demonstrated the exact "
        "failure mode required for the next phase. The next detector must "
        "therefore use information that the residual-only estimator does "
        "not contain: protected command consistency, temporal inverter "
        "dynamics, and topology-aware cross-device evidence."
    )

    print(
        "\nIMPORTANT LIMIT:"
    )
    print(
        "The stealth attack assumes the attacker can alter every local "
        "measurement affected by the chosen fictitious state. This is a "
        "synthetic observability benchmark, not a claim that the same "
        "measurement access exists in a real utility network."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-per-class",
        type=int,
        default=80,
        help=(
            "Matched replications per case. "
            "Default: 80"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    args = parser.parse_args()

    if not SLOW_PATH.exists():
        raise FileNotFoundError(
            f"Missing slow operating dataset: {SLOW_PATH}"
        )

    slow = pd.read_csv(
        SLOW_PATH
    )

    slow[
        "timestamp"
    ] = pd.to_datetime(
        slow[
            "timestamp"
        ]
    )

    if "powerflow_ok" in slow.columns:
        slow = slow[
            slow[
                "powerflow_ok"
            ].astype(bool)
        ].copy()

    if "event_label" in slow.columns:
        normal = slow[
            slow[
                "event_label"
            ].astype(str)
            == "normal"
        ]

        if len(normal) >= args.n_per_class:
            slow = normal.copy()

    slow = slow.reset_index(
        drop=True
    )

    if slow.empty:
        raise RuntimeError(
            "No usable operating contexts."
        )

    rng = np.random.default_rng(
        args.seed
    )

    print(
        "\n=== PHASE 2Q: NONLINEAR STEALTH FDIA vs WLS χ² BDD ==="
    )
    print(
        f"Matched replications per case : {args.n_per_class}"
    )
    print(
        f"Cases                         : {len(CASE_ORDER)}"
    )
    print(
        f"Expected successful rows      : "
        f"{args.n_per_class * len(CASE_ORDER)}"
    )
    print(
        f"BDD alpha                     : {ALPHA_BDD:.2f}"
    )

    rows = []

    # Establish measurement dimension once.
    probe = topology.build_microgrid()
    configure_context(
        probe,
        slow.iloc[0],
    )
    solve_truth(
        probe
    )
    _, probe_stds = (
        measurement_schema(
            probe
        )
    )

    n_meas = len(
        probe_stds
    )

    load_buses = [
        int(
            probe.gridra[
                "bus_load_a"
            ]
        ),
        int(
            probe.gridra[
                "bus_load_b"
            ]
        ),
    ]

    for rep in range(
        args.n_per_class
    ):
        context = slow.iloc[
            int(
                rng.integers(
                    0,
                    len(slow),
                )
            )
        ]

        target_bus = int(
            rng.choice(
                load_buses
            )
        )

        physical_multiplier = float(
            rng.uniform(
                1.25,
                1.70,
            )
        )

        # Same standardized noise across all four matched cases.
        eps = rng.normal(
            0.0,
            1.0,
            size=n_meas,
        )

        print(
            f"  {rep+1:3d}/{args.n_per_class} | "
            f"context={context['timestamp']} | "
            f"PV={1000*float(context['pv_p_mw']):.1f} kW | "
            f"Load={1000*float(context['load_total_p_mw']):.1f} kW | "
            f"target Bus {target_bus}"
        )

        for case_name in CASE_ORDER:
            try:
                row = one_case(
                    replication=rep,
                    case_name=case_name,
                    context=context,
                    target_bus=target_bus,
                    physical_multiplier=(
                        physical_multiplier
                    ),
                    standardized_noise=eps,
                    rng=rng,
                )
            except Exception as exc:
                print(
                    f"    WARNING {case_name}: {exc}"
                )
                row = None

            if row is not None:
                rows.append(
                    row
                )

    df = pd.DataFrame(
        rows
    )

    if df.empty:
        raise RuntimeError(
            "No successful WLS cases."
        )

    counts = (
        df[
            "case"
        ]
        .value_counts()
    )

    print(
        "\nSuccessful cases:"
    )
    print(
        counts.to_string()
    )

    summary = summarize(
        df
    )

    paired = (
        paired_stealth_analysis(
            df
        )
    )

    df.to_csv(
        RESULTS
        / "phase2q_stealth_fdia_trials.csv",
        index=False,
    )

    summary.to_csv(
        RESULTS
        / "phase2q_stealth_fdia_summary.csv",
        index=False,
    )

    paired.to_csv(
        RESULTS
        / "phase2q_matched_clean_vs_stealth.csv",
        index=False,
    )

    plot_bdd_rates(
        summary
    )
    plot_j_distributions(
        df
    )
    plot_state_bias_vs_bdd(
        df
    )

    print(
        "\n=== SUMMARY ==="
    )
    print(
        summary.to_string(
            index=False
        )
    )

    print_interpretation(
        summary,
        paired,
    )

    print(
        "\nSaved:"
    )
    print(
        "  results/phase2q_stealth_fdia_trials.csv"
    )
    print(
        "  results/phase2q_stealth_fdia_summary.csv"
    )
    print(
        "  results/phase2q_matched_clean_vs_stealth.csv"
    )
    print(
        "  figures/45_wls_bdd_stealth_gap.png"
    )
    print(
        "  figures/46_chi2_residual_distribution.png"
    )
    print(
        "  figures/47_state_bias_vs_bdd_statistic.png"
    )


if __name__ == "__main__":
    main()
