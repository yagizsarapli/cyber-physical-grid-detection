from pathlib import Path
import argparse
import importlib.util
import warnings
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PHASE 2R
# GRAPH-READY WLS + PROTECTED-PRIOR TELEMETRY
# ============================================================
#
# Defensive research purpose
# --------------------------
# This phase converts the synthetic 5-bus microgrid benchmark into
# graph-structured node/edge telemetry suitable for later topology-
# aware detection and localization.
#
# Important scientific point:
# Phase 2Q showed that residual-only WLS / chi-square BDD can be blind
# to nonlinear model-consistent FDIAs. A graph model fed ONLY those
# same residuals cannot magically recover information that is absent.
#
# Therefore this dataset adds a deliberately separate information source:
#
#   protected operating prior
#
# constructed from:
#   - authorized PV/BESS operating commands
#   - uncertain short-horizon load forecast
#   - physical network model
#
# Features include:
#   WLS estimated state
#   local normalized residuals
#   prior-vs-WLS innovations
#   topology / device-role metadata
#
# Labels include:
#   attack target bus
#   physical disturbance target bus
#   compromised channels
#
# LABEL COLUMNS ARE NOT DETECTOR FEATURES.
#
# This is still a synthetic reduced-order research benchmark, not an
# operational utility-grid attack or deployment model.
# ============================================================


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SLOW_PATH = DATA / "phase2e_7day_operating_dataset.csv"

RANDOM_SEED = 20260812

CASES = [
    "clean_noisy",
    "physical_load_disturbance",
    "naive_single_sensor_corruption",
    "nonlinear_model_consistent_fdia",
]

CASE_LABEL = {
    "clean_noisy": "Clean noisy",
    "physical_load_disturbance": "Physical disturbance",
    "naive_single_sensor_corruption": "Naive corruption",
    "nonlinear_model_consistent_fdia": "Model-consistent FDIA",
}

CASE_FAMILY = {
    "clean_noisy": "normal",
    "physical_load_disturbance": "physical",
    "naive_single_sensor_corruption": "cyber",
    "nonlinear_model_consistent_fdia": "cyber",
}

# Practical normalization scales for prior innovations.
# These are feature scalings, not bad-data thresholds.
SCALE_VM_PU = 0.005
SCALE_VA_DEG = 0.20
SCALE_P_MW = 0.003
SCALE_Q_MVAR = 0.003
SCALE_LINE_P_MW = 0.003
SCALE_LINE_Q_MVAR = 0.003


def load_module(filename, module_name):
    path = ROOT / filename
    if not path.exists():
        raise FileNotFoundError(f"Required project file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


phase21 = load_module(
    "21_nonlinear_stealth_fdia_bdd_benchmark.py",
    "phase21",
)
topology = phase21.topology


def wrap_deg_scalar(x):
    return float((float(x) + 180.0) % 360.0 - 180.0)


def make_replication_split(n_rep, seed):
    rng = np.random.default_rng(seed + 9173)
    reps = np.arange(n_rep, dtype=int)
    rng.shuffle(reps)

    n_train = int(round(0.70 * n_rep))
    n_val = int(round(0.15 * n_rep))
    n_train = min(max(n_train, 1), n_rep)
    n_val = min(max(n_val, 1), max(n_rep - n_train, 0))

    split = {}
    for r in reps[:n_train]:
        split[int(r)] = "train"
    for r in reps[n_train:n_train + n_val]:
        split[int(r)] = "validation"
    for r in reps[n_train + n_val:]:
        split[int(r)] = "test"

    return split


def device_role(net, bus):
    g = net.gridra
    if bus == int(g["bus_pcc"]):
        return "pcc"
    if bus == int(g["bus_pv"]):
        return "pv_gfl"
    if bus == int(g["bus_load_a"]):
        return "load_a"
    if bus == int(g["bus_bess"]):
        return "bess_gfm"
    if bus == int(g["bus_load_b"]):
        return "load_b"
    return "other"


def graph_adjacency(net):
    adj = {int(b): set() for b in net.bus.index}
    for _, line in net.line.iterrows():
        f = int(line["from_bus"])
        t = int(line["to_bus"])
        adj[f].add(t)
        adj[t].add(f)
    return adj


def hop_distance(adj, source, target):
    if source < 0 or target < 0:
        return -1
    if source == target:
        return 0
    q = deque([(source, 0)])
    seen = {source}
    while q:
        node, d = q.popleft()
        for nb in adj[node]:
            if nb == target:
                return d + 1
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return -1


def load_slow_dataset():
    if not SLOW_PATH.exists():
        raise FileNotFoundError(f"Missing operating dataset: {SLOW_PATH}")

    slow = pd.read_csv(SLOW_PATH)
    slow["timestamp"] = pd.to_datetime(slow["timestamp"])

    if "powerflow_ok" in slow.columns:
        slow = slow[slow["powerflow_ok"].astype(bool)].copy()

    if "event_label" in slow.columns:
        normal = slow[slow["event_label"].astype(str) == "normal"].copy()
        if not normal.empty:
            slow = normal

    slow = slow.reset_index(drop=True)

    if slow.empty:
        raise RuntimeError("No usable slow operating contexts.")

    return slow


def prior_context_from_actual(context, rng):
    """
    Protected prior:
      - PV and BESS setpoints are treated as authorized/protected commands.
      - Load is NOT known perfectly; each load gets a small forecast error.

    The same prior uncertainty is used across all matched cases in one
    replication, so case comparisons remain fair.
    """
    c = context.to_dict() if hasattr(context, "to_dict") else dict(context)

    ea = float(np.clip(rng.normal(0.0, 0.025), -0.06, 0.06))
    eb = float(np.clip(rng.normal(0.0, 0.025), -0.06, 0.06))

    c["load_a_p_mw"] = max(0.0, float(c["load_a_p_mw"]) * (1.0 + ea))
    c["load_b_p_mw"] = max(0.0, float(c["load_b_p_mw"]) * (1.0 + eb))
    c["load_total_p_mw"] = c["load_a_p_mw"] + c["load_b_p_mw"]

    return c, ea, eb


def solve_prior(prior_context):
    net = topology.build_microgrid()
    phase21.configure_context(net, prior_context)
    phase21.solve_truth(net)

    model = phase21.build_measurement_model(net)
    vm, va = phase21.truth_state(net)
    h = phase21.h_ac(model, vm, va)

    return {
        "net": net,
        "model": model,
        "vm": vm,
        "va": va,
        "h": h,
    }


def construct_case(
    context,
    case_name,
    target_bus,
    physical_multiplier,
    standardized_noise,
    rng,
):
    """
    Reproduce the Phase-2Q measurement construction, but retain the
    detailed vectors required for node/edge graph telemetry.
    """
    net = phase21.build_case_network(
        context,
        case_name,
        target_bus,
        physical_multiplier,
    )

    model = phase21.build_measurement_model(net)
    vm_true, va_true = phase21.truth_state(net)
    h_true = phase21.h_ac(model, vm_true, va_true)
    names, stds = phase21.measurement_schema(net)

    z_clean = h_true + stds * standardized_noise
    z_used = np.array(z_clean, copy=True)
    attack_vector = np.zeros_like(z_used)

    attack_meta = {
        "attack_dv_pu": 0.0,
        "attack_da_deg": 0.0,
    }

    if case_name == "naive_single_sensor_corruption":
        z_used, attack_vector, attack_meta = phase21.apply_naive_attack(
            z_clean,
            names,
            target_bus,
            rng,
        )
    elif case_name == "nonlinear_model_consistent_fdia":
        z_used, attack_vector, attack_meta = phase21.apply_model_consistent_attack(
            z_clean,
            h_true,
            model,
            vm_true,
            va_true,
            target_bus,
            rng,
        )

    result = phase21.run_wls(
        net,
        model,
        z_used,
        stds,
    )

    if result is None:
        return None

    return {
        "net": net,
        "model": model,
        "vm_true": vm_true,
        "va_true": va_true,
        "h_true": h_true,
        "names": names,
        "stds": stds,
        "z_clean": z_clean,
        "z_used": z_used,
        "attack_vector": attack_vector,
        "attack_meta": attack_meta,
        "result": result,
    }


def measurement_positions(n_bus, n_line):
    pos = {
        "v": {},
        "p_bus": {},
        "q_bus": {},
        "p_line": {},
        "q_line": {},
    }

    k = 0
    for b in range(n_bus):
        pos["v"][b] = k
        k += 1

    for b in range(n_bus):
        pos["p_bus"][b] = k
        k += 1
        pos["q_bus"][b] = k
        k += 1

    for line in range(n_line):
        pos["p_line"][line] = k
        k += 1
        pos["q_line"][line] = k
        k += 1

    return pos


def node_innovation_score(dv, da, dp, dq):
    return float(np.sqrt(
        (dv / SCALE_VM_PU) ** 2
        + (da / SCALE_VA_DEG) ** 2
        + (dp / SCALE_P_MW) ** 2
        + (dq / SCALE_Q_MVAR) ** 2
    ))


def node_residual_score(rv, rp, rq):
    return float(np.sqrt(rv ** 2 + rp ** 2 + rq ** 2))


def edge_innovation_score(dp, dq):
    return float(np.sqrt(
        (dp / SCALE_LINE_P_MW) ** 2
        + (dq / SCALE_LINE_Q_MVAR) ** 2
    ))


def extract_graph_rows(
    replication,
    split_name,
    case_name,
    context,
    target_bus,
    physical_multiplier,
    prior_pack,
    prior_error_a,
    prior_error_b,
    case_pack,
):
    net = case_pack["net"]
    result = case_pack["result"]

    n_bus = len(net.bus)
    n_line = len(net.line)
    pos = measurement_positions(n_bus, n_line)

    graph_id = f"rep{replication:04d}_{case_name}"

    vm_hat = np.asarray(result["vm_hat"], dtype=float)
    va_hat = np.asarray(result["va_hat"], dtype=float)
    h_hat = np.asarray(result["h_hat"], dtype=float)
    norm_res = np.asarray(result["norm_residual"], dtype=float)

    vm_true = np.asarray(case_pack["vm_true"], dtype=float)
    va_true = np.asarray(case_pack["va_true"], dtype=float)
    h_true = np.asarray(case_pack["h_true"], dtype=float)

    z = np.asarray(case_pack["z_used"], dtype=float)
    attack = np.asarray(case_pack["attack_vector"], dtype=float)
    stds = np.asarray(case_pack["stds"], dtype=float)
    attack_sigma = np.abs(attack) / stds

    prior_vm = np.asarray(prior_pack["vm"], dtype=float)
    prior_va = np.asarray(prior_pack["va"], dtype=float)
    prior_h = np.asarray(prior_pack["h"], dtype=float)

    adj = graph_adjacency(net)

    is_cyber = CASE_FAMILY[case_name] == "cyber"
    is_physical = CASE_FAMILY[case_name] == "physical"

    attack_target_bus = int(target_bus) if is_cyber else -1
    physical_target_bus = int(target_bus) if is_physical else -1

    node_rows = []

    for b in range(n_bus):
        iv = pos["v"][b]
        ip = pos["p_bus"][b]
        iq = pos["q_bus"][b]

        prior_p = float(prior_h[ip])
        prior_q = float(prior_h[iq])
        est_p = float(h_hat[ip])
        est_q = float(h_hat[iq])

        dv_prior = float(vm_hat[b] - prior_vm[b])
        da_prior = wrap_deg_scalar(va_hat[b] - prior_va[b])
        dp_prior = float(est_p - prior_p)
        dq_prior = float(est_q - prior_q)

        dv_truth = float(vm_hat[b] - vm_true[b])
        da_truth = wrap_deg_scalar(va_hat[b] - va_true[b])

        rv = float(norm_res[iv])
        rp = float(norm_res[ip])
        rq = float(norm_res[iq])

        inv_score = node_innovation_score(
            dv_prior, da_prior, dp_prior, dq_prior
        )
        res_score = node_residual_score(rv, rp, rq)

        role = device_role(net, b)

        node_rows.append({
            "graph_id": graph_id,
            "replication": int(replication),
            "dataset_split": split_name,
            "case": case_name,
            "case_label": CASE_LABEL[case_name],
            "family": CASE_FAMILY[case_name],
            "context_timestamp": context["timestamp"],
            "context_pv_kw": 1000.0 * float(context["pv_p_mw"]),
            "context_load_kw": 1000.0 * float(context["load_total_p_mw"]),
            "context_soc_pct": 100.0 * float(context["bess_soc"]),
            "prior_load_a_forecast_error_pct": 100.0 * prior_error_a,
            "prior_load_b_forecast_error_pct": 100.0 * prior_error_b,

            "node_id": int(b),
            "node_name": str(net.bus.at[b, "name"]),
            "node_role": role,
            "degree": int(len(adj[b])),

            "role_pcc": int(role == "pcc"),
            "role_pv_gfl": int(role == "pv_gfl"),
            "role_load": int(role in ("load_a", "load_b")),
            "role_bess_gfm": int(role == "bess_gfm"),

            # Protected-prior / WLS features
            "prior_vm_pu": float(prior_vm[b]),
            "prior_va_deg": float(prior_va[b]),
            "prior_p_mw": prior_p,
            "prior_q_mvar": prior_q,

            "wls_vm_pu": float(vm_hat[b]),
            "wls_va_deg": float(va_hat[b]),
            "wls_p_mw": est_p,
            "wls_q_mvar": est_q,

            "innovation_vm_pu": dv_prior,
            "innovation_va_deg": da_prior,
            "innovation_p_mw": dp_prior,
            "innovation_q_mvar": dq_prior,
            "innovation_score": inv_score,

            # Local WLS residual features
            "norm_residual_v": rv,
            "norm_residual_p": rp,
            "norm_residual_q": rq,
            "abs_norm_residual_v": abs(rv),
            "abs_norm_residual_p": abs(rp),
            "abs_norm_residual_q": abs(rq),
            "residual_score": res_score,

            # Raw local measurements (possible graph features)
            "measured_vm_pu": float(z[iv]),
            "measured_p_mw": float(z[ip]),
            "measured_q_mvar": float(z[iq]),

            # Truth/evaluation only -- DO NOT use as model input.
            "eval_true_vm_pu": float(vm_true[b]),
            "eval_true_va_deg": float(va_true[b]),
            "eval_state_bias_vm_pu": dv_truth,
            "eval_state_bias_va_deg": da_truth,

            # Labels / attack access diagnostics -- DO NOT use as features.
            "label_attack_target_bus": attack_target_bus,
            "label_physical_target_bus": physical_target_bus,
            "label_is_attack_target_node": int(
                is_cyber and b == target_bus
            ),
            "label_is_physical_target_node": int(
                is_physical and b == target_bus
            ),
            "label_hops_to_attack_target": hop_distance(
                adj, b, attack_target_bus
            ),
            "label_attack_sigma_v": float(attack_sigma[iv]),
            "label_attack_sigma_p": float(attack_sigma[ip]),
            "label_attack_sigma_q": float(attack_sigma[iq]),
            "label_node_compromised": int(
                max(
                    attack_sigma[iv],
                    attack_sigma[ip],
                    attack_sigma[iq],
                ) > 1e-3
            ),

            # Graph-level BDD copied for convenience.
            "chi2_j": float(result["j_stat"]),
            "chi2_threshold": float(result["chi2_threshold"]),
            "bdd_flag": int(result["bdd_flag"]),
        })

    edge_rows = []

    for line in range(n_line):
        ip = pos["p_line"][line]
        iq = pos["q_line"][line]

        line_row = net.line.loc[line]
        f = int(line_row["from_bus"])
        t = int(line_row["to_bus"])

        prior_p = float(prior_h[ip])
        prior_q = float(prior_h[iq])
        est_p = float(h_hat[ip])
        est_q = float(h_hat[iq])

        dp = float(est_p - prior_p)
        dq = float(est_q - prior_q)

        rp = float(norm_res[ip])
        rq = float(norm_res[iq])

        edge_rows.append({
            "graph_id": graph_id,
            "replication": int(replication),
            "dataset_split": split_name,
            "case": case_name,
            "case_label": CASE_LABEL[case_name],
            "family": CASE_FAMILY[case_name],

            "edge_id": int(line),
            "edge_name": str(line_row["name"]),
            "from_bus": f,
            "to_bus": t,

            # Static topology / electrical parameters
            "length_km": float(line_row["length_km"]),
            "r_ohm_per_km": float(line_row["r_ohm_per_km"]),
            "x_ohm_per_km": float(line_row["x_ohm_per_km"]),
            "max_i_ka": float(line_row["max_i_ka"]),

            # Prior / WLS edge telemetry
            "prior_p_from_mw": prior_p,
            "prior_q_from_mvar": prior_q,
            "wls_p_from_mw": est_p,
            "wls_q_from_mvar": est_q,
            "innovation_p_from_mw": dp,
            "innovation_q_from_mvar": dq,
            "innovation_score": edge_innovation_score(dp, dq),

            "norm_residual_p_from": rp,
            "norm_residual_q_from": rq,
            "residual_score": float(np.sqrt(rp ** 2 + rq ** 2)),

            "measured_p_from_mw": float(z[ip]),
            "measured_q_from_mvar": float(z[iq]),

            # Truth/evaluation only.
            "eval_true_p_from_mw": float(h_true[ip]),
            "eval_true_q_from_mvar": float(h_true[iq]),

            # Labels/evaluation only.
            "label_attack_target_bus": attack_target_bus,
            "label_is_adjacent_to_attack_target": int(
                is_cyber and (f == target_bus or t == target_bus)
            ),
            "label_attack_sigma_p": float(attack_sigma[ip]),
            "label_attack_sigma_q": float(attack_sigma[iq]),
            "label_edge_compromised": int(
                max(attack_sigma[ip], attack_sigma[iq]) > 1e-3
            ),

            "chi2_j": float(result["j_stat"]),
            "chi2_threshold": float(result["chi2_threshold"]),
            "bdd_flag": int(result["bdd_flag"]),
        })

    graph_row = {
        "graph_id": graph_id,
        "replication": int(replication),
        "dataset_split": split_name,
        "case": case_name,
        "case_label": CASE_LABEL[case_name],
        "family": CASE_FAMILY[case_name],
        "context_timestamp": context["timestamp"],
        "context_pv_kw": 1000.0 * float(context["pv_p_mw"]),
        "context_load_kw": 1000.0 * float(context["load_total_p_mw"]),
        "context_soc_pct": 100.0 * float(context["bess_soc"]),
        "target_bus": int(target_bus),
        "attack_target_bus": attack_target_bus,
        "physical_target_bus": physical_target_bus,
        "physical_load_multiplier": (
            float(physical_multiplier) if is_physical else 1.0
        ),
        "chi2_j": float(result["j_stat"]),
        "chi2_threshold": float(result["chi2_threshold"]),
        "chi2_p_value": float(result["chi2_p_value"]),
        "bdd_flag": int(result["bdd_flag"]),
        "max_abs_norm_residual": float(result["max_abs_norm_residual"]),
        "rms_norm_residual": float(result["rms_norm_residual"]),
        "n_compromised_measurements": int(
            np.sum(attack_sigma > 1e-3)
        ),
        "max_attack_sigma": float(np.max(attack_sigma)),
        "attack_dv_pu": float(
            case_pack["attack_meta"].get("attack_dv_pu", 0.0)
        ),
        "attack_da_deg": float(
            case_pack["attack_meta"].get("attack_da_deg", 0.0)
        ),
    }

    return node_rows, edge_rows, graph_row


def localization_diagnostics(node_df):
    rows = []

    for case_name in [
        "physical_load_disturbance",
        "naive_single_sensor_corruption",
        "nonlinear_model_consistent_fdia",
    ]:
        d = node_df[node_df["case"] == case_name].copy()
        if d.empty:
            continue

        residual_hits = []
        innovation_hits = []

        for graph_id, g in d.groupby("graph_id"):
            if case_name == "physical_load_disturbance":
                target = int(g["label_physical_target_bus"].iloc[0])
            else:
                target = int(g["label_attack_target_bus"].iloc[0])

            pred_res = int(
                g.loc[g["residual_score"].idxmax(), "node_id"]
            )
            pred_inv = int(
                g.loc[g["innovation_score"].idxmax(), "node_id"]
            )

            residual_hits.append(int(pred_res == target))
            innovation_hits.append(int(pred_inv == target))

        rows.append({
            "case": case_name,
            "case_label": CASE_LABEL[case_name],
            "n_graphs": len(residual_hits),
            "top1_residual_locator_accuracy": float(np.mean(residual_hits)),
            "top1_prior_innovation_locator_accuracy": float(
                np.mean(innovation_hits)
            ),
        })

    return pd.DataFrame(rows)


def graph_summary(graph_df, node_df):
    rows = []

    for case_name in CASES:
        g = graph_df[graph_df["case"] == case_name]
        n = node_df[node_df["case"] == case_name]

        if g.empty:
            continue

        if CASE_FAMILY[case_name] == "cyber":
            target_mask = n["label_is_attack_target_node"] == 1
        elif CASE_FAMILY[case_name] == "physical":
            target_mask = n["label_is_physical_target_node"] == 1
        else:
            target_mask = pd.Series(False, index=n.index)

        non_target_mask = ~target_mask

        rows.append({
            "case": case_name,
            "case_label": CASE_LABEL[case_name],
            "family": CASE_FAMILY[case_name],
            "n_graphs": len(g),
            "bdd_flag_rate": float(g["bdd_flag"].mean()),
            "median_chi2_j": float(g["chi2_j"].median()),
            "median_max_norm_residual": float(
                g["max_abs_norm_residual"].median()
            ),
            "median_target_innovation_score": (
                float(n.loc[target_mask, "innovation_score"].median())
                if target_mask.any() else np.nan
            ),
            "median_non_target_innovation_score": (
                float(n.loc[non_target_mask, "innovation_score"].median())
                if non_target_mask.any() else np.nan
            ),
            "median_target_residual_score": (
                float(n.loc[target_mask, "residual_score"].median())
                if target_mask.any() else np.nan
            ),
            "median_non_target_residual_score": (
                float(n.loc[non_target_mask, "residual_score"].median())
                if non_target_mask.any() else np.nan
            ),
        })

    return pd.DataFrame(rows)


def plot_target_vs_nontarget(node_df):
    cases = [
        "physical_load_disturbance",
        "naive_single_sensor_corruption",
        "nonlinear_model_consistent_fdia",
    ]

    labels = []
    target_vals = []
    other_vals = []

    for c in cases:
        d = node_df[node_df["case"] == c]

        if CASE_FAMILY[c] == "cyber":
            mask = d["label_is_attack_target_node"] == 1
        else:
            mask = d["label_is_physical_target_node"] == 1

        labels.append(CASE_LABEL[c])
        target_vals.append(float(d.loc[mask, "innovation_score"].median()))
        other_vals.append(float(d.loc[~mask, "innovation_score"].median()))

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, target_vals, width, label="Target node")
    ax.bar(x + width / 2, other_vals, width, label="Non-target nodes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Median protected-prior innovation score")
    ax.set_title("Graph-local prior innovation: target vs non-target nodes")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "48_graph_prior_innovation_target_vs_nontarget.png",
        dpi=220,
    )
    plt.close(fig)


def plot_localization(diag):
    if diag.empty:
        return

    x = np.arange(len(diag))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(
        x - width / 2,
        100.0 * diag["top1_residual_locator_accuracy"],
        width,
        label="WLS residual score",
    )
    ax.bar(
        x + width / 2,
        100.0 * diag["top1_prior_innovation_locator_accuracy"],
        width,
        label="Protected-prior innovation",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(diag["case_label"], rotation=10, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 target localization accuracy (%)")
    ax.set_title("Simple localization diagnostic before graph learning")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "49_residual_vs_prior_localization.png",
        dpi=220,
    )
    plt.close(fig)


def plot_stealth_relation(node_df):
    d = node_df[
        node_df["case"] == "nonlinear_model_consistent_fdia"
    ].copy()

    if d.empty:
        return

    def relation(row):
        h = int(row["label_hops_to_attack_target"])
        if h == 0:
            return "target"
        if h == 1:
            return "1-hop neighbor"
        return "other"

    d["relation"] = d.apply(relation, axis=1)

    order = ["target", "1-hop neighbor", "other"]
    med_inv = [
        d.loc[d["relation"] == k, "innovation_score"].median()
        for k in order
    ]
    med_res = [
        d.loc[d["relation"] == k, "residual_score"].median()
        for k in order
    ]

    x = np.arange(len(order))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width / 2, med_inv, width, label="Prior innovation")
    ax.bar(x + width / 2, med_res, width, label="WLS residual")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Median local score")
    ax.set_title("Stealth FDIA topology footprint")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        FIGURES / "50_stealth_topology_footprint.png",
        dpi=220,
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-rep",
        type=int,
        default=120,
        help="Matched graph replications. Default: 120",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    args = parser.parse_args()

    slow = load_slow_dataset()
    rng = np.random.default_rng(args.seed)
    split_map = make_replication_split(args.n_rep, args.seed)

    probe = topology.build_microgrid()
    phase21.configure_context(probe, slow.iloc[0])
    phase21.solve_truth(probe)
    _, probe_stds = phase21.measurement_schema(probe)

    n_meas = len(probe_stds)
    load_buses = [
        int(probe.gridra["bus_load_a"]),
        int(probe.gridra["bus_load_b"]),
    ]

    print(
        "\n=== PHASE 2R: GRAPH-READY WLS + PROTECTED-PRIOR TELEMETRY ==="
    )
    print(f"Matched replications      : {args.n_rep}")
    print(f"Cases per replication     : {len(CASES)}")
    print(f"Expected graph snapshots  : {args.n_rep * len(CASES)}")
    print("Nodes per graph           : 5")
    print("Edges per graph           : 4")
    print("Prior load forecast sigma : 2.5% per load, clipped to +/-6%")
    print("PV/BESS prior             : authorized operating commands")
    print("Split policy              : replication-group safe")

    node_rows = []
    edge_rows = []
    graph_rows = []

    for rep in range(args.n_rep):
        context = slow.iloc[
            int(rng.integers(0, len(slow)))
        ]

        target_bus = int(rng.choice(load_buses))
        physical_multiplier = float(rng.uniform(1.25, 1.70))

        # Same measurement-noise realization across all matched cases.
        eps = rng.normal(0.0, 1.0, size=n_meas)

        # Same uncertain protected prior across all matched cases.
        prior_ctx, prior_err_a, prior_err_b = prior_context_from_actual(
            context, rng
        )
        prior_pack = solve_prior(prior_ctx)

        print(
            f"  {rep + 1:3d}/{args.n_rep} | "
            f"{split_map[rep]:10s} | "
            f"context={context['timestamp']} | "
            f"PV={1000*float(context['pv_p_mw']):.1f} kW | "
            f"Load={1000*float(context['load_total_p_mw']):.1f} kW | "
            f"target Bus {target_bus}"
        )

        for case_name in CASES:
            try:
                pack = construct_case(
                    context=context,
                    case_name=case_name,
                    target_bus=target_bus,
                    physical_multiplier=physical_multiplier,
                    standardized_noise=eps,
                    rng=rng,
                )
            except Exception as exc:
                print(f"    WARNING {case_name}: {exc}")
                pack = None

            if pack is None:
                continue

            nrows, erows, grow = extract_graph_rows(
                replication=rep,
                split_name=split_map[rep],
                case_name=case_name,
                context=context,
                target_bus=target_bus,
                physical_multiplier=physical_multiplier,
                prior_pack=prior_pack,
                prior_error_a=prior_err_a,
                prior_error_b=prior_err_b,
                case_pack=pack,
            )

            node_rows.extend(nrows)
            edge_rows.extend(erows)
            graph_rows.append(grow)

    node_df = pd.DataFrame(node_rows)
    edge_df = pd.DataFrame(edge_rows)
    graph_df = pd.DataFrame(graph_rows)

    if graph_df.empty:
        raise RuntimeError("No graph snapshots generated.")

    summary = graph_summary(graph_df, node_df)
    diag = localization_diagnostics(node_df)

    node_df.to_csv(
        DATA / "phase2r_graph_node_features.csv",
        index=False,
    )
    edge_df.to_csv(
        DATA / "phase2r_graph_edge_features.csv",
        index=False,
    )
    graph_df.to_csv(
        DATA / "phase2r_graph_metadata.csv",
        index=False,
    )
    summary.to_csv(
        RESULTS / "phase2r_graph_telemetry_summary.csv",
        index=False,
    )
    diag.to_csv(
        RESULTS / "phase2r_localization_diagnostic.csv",
        index=False,
    )

    plot_target_vs_nontarget(node_df)
    plot_localization(diag)
    plot_stealth_relation(node_df)

    print("\n=== DATASET COUNTS ===")
    print(f"Graph snapshots : {len(graph_df):,}")
    print(f"Node rows       : {len(node_df):,}")
    print(f"Edge rows       : {len(edge_df):,}")

    print("\nGraph split counts:")
    print(
        graph_df.groupby(["dataset_split", "case"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\n=== GRAPH TELEMETRY SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== SIMPLE LOCALIZATION DIAGNOSTIC ===")
    print(diag.to_string(index=False))

    stealth = summary[
        summary["case"] == "nonlinear_model_consistent_fdia"
    ]

    if not stealth.empty:
        s = stealth.iloc[0]
        print("\n=== PHASE-2R SCIENTIFIC CHECK ===")
        print(
            f"Stealth BDD flag rate                    : "
            f"{100*s['bdd_flag_rate']:.1f}%"
        )
        print(
            f"Stealth median target innovation score   : "
            f"{s['median_target_innovation_score']:.3f}"
        )
        print(
            f"Stealth median non-target innovation     : "
            f"{s['median_non_target_innovation_score']:.3f}"
        )
        print(
            f"Stealth median target residual score     : "
            f"{s['median_target_residual_score']:.3f}"
        )

    print(
        "\nINTERPRETATION POLICY:\n"
        "1) WLS residual features are deployable-style diagnostics but are "
        "expected to remain weak for exact model-consistent attacks.\n"
        "2) Prior-innovation features use a separate protected model prior; "
        "they are not AC-state truth and include load-forecast uncertainty.\n"
        "3) If both physical load disturbances and cyber FDIAs create strong "
        "prior innovations, that is NOT failure. It is the hard-negative "
        "ambiguity the next temporal/topology-aware detector must solve.\n"
        "4) Columns beginning with 'eval_' or 'label_' are evaluation/label "
        "information and MUST NOT be fed to a learned detector.\n"
        "5) The next phase should perform cyber-vs-physical classification "
        "and target localization using node/edge graph structure plus protected "
        "prior innovations; only after that is a GNN comparison justified."
    )

    print("\nSaved:")
    print("  data/phase2r_graph_node_features.csv")
    print("  data/phase2r_graph_edge_features.csv")
    print("  data/phase2r_graph_metadata.csv")
    print("  results/phase2r_graph_telemetry_summary.csv")
    print("  results/phase2r_localization_diagnostic.csv")
    print("  figures/48_graph_prior_innovation_target_vs_nontarget.png")
    print("  figures/49_residual_vs_prior_localization.png")
    print("  figures/50_stealth_topology_footprint.png")


if __name__ == "__main__":
    main()
