import sys
import copy
import logging
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import pandapower as pp
from pandapower.estimation import estimate

# Keep pandapower from flooding the terminal; we print explicit results below.
logging.getLogger("pandapower").setLevel(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology = load_module("01_microgrid_topology.py", "topology")

RNG = np.random.default_rng(123)

STD_V_PU = 0.005
STD_P_MW = 0.0010
STD_Q_MVAR = 0.0008
STD_LINE_P_MW = 0.0012
STD_LINE_Q_MVAR = 0.0010


def noisy(x, std):
    return float(x + RNG.normal(0.0, std))


def add_measurements(net):
    if len(net.measurement):
        net.measurement.drop(net.measurement.index, inplace=True)

    for bus in net.bus.index:
        pp.create_measurement(
            net, "v", "bus",
            noisy(float(net.res_bus.at[bus, "vm_pu"]), STD_V_PU),
            STD_V_PU, int(bus),
            name=f"V_bus_{bus}"
        )
        pp.create_measurement(
            net, "p", "bus",
            noisy(float(net.res_bus.at[bus, "p_mw"]), STD_P_MW),
            STD_P_MW, int(bus),
            name=f"P_bus_{bus}"
        )
        pp.create_measurement(
            net, "q", "bus",
            noisy(float(net.res_bus.at[bus, "q_mvar"]), STD_Q_MVAR),
            STD_Q_MVAR, int(bus),
            name=f"Q_bus_{bus}"
        )

    for line in net.line.index:
        pp.create_measurement(
            net, "p", "line",
            noisy(float(net.res_line.at[line, "p_from_mw"]), STD_LINE_P_MW),
            STD_LINE_P_MW, int(line),
            side="from",
            name=f"P_line_{line}_from"
        )
        pp.create_measurement(
            net, "q", "line",
            noisy(float(net.res_line.at[line, "q_from_mvar"]), STD_LINE_Q_MVAR),
            STD_LINE_Q_MVAR, int(line),
            side="from",
            name=f"Q_line_{line}_from"
        )


def build_case():
    net = topology.build_microgrid()
    topology.solve_base_case(net)
    add_measurements(net)
    return net


TESTS = [
    (
        "WLS flat",
        dict(
            algorithm="wls",
            init="flat",
            tolerance=1e-7,
            maximum_iterations=30,
            calculate_voltage_angles=True,
        ),
    ),
    (
        "SHGM flat",
        dict(
            algorithm="irwls",
            estimator="shgm",
            a=5,
            init="flat",
            tolerance=1e-7,
            maximum_iterations=30,
            calculate_voltage_angles=True,
        ),
    ),
    (
        "LAV LP flat",
        dict(
            algorithm="lp",
            init="flat",
            tolerance=1e-7,
            maximum_iterations=50,
            calculate_voltage_angles=True,
        ),
    ),
]

print("\n=== ENVIRONMENT ===")
print("Python      :", sys.version.split()[0])
print("pandapower  :", pp.__version__)
print("numpy       :", np.__version__)
print("pandas      :", pd.__version__)
print("scipy       :", scipy.__version__)

base = build_case()
print("measurements:", len(base.measurement))

wls_result_net = None

print("\n=== ESTIMATOR DIAGNOSTIC ===")
for name, kwargs in TESTS:
    net = copy.deepcopy(base)

    try:
        ok = bool(estimate(net, **kwargs))
        has_results = hasattr(net, "res_bus_est") and len(net.res_bus_est) > 0

        print(f"\n{name}")
        print("  success returned :", ok)
        print("  res_bus_est ready:", has_results)

        if ok and has_results:
            print(
                "  V range          : "
                f"{net.res_bus_est.vm_pu.min():.6f} .. "
                f"{net.res_bus_est.vm_pu.max():.6f} pu"
            )

            if name.startswith("WLS"):
                wls_result_net = net

    except Exception as exc:
        print(f"\n{name}")
        print("  EXCEPTION:", type(exc).__name__)
        print("  message  :", repr(str(exc)))


# Extra diagnostic: use WLS solution as a warm start for algorithms
# that support "results". This is only a convergence check.
if wls_result_net is not None:
    print("\n=== WARM-START CHECKS ===")

    for name, kwargs in [
        (
            "SHGM init=results",
            dict(
                algorithm="irwls",
                estimator="shgm",
                a=5,
                init="results",
                tolerance=1e-7,
                maximum_iterations=50,
                calculate_voltage_angles=True,
            ),
        ),
        (
            "LAV LP init=results",
            dict(
                algorithm="lp",
                init="results",
                tolerance=1e-7,
                maximum_iterations=100,
                calculate_voltage_angles=True,
            ),
        ),
    ]:
        net = copy.deepcopy(wls_result_net)

        try:
            ok = bool(estimate(net, **kwargs))
            print(f"\n{name}")
            print("  success returned :", ok)
            if ok:
                print(
                    "  V range          : "
                    f"{net.res_bus_est.vm_pu.min():.6f} .. "
                    f"{net.res_bus_est.vm_pu.max():.6f} pu"
                )
        except Exception as exc:
            print(f"\n{name}")
            print("  EXCEPTION:", type(exc).__name__)
            print("  message  :", repr(str(exc)))

print(
    "\nCopy the complete output back to ChatGPT. "
    "Do not rerun the 50x benchmark yet."
)
