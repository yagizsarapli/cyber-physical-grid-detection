import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology = load_module("01_microgrid_topology.py", "topology")
inv = load_module("02_inverter_models.py", "inverter_models")

net = topology.build_microgrid()
topology.solve_base_case(net)

pv_bus = net.gridra["bus_pv"]
bess_bus = net.gridra["bus_bess"]

v_pv = float(net.res_bus.loc[pv_bus, "vm_pu"])
a_pv = np.deg2rad(float(net.res_bus.loc[pv_bus, "va_degree"]))

v_bess = float(net.res_bus.loc[bess_bus, "vm_pu"])
a_bess = np.deg2rad(float(net.res_bus.loc[bess_bus, "va_degree"]))

gfl = inv.GFLInverter()
gfm = inv.GFMInverter()

gfl.reset(theta0=a_pv)
gfm.reset(theta0=a_bess)

dt = 1e-3
for _ in range(2000):
    gfl_out = gfl.step(v_pv, a_pv, p_ref_pu=0.35, q_ref_pu=0.0, dt=dt)
    gfm_out = gfm.step(v_bess, a_bess, p_ref_pu=0.15, dt=dt)

print("\n=== PHASE 2A SMOKE TEST ===")
print(f"PV bus voltage     : {v_pv:.4f} pu")
print(f"BESS bus voltage   : {v_bess:.4f} pu")

print("\nGFL internal state:")
for k in ["theta_pll_deg", "pll_freq_dev_hz", "id_pu", "iq_pu", "i_mag_pu", "limiter_active"]:
    print(f"  {k:20s}: {gfl_out[k]}")

print("\nGFM internal state:")
for k in ["theta_internal_deg", "gfm_freq_dev_hz", "p_out_pu", "q_out_pu", "i_mag_pu", "limiter_active"]:
    print(f"  {k:20s}: {gfm_out[k]}")

print("\nOK: network + GFL + GFM backbone is running.")
