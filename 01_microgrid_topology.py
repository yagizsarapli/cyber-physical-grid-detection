import pandapower as pp

S_BASE_MVA = 0.10  # 100 kVA system base


def build_microgrid():
    """
    Build a compact 5-bus, 0.4-kV grid-connected microgrid.

    Bus 0: PCC / utility-grid connection
    Bus 1: PV inverter connection
    Bus 2: load A
    Bus 3: BESS inverter connection
    Bus 4: load B
    """
    net = pp.create_empty_network(sn_mva=S_BASE_MVA)

    b0 = pp.create_bus(net, vn_kv=0.4, name="Bus 0 - PCC")
    b1 = pp.create_bus(net, vn_kv=0.4, name="Bus 1 - PV GFL")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2 - Load A")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3 - BESS GFM")
    b4 = pp.create_bus(net, vn_kv=0.4, name="Bus 4 - Load B")

    pp.create_ext_grid(net, bus=b0, vm_pu=1.0, va_degree=0.0, name="Utility Grid")

    line_data = [
        (b0, b1, 0.08, "L01"),
        (b1, b2, 0.10, "L12"),
        (b1, b3, 0.12, "L13"),
        (b3, b4, 0.10, "L34"),
    ]

    for fb, tb, length_km, name in line_data:
        pp.create_line_from_parameters(
            net,
            from_bus=fb,
            to_bus=tb,
            length_km=length_km,
            r_ohm_per_km=0.642,
            x_ohm_per_km=0.083,
            c_nf_per_km=210.0,
            max_i_ka=0.20,
            name=name,
        )

    pp.create_load(net, bus=b2, p_mw=0.030, q_mvar=0.008, name="Load A")
    pp.create_load(net, bus=b4, p_mw=0.025, q_mvar=0.006, name="Load B")

    pv_idx = pp.create_sgen(net, bus=b1, p_mw=0.035, q_mvar=0.0, name="PV - GFL")
    bess_idx = pp.create_sgen(net, bus=b3, p_mw=0.015, q_mvar=0.0, name="BESS - GFM")

    net.gridra = {
        "bus_pcc": b0,
        "bus_pv": b1,
        "bus_load_a": b2,
        "bus_bess": b3,
        "bus_load_b": b4,
        "pv_sgen": pv_idx,
        "bess_sgen": bess_idx,
        "s_base_mva": S_BASE_MVA,
    }
    return net


def solve_base_case(net):
    pp.runpp(
        net,
        algorithm="nr",
        calculate_voltage_angles=True,
        init="flat",
        tolerance_mva=1e-8,
        max_iteration=30,
    )
    return net


if __name__ == "__main__":
    net = build_microgrid()
    solve_base_case(net)

    print("\n=== 5-BUS MICROGRID BASE CASE ===")
    print("\nBus voltages:")
    print(net.res_bus[["vm_pu", "va_degree"]])

    print("\nLine loading:")
    print(net.res_line[["loading_percent", "p_from_mw", "q_from_mvar"]])

    print("\nExternal-grid exchange:")
    print(net.res_ext_grid[["p_mw", "q_mvar"]])
