from dataclasses import dataclass
import numpy as np


def wrap_angle(angle_rad):
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class GFLState:
    theta_pll: float = 0.0
    pll_integrator: float = 0.0


class GFLInverter:
    """
    Reduced-order PLL-based Grid-Following inverter.

    This is not an EMT switching model. The goal is to expose internal
    cyber-physical variables that can later be used for attack detection.
    """

    def __init__(
        self,
        kp_pll=18.0,
        ki_pll=250.0,
        i_max_pu=1.20,
        q_voltage_gain=2.0,
        v_floor_pu=0.15,
    ):
        self.kp_pll = kp_pll
        self.ki_pll = ki_pll
        self.i_max_pu = i_max_pu
        self.q_voltage_gain = q_voltage_gain
        self.v_floor_pu = v_floor_pu
        self.state = GFLState()

    def reset(self, theta0=0.0):
        self.state = GFLState(theta_pll=theta0, pll_integrator=0.0)

    def step(self, v_meas_pu, theta_meas_rad, p_ref_pu, q_ref_pu=0.0, dt=1e-3):
        phase_error = wrap_angle(theta_meas_rad - self.state.theta_pll)
        v_q = v_meas_pu * np.sin(phase_error)

        self.state.pll_integrator += v_q * dt
        omega_dev = self.kp_pll * v_q + self.ki_pll * self.state.pll_integrator
        self.state.theta_pll += omega_dev * dt

        v_eff = max(v_meas_pu, self.v_floor_pu)
        id_cmd = p_ref_pu / v_eff
        iq_cmd = -q_ref_pu / v_eff + self.q_voltage_gain * max(0.0, 1.0 - v_meas_pu)

        i_mag_cmd = np.hypot(id_cmd, iq_cmd)
        limiter_active = i_mag_cmd > self.i_max_pu

        if limiter_active:
            scale = self.i_max_pu / max(i_mag_cmd, 1e-9)
            id_out = id_cmd * scale
            iq_out = iq_cmd * scale
        else:
            id_out = id_cmd
            iq_out = iq_cmd

        p_out_pu = v_meas_pu * id_out
        q_out_pu = -v_meas_pu * iq_out

        return {
            "theta_pll_rad": self.state.theta_pll,
            "theta_pll_deg": np.rad2deg(self.state.theta_pll),
            "pll_freq_dev_hz": omega_dev / (2.0 * np.pi),
            "v_q_pu": v_q,
            "id_pu": id_out,
            "iq_pu": iq_out,
            "i_mag_pu": np.hypot(id_out, iq_out),
            "limiter_active": bool(limiter_active),
            "p_out_pu": p_out_pu,
            "q_out_pu": q_out_pu,
        }


@dataclass
class GFMState:
    theta_internal: float = 0.0
    omega_dev: float = 0.0


class GFMInverter:
    """
    Reduced-order voltage-source / VSM-like Grid-Forming inverter.

    Internal voltage source E∠theta is placed behind a virtual reactance.
    Converter current is limited to Imax.
    """

    def __init__(
        self,
        m_inertia=0.25,
        d_damping=0.80,
        e_internal_pu=1.02,
        x_virtual_pu=0.25,
        i_max_pu=1.20,
    ):
        self.m_inertia = m_inertia
        self.d_damping = d_damping
        self.e_internal_pu = e_internal_pu
        self.x_virtual_pu = x_virtual_pu
        self.i_max_pu = i_max_pu
        self.state = GFMState()

    def reset(self, theta0=0.0):
        self.state = GFMState(theta_internal=theta0, omega_dev=0.0)

    def _electrical_output(self, v_pcc_pu, theta_pcc_rad):
        v_inv = self.e_internal_pu * np.exp(1j * self.state.theta_internal)
        v_pcc = v_pcc_pu * np.exp(1j * theta_pcc_rad)

        i_unlimited = (v_inv - v_pcc) / (1j * self.x_virtual_pu)
        i_mag_unlimited = abs(i_unlimited)

        limiter_active = i_mag_unlimited > self.i_max_pu
        if limiter_active:
            i_out = i_unlimited * (self.i_max_pu / max(i_mag_unlimited, 1e-9))
        else:
            i_out = i_unlimited

        s_pcc = v_pcc * np.conj(i_out)
        return {
            "i_mag_pu": abs(i_out),
            "limiter_active": bool(limiter_active),
            "p_out_pu": float(np.real(s_pcc)),
            "q_out_pu": float(np.imag(s_pcc)),
        }

    def step(self, v_pcc_pu, theta_pcc_rad, p_ref_pu, dt=1e-3):
        electrical = self._electrical_output(v_pcc_pu, theta_pcc_rad)
        p_e = electrical["p_out_pu"]

        domega = (p_ref_pu - p_e - self.d_damping * self.state.omega_dev) / self.m_inertia
        self.state.omega_dev += domega * dt
        self.state.theta_internal += self.state.omega_dev * dt

        electrical = self._electrical_output(v_pcc_pu, theta_pcc_rad)
        return {
            "theta_internal_rad": self.state.theta_internal,
            "theta_internal_deg": np.rad2deg(self.state.theta_internal),
            "gfm_freq_dev_hz": self.state.omega_dev / (2.0 * np.pi),
            "i_mag_pu": electrical["i_mag_pu"],
            "limiter_active": electrical["limiter_active"],
            "p_out_pu": electrical["p_out_pu"],
            "q_out_pu": electrical["q_out_pu"],
        }
