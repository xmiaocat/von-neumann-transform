#!/usr/bin/env python

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from von_neumann_transform import (
    VonNeumannTransform,
    BasisMethod,
    MatVecMethod,
    SolverMethod,
)

DATA_DIR = Path(__file__).parents[1] / "data"


def get_pulse(
    omega,
    omega0,
    sigma,
    phi1=0.0,
    phi2=0.0,
    phi3=0.0,
    phi4=0.0,
    blank_level=0.001,
):
    delta_w = omega - omega0

    amplitude = np.exp(-0.5 * (delta_w / sigma) ** 2)
    phase = (
        phi1 * delta_w
        + phi2 * delta_w**2
        + phi3 * delta_w**3
        + phi4 * delta_w**4
    )
    phase = (phase + np.pi) % (2 * np.pi) - np.pi

    if blank_level > 0.0:
        blank_mask = np.abs(amplitude) < blank_level
        phase[blank_mask] = 0.0

    return amplitude, phase


def get_multi_pulse(omega, params_list):
    components = []
    for p in params_list:
        amplitude, phase = get_pulse(
            omega,
            p.get("omega0"),
            p.get("sigma"),
            phi1=p.get("phi1", 0.0),
            phi2=p.get("phi2", 0.0),
            phi3=p.get("phi3", 0.0),
            phi4=p.get("phi4", 0.0),
        )
        components.append(amplitude * np.exp(1j * phase))
    return np.sum(components, axis=0)


PARAMS_LIST0 = [
    {"omega0": 2.35, "sigma": 0.025, "phi2": 30000.0},
]
PARAMS_LIST1 = [
    {"omega0": 2.35, "sigma": 0.025, "phi1": 2000.0},
    {"omega0": 2.35, "sigma": 0.025, "phi1": -2000.0},
]
PARAMS_LIST2 = [
    {"omega0": 2.30, "sigma": 0.020, "phi1": 500.0},
    {"omega0": 2.40, "sigma": 0.020, "phi1": -500.0},
    {"omega0": 2.35, "sigma": 0.025, "phi2": 10000.0},
]
PARAMS_LIST3 = [
    {"omega0": 2.30, "sigma": 0.020, "phi1": 50.0},
    {"omega0": 2.40, "sigma": 0.020, "phi1": -50.0},
    {"omega0": 2.35, "sigma": 0.025, "phi2": 10000.0},
    {"omega0": 2.35, "sigma": 0.025, "phi4": 50000.0},
]
PARAMS_LIST4 = [
    {"omega0": 2.30, "sigma": 0.020, "phi1": 1000.0},
    {"omega0": 2.40, "sigma": 0.020, "phi1": -1000.0},
    {
        "omega0": 2.33,
        "sigma": 0.030,
        "phi1": 200.0,
        "phi2": 5000.0,
        "phi3": -20000.0,
        "phi4": 100000.0,
    },
    {
        "omega0": 2.37,
        "sigma": 0.040,
        "phi1": -200.0,
        "phi2": 10000.0,
        "phi3": 50000.0,
        "phi4": 200000.0,
    },
]

OMEGA_MIN, OMEGA_MAX = 2.05, 2.65
NPOINTS = 1024

omega = np.linspace(OMEGA_MIN, OMEGA_MAX, NPOINTS)

all_pulses = []
all_vncs = []
for pl in (
    PARAMS_LIST0,
    PARAMS_LIST1,
    PARAMS_LIST2,
    PARAMS_LIST3,
    PARAMS_LIST4,
):
    pulse = get_multi_pulse(omega, pl)
    all_pulses.append(pulse)

    vnt = VonNeumannTransform(NPOINTS, OMEGA_MIN, OMEGA_MAX)
    q_nm = vnt.transform(
        pulse,
        basis_method=BasisMethod.DIRECT,
        matvec_method=MatVecMethod.DIRECT,
        solver_method=SolverMethod.DIRECT,
    )
    all_vncs.append(q_nm)

# fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
# axs[0].plot(omega, np.abs(pulse), label='Amplitude')
# axs[1].plot(omega, np.angle(pulse), label='Phase')
#
# fig.tight_layout()
#
# fig2, axs2 = plt.subplots(1, 2, figsize=(8, 4))
# axs2[0].matshow(np.abs(q_nm), aspect='auto', cmap='viridis')
# axs2[1].matshow(np.angle(q_nm), aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
#
# fig2.tight_layout()
#
# plt.show()

np.savez(
    DATA_DIR / "example_pulse_vnt.npz",
    npoints=NPOINTS,
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    pulses=all_pulses,
    vncs=all_vncs,
)
