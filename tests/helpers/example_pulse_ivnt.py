#!/usr/bin/env python

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from von_neumann_transform import VonNeumannTransform, BasisMethod

DATA_DIR = Path(__file__).parents[1] / "data"

with np.load(DATA_DIR / "example_pulse_vnt.npz") as data:
    NPOINTS = int(data["npoints"])
    OMEGA_MIN = data["omega_min"]
    OMEGA_MAX = data["omega_max"]
    pulses = data["pulses"]
    vncs = data["vncs"]

omega = np.linspace(OMEGA_MIN, OMEGA_MAX, NPOINTS)
vnt = VonNeumannTransform(NPOINTS, OMEGA_MIN, OMEGA_MAX)

all_vncs = []
all_pulses = []
for pulse_ref, coeffs in zip(pulses, vncs):
    all_vncs.append(coeffs)

    pulse_recon = vnt.inverse_transform(coeffs, BasisMethod.DIRECT)
    all_pulses.append(pulse_recon)

# fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
#
# axs[0].plot(omega, np.abs(pulse_ref), linestyle='--', label='reference amplitude')
# axs[0].plot(omega, np.abs(pulse_recon), label='reconstructed amplitude')
# axs[1].plot(omega, np.angle(pulse_ref), linestyle='--', label='reference phase')
# axs[1].plot(omega, np.angle(pulse_recon), label='reconstructed phase')
#
# fig.tight_layout()
#
# plt.show()

np.savez(
    DATA_DIR / "example_pulse_ivnt.npz",
    npoints=NPOINTS,
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    pulses=all_pulses,
    vncs=all_vncs,
)
