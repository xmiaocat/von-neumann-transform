from pathlib import Path
import numpy as np
import pytest

from von_neumann_transform import VonNeumannTransform, BasisMethod

DATA_DIR = Path(__file__).parent / "data"

BASIS_METHODS = [BasisMethod.DIRECT, BasisMethod.FACTORISE, BasisMethod.FFT]


@pytest.fixture(scope="module")
def ref_and_vnt():
    with np.load(DATA_DIR / "example_pulse_ivnt.npz") as data:
        npoints = int(data["npoints"])
        omega_min = data["omega_min"]
        omega_max = data["omega_max"]
        vncs = data["vncs"]
        pulses_ref = data["pulses"]

    vnt = VonNeumannTransform(npoints, omega_min, omega_max)

    return vnt, vncs, pulses_ref


@pytest.mark.parametrize("basis_method", BASIS_METHODS)
def test_ivnt_example_pulses(ref_and_vnt, basis_method):
    vnt, vncs, pulses_ref = ref_and_vnt
    for vnc, pulse_ref in zip(vncs, pulses_ref):
        print(vnc.shape, pulse_ref.shape)
        pulse = vnt.inverse_transform(vnc, method=basis_method)
        np.testing.assert_allclose(pulse, pulse_ref, rtol=1e-8, atol=1e-9)
