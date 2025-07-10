from pathlib import Path
import numpy as np
import pytest

from von_neumann_transform import (
    VonNeumannTransform,
    BasisMethod,
    MatVecMethod,
    SolverMethod,
)

DATA_DIR = Path(__file__).parent / "data"

BASIS_METHODS = [BasisMethod.DIRECT, BasisMethod.FACTORISE, BasisMethod.FFT]
MATVEC_AND_SOLVER_METHODS = [
    (MatVecMethod.DIRECT, SolverMethod.DIRECT),
    (MatVecMethod.TOEPLITZ_MATMUL, SolverMethod.CG),
    (MatVecMethod.TOEPLITZ_MATMUL, SolverMethod.BICGSTAB),
    (MatVecMethod.TOEPLITZ_MATMUL, SolverMethod.LGMRES),
    (MatVecMethod.TOEPLITZ_EINSUM, SolverMethod.CG),
    (MatVecMethod.TOEPLITZ_EINSUM, SolverMethod.BICGSTAB),
    (MatVecMethod.TOEPLITZ_EINSUM, SolverMethod.LGMRES),
    pytest.param(
        MatVecMethod.TOEPLITZ_HANKEL,
        SolverMethod.CG,
        marks=pytest.mark.xfail(reason="Hankel method not implemented yet"),
    ),
    pytest.param(
        MatVecMethod.TOEPLITZ_HANKEL,
        SolverMethod.BICGSTAB,
        marks=pytest.mark.xfail(reason="Hankel method not implemented yet"),
    ),
    pytest.param(
        MatVecMethod.TOEPLITZ_HANKEL,
        SolverMethod.LGMRES,
        marks=pytest.mark.xfail(reason="Hankel method not implemented yet"),
    ),
]


@pytest.fixture(scope="module")
def ref_and_vnt():
    with np.load(DATA_DIR / "example_pulse_vnt.npz") as data:
        npoints = int(data["npoints"])
        omega_min = data["omega_min"]
        omega_max = data["omega_max"]
        pulses = data["pulses"]
        vncs_ref = data["vncs"]

    vnt = VonNeumannTransform(npoints, omega_min, omega_max)

    return vnt, pulses, vncs_ref


@pytest.mark.parametrize(
    "matvec_method,solver_method", MATVEC_AND_SOLVER_METHODS
)
@pytest.mark.parametrize("basis_method", BASIS_METHODS)
def test_vnt_example_pulses(
    ref_and_vnt,
    basis_method,
    matvec_method,
    solver_method,
):
    vnt, pulses, vncs_ref = ref_and_vnt
    for pulse, q_nm_ref in zip(pulses, vncs_ref):
        q_nm = vnt.transform(
            pulse,
            basis_method=basis_method,
            matvec_method=matvec_method,
            solver_method=solver_method,
        )

        np.testing.assert_allclose(q_nm, q_nm_ref, rtol=1e-8, atol=1e-9)
