import numpy as np

from von_neumann_transform import BasisMethod, VonNeumannTransform


def test_direct_projection_rebuilds_basis_when_alpha_changes():
    vnt = VonNeumannTransform(16, 0.2, 5.0)
    signal = np.arange(16, dtype=np.complex128) + 1j
    vnt.get_signal_projection(
        vnt.w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        vnt.alpha,
        signal,
        BasisMethod.DIRECT,
    )
    cached_basis = vnt.alpha_nmo

    changed_alpha = 2 * vnt.alpha
    direct = vnt.get_signal_projection(
        vnt.w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        changed_alpha,
        signal,
        BasisMethod.DIRECT,
    )
    assert vnt.alpha_nmo is not cached_basis
    updated_basis = vnt.alpha_nmo
    vnt.get_signal_projection(
        vnt.w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        changed_alpha,
        signal,
        BasisMethod.DIRECT,
    )
    assert vnt.alpha_nmo is updated_basis
    reference = vnt.get_signal_projection(
        vnt.w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        changed_alpha,
        signal,
        BasisMethod.FACTORISE,
    )
    np.testing.assert_allclose(direct, reference, rtol=1e-13, atol=1e-13)


def test_direct_projection_detects_in_place_grid_change_and_restores_basis():
    vnt = VonNeumannTransform(16, 0.2, 5.0)
    signal = np.arange(16, dtype=np.complex128) + 1j
    w_grid = vnt.w_grid.copy()
    vnt.get_signal_projection(
        w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        vnt.alpha,
        signal,
        BasisMethod.DIRECT,
    )

    w_grid += 0.01
    direct = vnt.get_signal_projection(
        w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        vnt.alpha,
        signal,
        BasisMethod.DIRECT,
    )
    reference = vnt.get_signal_projection(
        w_grid,
        vnt.w_n_arr,
        vnt.t_n_arr,
        vnt.alpha,
        signal,
        BasisMethod.FACTORISE,
    )
    np.testing.assert_allclose(direct, reference, rtol=1e-13, atol=1e-13)

    q_nm = np.arange(16, dtype=np.complex128).reshape(4, 4) + 1j
    reconstructed = vnt.inverse_transform(q_nm, BasisMethod.DIRECT)
    reference = vnt.inverse_transform(q_nm, BasisMethod.FACTORISE)
    np.testing.assert_allclose(
        reconstructed, reference, rtol=1e-13, atol=1e-13
    )
