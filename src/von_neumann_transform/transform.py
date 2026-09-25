import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, lgmres

from .basis import _evaluate_basis_functions, _get_grid
from .methods import BasisMethod, MatVecMethod, PrecondMethod, SolverMethod
from .overlap import _get_ovlp_direct, _get_ovlp_linop
from .projection import (
    _get_signal_projection_factorise,
    _get_signal_projection_fft,
    _project_signal,
)
from .reconstruction import (
    _reconstruct_signal,
    _reconstruct_signal_factorise,
    _reconstruct_signal_fft,
)


class VonNeumannTransform:
    def __init__(self, npoints: int, omega_min: float, omega_max: float):
        # Validate input parameters
        if not isinstance(npoints, int):
            raise TypeError("Number of points must be an integer.")
        if npoints < 4:
            raise ValueError("Number of points must be at least 4.")
        if omega_min < 0.0 or omega_max < 0.0:
            raise ValueError("Angular frequencies must be non-negative.")
        if omega_max <= omega_min:
            raise ValueError(
                "Maximum angular frequency must be greater "
                "than minimum angular frequency."
            )

        self.npoints = npoints
        self.w_min = omega_min
        self.w_max = omega_max

        # Build the grid for the von Neumann transform
        (
            self.w_grid,
            self.t_span,
            self.w_n_arr,
            self.t_n_arr,
            self.k,
            self.alpha,
        ) = _get_grid(npoints, self.w_min, self.w_max)

        # Cache the evaluated basis together with snapshots of its inputs.
        self.alpha_nmo: None | np.ndarray = None
        self._basis_cache_inputs: (
            tuple[np.ndarray, np.ndarray, np.ndarray, float] | None
        ) = None

    def _get_basis_functions(
        self,
        w_grid: np.ndarray,
        w_n_arr: np.ndarray,
        t_n_arr: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        cached = self._basis_cache_inputs
        basis = self.alpha_nmo
        if (
            basis is None
            or cached is None
            or cached[3] != alpha
            or not np.array_equal(cached[0], w_grid)
            or not np.array_equal(cached[1], w_n_arr)
            or not np.array_equal(cached[2], t_n_arr)
        ):
            basis = _evaluate_basis_functions(w_grid, w_n_arr, t_n_arr, alpha)
            self.alpha_nmo = basis
            self._basis_cache_inputs = (
                w_grid.copy(),
                w_n_arr.copy(),
                t_n_arr.copy(),
                alpha,
            )
        return basis

    def get_signal_projection(
        self,
        w_grid: np.ndarray,
        w_n_arr: np.ndarray,
        t_n_arr: np.ndarray,
        alpha: float,
        signal: np.ndarray,
        method: BasisMethod = BasisMethod.FFT,
    ) -> np.ndarray:
        """
        Computes the projection of the input signal onto the
        basis functions defined by the von Neumann transform.
        The projection is computed as:
                   /
        alpha_nm = | alpha_ij^* (w) * signal(w) dw
                   /
        where
        alpha_ij (w) = (2 * alpha / pi)^(1/4)
                       * exp(-alpha * (w - w_i)^2 - i * t_j * (w - w_i))

        Parameters:
            w_grid (np.ndarray): Angular frequency grid of the
                input signal in the frequency domain.
            w_n_arr (np.ndarray): Angular frequency grid in the
                von Neumann plane.
            t_n_arr (np.ndarray): Time grid in the von Neumann plane.
            alpha (float): Width of the basis functions.
            signal (np.ndarray): Input signal in the frequency domain.
            method (BasisMethod): Method to compute the projection.
                Options are DIRECT, FACTORISE, and FFT.

                The DIRECT method computes all the basis functions
                evaluated at the grid points, and then computes the
                projection using a dot product. The evaluated basis
                functions are stored in memory.

                The FACTORISE method computes the projection by
                factorising the basis function and applying each
                factor sequentially. The evaluated basis
                functions are not stored in memory.

                The FFT method computes the projection using the
                Fast Fourier Transform. The evaluated basis functions
                are not stored in memory. While this method is
                typically the fastest, it can introduce (slightly)
                more numerical noise due to the FFT algorithm.

        Returns:
            alpha_nm (np.ndarray): Projection of the signal onto the
                basis functions.
        """

        if method is BasisMethod.DIRECT:
            alpha_nmo = self._get_basis_functions(
                w_grid, w_n_arr, t_n_arr, alpha
            )
            # Project the signal onto the basis functions
            dw = w_grid[1] - w_grid[0]
            alpha_nm = _project_signal(
                alpha_nmo,
                signal,
                dw,
            )
        elif method is BasisMethod.FACTORISE:
            alpha_nm = _get_signal_projection_factorise(
                w_grid,
                w_n_arr,
                t_n_arr,
                alpha,
                signal,
            )
        elif method is BasisMethod.FFT:
            alpha_nm = _get_signal_projection_fft(
                w_grid,
                w_n_arr,
                t_n_arr,
                alpha,
                signal,
            )
        else:
            assert False, (
                f"Unknown basis method: {method!r}"
            )  # pragma: no cover

        return alpha_nm

    def get_ovlp(
        self,
        alpha: float,
        w_n_arr: np.ndarray,
        t_n_arr: np.ndarray,
        method: MatVecMethod = MatVecMethod.TOEPLITZ_MATMUL,
        precond_method: PrecondMethod = PrecondMethod.AUTO,
    ) -> np.ndarray | tuple[LinearOperator, LinearOperator]:
        """
        Computes the overlap matrix for the basis functions defined by
        the von Neumann transform. The overlap matrix is defined as:
        S_{(n,m), (i,j)} = exp(
            -alpha/2 * (w_n - w_i)^2
            - (1 / (8 * alpha)) * (t_j - t_m)^2
            + i/2 * (w_i - w_n) * (t_j + t_m)
        )

        Parameters:
            alpha (float): Width of the basis functions.
            w_n_arr (np.ndarray): Angular frequency grid in the
                von Neumann plane.
            t_n_arr (np.ndarray): Time grid in the von Neumann plane.
            method (MatVecMethod): Method to compute the overlap matrix.
                Options are DIRECT, TOEPLITZ_MATMUL, TOEPLITZ_EINSUM,
                TOEPLITZ_BANDED, and GAUSSIAN_STENCIL.

                The DIRECT method computes the overlap matrix
                as a dense array.

                The TOEPLITZ_MATMUL method computes the overlap
                matrix as a LinearOperator that exploits the
                block Toeplitz structure of the matrix to compute
                matrix-vector products efficiently. Products
                of each block with a vector are computed using
                batched matrix multiplication.

                The TOEPLITZ_EINSUM method computes the overlap
                matrix as a LinearOperator that exploits the
                block Toeplitz structure of the matrix to compute
                matrix-vector products efficiently. Products
                of each block with a vector are computed using
                the einsum function.

                The TOEPLITZ_BANDED method uses the same block FFT,
                retaining the main and four adjacent diagonals on
                each side of each Fourier-domain block.

                The GAUSSIAN_STENCIL method keeps overlap entries
                within four outer and inner grid offsets and uses a
                sparse matrix-vector product without an FFT.

            precond_method (PrecondMethod): Preconditioner for iterative
                solvers. AUTO preserves the default for each matrix-vector
                method: dense circulant for TOEPLITZ_MATMUL and
                TOEPLITZ_EINSUM, banded circulant for TOEPLITZ_BANDED,
                and incomplete Cholesky with zero fill for
                GAUSSIAN_STENCIL. NONE selects identity. All explicit
                preconditioners can be chosen independently of the
                matrix-vector method.

        Returns:
            ovlp (np.ndarray | tuple[LinearOperator, LinearOperator]):
                Overlap matrix for the basis functions. If the method
                is DIRECT, the overlap matrix is returned as a dense array.
                Otherwise, it is returned as a tuple of LinearOperators
                (s_op, m_op) where s_op is the overlap operator
                that can be used to compute matrix-vector products,
                and m_op is the preconditioner.
        """

        if method is MatVecMethod.DIRECT:
            if precond_method not in (PrecondMethod.AUTO, PrecondMethod.NONE):
                raise ValueError(
                    "DIRECT method does not use a preconditioner."
                )
            return _get_ovlp_direct(alpha, w_n_arr, t_n_arr)
        elif method in (
            MatVecMethod.TOEPLITZ_MATMUL,
            MatVecMethod.TOEPLITZ_EINSUM,
            MatVecMethod.TOEPLITZ_BANDED,
            MatVecMethod.GAUSSIAN_STENCIL,
        ):
            return _get_ovlp_linop(
                alpha,
                w_n_arr,
                t_n_arr,
                method,
                precond_method,
            )
        else:
            assert False, (
                f"Unknown matvec method: {method!r}"
            )  # pragma: no cover

    def solve_ovlp(
        self,
        ovlp: np.ndarray | tuple[LinearOperator, LinearOperator],
        alpha_nm: np.ndarray,
        method: SolverMethod = SolverMethod.CG,
        rtol: float = 1e-10,
        atol: float = 0.0,
        maxiter: int = 1000,
    ) -> np.ndarray:
        """
        Solves the linear system S * q_nm = alpha_nm, where S is the
        overlap matrix, alpha_nm is the projection of the signal onto the
        basis functions, and q_nm is the von Neumann coefficients.
        Parameters:
            ovlp (np.ndarray | tuple[LinearOperator, LinearOperator]):
                Overlap matrix for the basis functions. If the method
                is DIRECT, the overlap matrix is a dense array.
                Otherwise, it is a tuple of LinearOperators
                (s_op, m_op) where s_op is the overlap operator
                that can be used to compute matrix-vector products,
                and m_op is the preconditioner.
            alpha_nm (np.ndarray): Projection of the signal onto the
                basis functions.
            method (SolverMethod): Method to solve the linear system.
                Options are DIRECT, CG, BICGSTAB, and LGMRES.
                The DIRECT method solves the linear system using GESV.
                The CG method uses the Conjugate Gradient method,
                the BICGSTAB method uses the BIConjugate Gradient
                STABilised method,
                and the LGMRES method uses the Loose Generalised
                Minimum RESidual method.
                All iterative methods require the overlap matrix
                to be a tuple of LinearOperators.
            rtol, atol (float): Relative and absolute tolerances for the
                iterative solver. For convergence of the system A @ x = b,
                norm(b - A @ x) <= max(rtol*norm(b), atol)
                must be satisfied.
            maxiter (int): Maximum number of iterations for the iterative
                solver. If the solver does not converge within this
                number of iterations, a warning is printed.
        Returns:
            q_nm (np.ndarray): Von Neumann coefficients, solution of the
                linear system S * q_nm = alpha_nm.
        """

        if method is SolverMethod.DIRECT:
            if not isinstance(ovlp, np.ndarray):
                raise TypeError(
                    "DIRECT method requires the overlap matrix to be "
                    "a dense array."
                )
            q_nm = np.linalg.solve(ovlp, alpha_nm.flatten())
        elif method in (
            SolverMethod.CG,
            SolverMethod.BICGSTAB,
            SolverMethod.LGMRES,
        ):
            if not isinstance(ovlp, tuple):
                raise TypeError(
                    "Iterative methods require the overlap matrix "
                    "to be a tuple of two LinearOperators."
                )
            else:
                if len(ovlp) != 2 or not all(
                    isinstance(op, LinearOperator) for op in ovlp
                ):
                    raise ValueError(
                        "Iterative methods require the overlap matrix "
                        "to be a tuple of two LinearOperators."
                    )
            s_op, m_op = ovlp
            if method is SolverMethod.CG:
                solver = cg
            elif method is SolverMethod.BICGSTAB:
                solver = bicgstab
            elif method is SolverMethod.LGMRES:
                solver = lgmres

            q_nm, info = solver(
                s_op,
                alpha_nm.flatten(),
                M=m_op,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
            if info < 0:
                print(
                    "Warning: Illegal input or breakdown in the solver."
                )  # pragma: no cover
            elif info > 0:
                print(
                    f"Warning: Solver did not converge after {info} iterations."
                )  # pragma: no cover
        else:
            assert False, (
                f"Unknown solver method: {method!r}"
            )  # pragma: no cover

        q_nm = q_nm.reshape(self.k, self.k)

        return q_nm

    def transform(
        self,
        signal: np.ndarray,
        basis_method: BasisMethod = BasisMethod.FFT,
        matvec_method: MatVecMethod = MatVecMethod.TOEPLITZ_MATMUL,
        solver_method: SolverMethod = SolverMethod.CG,
        rtol: float = 1e-10,
        atol: float = 0.0,
        maxiter: int = 1000,
        precond_method: PrecondMethod = PrecondMethod.AUTO,
    ) -> np.ndarray:
        """
        Computes the von Neumann coefficients for the input signal.
        The coefficients are computed as:
        q_nm = S^{-1} * alpha_nm, where
        S is the overlap matrix, and
        alpha_nm is the projection of the signal onto the basis functions.

        Parameters:
            signal (np.ndarray): Input signal in the frequency domain.
            basis_method (BasisMethod): Method to compute the projection
                of the signal onto the basis functions.
            matvec_method (MatVecMethod): Method to compute the overlap matrix.
            solver_method (SolverMethod): Method to solve the linear system.
            precond_method (PrecondMethod): Preconditioner for iterative
                solvers. AUTO preserves the default for each matvec method.
            rtol, atol (float): Relative and absolute tolerances for the
                iterative solver.
            maxiter (int): Maximum number of iterations for the iterative
                solver.

        Returns:
            q_nm (np.ndarray): Von Neumann coefficients, solution of the
                linear system S * q_nm = alpha_nm.
        """

        # Get the projection of the signal onto the basis functions
        alpha_nm = self.get_signal_projection(
            self.w_grid,
            self.w_n_arr,
            self.t_n_arr,
            self.alpha,
            signal,
            basis_method,
        )

        # Get the overlap matrix
        ovlp = self.get_ovlp(
            self.alpha,
            self.w_n_arr,
            self.t_n_arr,
            matvec_method,
            precond_method,
        )

        # Solve the linear system to get the von Neumann coefficients
        q_nm = self.solve_ovlp(
            ovlp,
            alpha_nm,
            solver_method,
            rtol,
            atol,
            maxiter,
        )

        return q_nm

    def inverse_transform(
        self,
        q_nm: np.ndarray,
        method: BasisMethod = BasisMethod.FFT,
    ) -> np.ndarray:
        """
        Reconstructs the signal from the von Neumann coefficients.
        The signal is reconstructed as:
        signal(w) = \\sum_{n,m} q_nm * alpha_nm (w)
        where
        alpha_nm (w) = (2 * alpha / pi)^(1/4)
                       * exp(-alpha * (w - w_n)^2 - i * t_m * (w - w_n))
        Parameters:
            q_nm (np.ndarray): Von Neumann coefficients.
            method (BasisMethod): Method to reconstruct the signal.
                Options are DIRECT, FACTORISE, and FFT.

                The DIRECT method reconstructs the signal by
                evaluating the basis functions at the grid points
                and computing the dot product with the von Neumann
                coefficients. If the basis functions were not
                evaluated before, they are evaluated now and stored
                in memory.

                The FACTORISE method reconstructs the signal
                by factorising the basis functions and applying
                each factor to the von Neumann coefficients
                sequentially. The evaluated basis functions are not
                stored in memory.

                The FFT method reconstructs the signal using the
                Fast Fourier Transform. The evaluated basis
                functions are not stored in memory. While this
                method is typically the fastest, it can introduce
                (slightly) more numerical noise due to
                the FFT algorithm.
        Returns:
            signal (np.ndarray): Reconstructed signal in the frequency domain.
        """

        if method is BasisMethod.DIRECT:
            alpha_nmo = self._get_basis_functions(
                self.w_grid, self.w_n_arr, self.t_n_arr, self.alpha
            )
            signal = _reconstruct_signal(q_nm, alpha_nmo)
        elif method is BasisMethod.FACTORISE:
            signal = _reconstruct_signal_factorise(
                q_nm,
                self.w_grid,
                self.w_n_arr,
                self.t_n_arr,
                self.alpha,
            )
        elif method is BasisMethod.FFT:
            signal = _reconstruct_signal_fft(
                q_nm,
                self.w_grid,
                self.w_n_arr,
                self.t_n_arr,
                self.alpha,
            )
        else:
            assert False, (
                f"Unknown basis method: {method!r}"
            )  # pragma: no cover

        return signal
