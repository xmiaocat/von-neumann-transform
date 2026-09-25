# Von Neumann Transform

[![Tests](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/tests.yml/badge.svg)](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/tests.yml)
[![Ruff](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/ruff.yml/badge.svg)](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/ruff.yml)
[![Mypy](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/mypy.yml/badge.svg)](https://github.com/xmiaocat/von-neumann-transform/actions/workflows/mypy.yml)
[![Codecov](https://codecov.io/gh/xmiaocat/von-neumann-transform/branch/main/graph/badge.svg)](https://codecov.io/gh/xmiaocat/von-neumann-transform)
[![PyPI version](https://img.shields.io/pypi/v/von-neumann-transform.svg)](https://pypi.org/project/von-neumann-transform/)

A Python package for efficient computation of the von Neumann representation 
of a signal given in the frequency domain.
The von Neumann representation is a joint time-frequency representation 
defined in
- S. Fechner, F. Dimler, T. Brixner, G. Gerber, J. Tannor,
  *Opt. Express* **2007**, *15*, 15387–15401.
- F. Dimler, S. Fechner, A. Rodenberg, T. Brixner, J. Tannor,
  *New J. Phys.* **2009**, *11*, 105052.

## TODO
- [ ] Add consistency tests for solvers.
- [ ] Add consistency tests for inverse transform.


## Features
- Grid generation: Build uniform time-frequency grids in the
  von Neumann plane.
- Signal projection: Compute the projection of the frequency-domain 
  signal onto the von Neumann basis functions *via*:
  - Direct method: precompute and store the basis functions.
  - Factorisation method: sequentially compute projections using
    a factorisation of the basis.
  - FFT-based method: compute the projection with the help of FFT.
- Overlap assembly & solvers: Solve for von Neumann coefficients
  accounting for basis overlap using:
  - Direct solve: assemble the overlap matrix and apply a
    direct linear solver.
  - Iterative solve: apply the overlap through a block FFT or
    a local Gaussian stencil.
- Signal reconstruction: Reconstruct the original frequency-domain
  signal from von Neumann coefficients.
- Type-safe API: Enums (`BasisMethod`, `MatVecMethod`, `PrecondMethod`,
  `SolverMethod`)
  select algorithms, all functions and methods include type hints.


## Installation

Install from PyPI:
```bash
pip install von-neumann-transform
```

Install the latest development version from GitHub:
```bash
pip install git+https://github.com/xmiaocat/von-neumann-transform.git
```

For development, clone the repository and run:
```bash
uv sync
uv run pytest
```

The development tools are installed by default from the `dev` dependency
group.

Install the additional dependencies for running and analysing benchmarks with:

```bash
uv sync --group benchmark
```


## Quickstart
```python
import numpy as np
from von_neumann_transform import VonNeumannTransform

NPOINTS = 4096  # length of the signal
W_MIN = 0.0  # minimum angular frequency
W_MAX = 5.0  # maximum angular frequency

# Your signal in the frequency domain
signal = np.random.rand(NPOINTS) + 1.0j * np.random.rand(NPOINTS)

# Create a Von Neumann Transform instance
vnt = VonNeumannTransform(NPOINTS, W_MIN, W_MAX)

# Compute the von Neumann representation of the signal
q_nm = vnt.transform(signal)

# Reconstruct the original signal from the von Neumann coefficients
signal_recon = vnt.inverse_transform(q_nm)
```


## Threading and Performance

Some NumPy and SciPy operations used by this package are parallelised by
default. For repeated small operations, the computation may be dominated by
thread-management overhead, so single-threaded execution is recommended.
For dense direct solves and large factorisations, multiple threads may be
beneficial.


## Benchmarks

The benchmark suite compares the overlap operators and preconditioners, as
well as the direct, factorised, and FFT-based projection and reconstruction
methods. Timing and peak-memory measurements run in separate fresh subprocesses
to keep measurement overhead out of the timed operations.

Run the default benchmark configuration and generate its summaries and figures
with:

```bash
uv run --group benchmark python benchmarks/run_benchmarks.py
uv run --group benchmark python benchmarks/analyze_benchmarks.py
```

Raw measurements are written to `benchmarks/results/`, and summaries and plots
to `benchmarks/reports/`; both directories are ignored by Git. See the
[benchmark documentation](benchmarks/README.md) for study definitions,
configuration overrides, smoke runs, and output details.


## Algorithmic Details
This section provides an overview of each method and its computational complexity.
Suppose the signal has length $N$. Then $k = \sqrt{N}$ is chosen, so the
von Neumann representation is a $k \times k$ matrix.

### Signal Projection
The signal projection is defined as
```math
  \langle \alpha_{\omega_i t_j} | \epsilon \rangle
  = \int_{-\infty}^{\infty} \alpha^*_{\omega_i t_j}(\omega) \epsilon(\omega) \mathrm{d}\omega
```
where $\alpha_{\omega_i t_j}(\omega)$ are the von Neumann basis functions
```math
  \alpha_{\omega_n t_m}(\omega) 
  = \left(\frac{2\alpha}{\pi}\right)^{1/4} 
    \exp \left[ -\alpha (\omega - \omega_n)^2 - \mathrm{i} t_m (\omega - \omega_n) \right]
```
and $\epsilon(\omega)$ is the signal in the frequency domain.

In the discrete world, the basis functions $\alpha_{\omega_n t_m}(\omega)$
are sampled at $N$ points in the frequency domain, and therefore have the
shape `(k, k, N)`. For uniform grid spacing $\Delta\omega$, the projection
is approximated as
```math
  \alpha_{nm} \approx \Delta\omega
  \sum_{p=0}^{N-1} \alpha^*_{\omega_n t_m}(\omega_p)\epsilon(\omega_p)\,.
```
The time complexity of this operation is $\mathcal{O}(k^2 N)$.
The space complexity is also $\mathcal{O}(k^2 N)$ if the basis functions
are precomputed and stored. 

The basis functions can be factorised as
```math
  \begin{align}
  \alpha_{\omega_n t_m}(\omega) 
  &= \left(\frac{2\alpha}{\pi}\right)^{1/4} 
    \exp \left[ -\alpha (\omega - \omega_n)^2 \right] 
    \exp \left[ -\mathrm{i} t_m \omega \right]
    \exp \left[ \mathrm{i} t_m \omega_n \right] \\
  &=: \left(\frac{2\alpha}{\pi}\right)^{1/4} 
    \alpha_{\omega_n}(\omega) \alpha_{t_m}(\omega) \alpha_{\omega_n t_m}\,.
  \end{align}
```
The discretised version of the factors $\alpha_{\omega_n}(\omega)$,
$\alpha_{t_m}(\omega)$ and $\alpha_{\omega_n t_m}$ have the shapes
`(k, N)`, `(k, N)` and `(k, k)` respectively. This way, 
although the time complexity of the projection is still
$\mathcal{O}(k^2 N)$, the space complexity can be reduced to
$\mathcal{O}(k N)$.

The factorisation shows another way to compute the projection.
Because of the factor $\exp \left[ \mathrm{i} t_m \omega_n \right]$,
this projection can be viewed as the (inverse) Fourier transform of the
signal $\epsilon(\omega)$ multiplied by the factor $\alpha_{\omega_n}(\omega)$
with a phase correction afterwards.
This reduces the time complexity to $\mathcal{O}(k N \log N)$.

The computational complexity of all three methods is summarised in the table below:
| Method          | Time Complexity   | Space Complexity |
|-----------------|-------------------|------------------|
| direct (`BasisMethod.DIRECT`) | $\mathcal{O}(k^2 N) = \mathcal{O}(N^2)$ | $\mathcal{O}(k^2 N) = \mathcal{O}(N^2)$ |
| factorisation (`BasisMethod.FACTORISE`) | $\mathcal{O}(k^2 N) = \mathcal{O}(N^2)$ | $\mathcal{O}(k N) = \mathcal{O}(N^{3/2})$ |
| FFT-based (`BasisMethod.FFT`) | $\mathcal{O}(k N \log N) = \mathcal{O}(N^{3/2} \log N)$ | $\mathcal{O}(k N) = \mathcal{O}(N^{3/2})$ |

### Overlap Assembly & Solvers
The $N=k^2$ basis functions are not orthogonal. Their coefficients are
therefore obtained by solving a linear system involving the $N\times N$
overlap matrix
```math
  S_{(n,m),(i,j)}
  = \langle \alpha_{\omega_n t_m}|\alpha_{\omega_i t_j}\rangle
  = \exp \left[
      -\frac{\alpha}{2}(\omega_n-\omega_i)^2
      -\frac{(t_m-t_j)^2}{8\alpha}
      +\frac{\mathrm{i}}{2}(\omega_i-\omega_n)(t_m+t_j)
    \right].
```
The matrix is Hermitian, and the system is
```math
  \sum_{i,j} S_{(n,m),(i,j)}q_{(i,j)}
  = \langle \alpha_{\omega_n t_m}|\epsilon\rangle.
```

The overlap has two useful properties. First, its blocks repeat along the
frequency-grid direction. Second, the magnitude decays rapidly with
separation in both the frequency and time directions. To see both, let
$d=n-i$ and $\ell=m-j$, and denote the uniform grid spacings by
$\Delta\omega$ and $\Delta t$. With
$a=\alpha(\Delta\omega)^2/2$ and $b=(\Delta t)^2/(8\alpha)$,
the entries can be written as
```math
  S_{(n,m),(i,j)}
  = e^{-ad^2-b\ell^2}
    \exp\left[-\mathrm{i}d\Delta\omega
      \left(t_{\min}+\frac{m+j}{2}\Delta t\right)\right].
```
The phase has unit magnitude, so the Gaussian factor determines the
localisation. On the grid used here,
$a=b=\frac{\pi}{2}(1-k^{-2})$. The block indexed by $d$ is independent
of the absolute frequency indices $n$ and $i$, whereas its phase still
depends on the sum of the time indices $m+j$.

One option is to assemble the full matrix and solve the system directly.
This requires $\mathcal{O}(N^2)$ storage and
$\mathcal{O}(N^3)$ time. Alternatively, an iterative solver can use a
matrix-free operator that evaluates rows only as needed. This reduces
storage to $\mathcal{O}(N)$, but each matrix-vector product still requires
$\mathcal{O}(N^2)$ work. This row approach is not implemented. The
implemented iterative methods reduce the product cost by exploiting either
the repeated blocks or the Gaussian localisation.

For the block-FFT methods, the dependence on $d$ gives a block Toeplitz
matrix. A circulant embedding in the block direction turns the
matrix-vector product into FFTs and independent products with $k\times k$
Fourier-domain blocks. Keeping these blocks dense gives
$\mathcal{O}(k^3)$ work per product and
$\mathcal{O}(k^3)$ storage. The `MatVecMethod.TOEPLITZ_MATMUL` and
`MatVecMethod.TOEPLITZ_EINSUM` methods differ only in how they contract
those dense blocks with vectors. A circulant preconditioner is obtained
by factorising the Fourier blocks, which adds
$\mathcal{O}(k^4)$ preparation work.

The Fourier-domain blocks retain the factor $e^{-b(m-j)^2}$.
Consequently, `MatVecMethod.TOEPLITZ_BANDED` keeps only entries with
$|m-j|\leq R$ in each Fourier block. The outer block FFT is unchanged,
but the products in Fourier space now use bands rather than dense
matrices. The retained blocks also admit a banded Cholesky circulant
preconditioner. For fixed $R$, the FFTs determine the
$\mathcal{O}(N\log N)$ cost per matrix-vector product, while banded
multiplication and preconditioning require $\mathcal{O}(N)$ work.

Alternatively, `MatVecMethod.GAUSSIAN_STENCIL` truncates the original
overlap matrix to $|d|\leq R$ and $|\ell|\leq R$. Each output couples
to at most $(2R+1)^2$ neighbouring coefficients. The retained entries
are stored in a sparse matrix and applied directly, with no circulant
embedding or FFT. For fixed $R$, both storage and each matrix-vector
product scale as $\mathcal{O}(N)$.

Both approximations discard terms controlled by the Gaussian envelope.
The Fourier-banded method truncates only the inner separation after the
block FFT, while the stencil truncates both separations before multiplication.
The current implementation uses $R=4$ for both methods.

The preconditioner can be selected independently of the matrix-vector method.
By default, `PrecondMethod.AUTO` chooses a preconditioner that matches the
selected method's scaling. Notice that this is not necessarily the fastest choice.
See [`PrecondMethod`](#precondmethod) for the exact mapping and the available 
explicit choices. Direct solves do not use a preconditioner.

The costs of the preconditioners are shown below. Banded bounds assume
fixed $R$.

| Preconditioner | Setup time | Application time | Storage |
|----------------|------------|------------------|---------|
| identity (`PrecondMethod.NONE`) | $\mathcal{O}(1)$ | $\mathcal{O}(N)$ | $\mathcal{O}(1)$ |
| dense circulant (`PrecondMethod.CIRCULANT_DENSE`) | $\mathcal{O}(N^2)$ | $\mathcal{O}(N^{3/2})$ | $\mathcal{O}(N^{3/2})$ |
| banded circulant (`PrecondMethod.CIRCULANT_BANDED`) | $\mathcal{O}(N\log N)$ | $\mathcal{O}(N\log N)$ | $\mathcal{O}(N)$ |
| IC(0) (`PrecondMethod.INCOMPLETE_CHOLESKY`) | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ | $\mathcal{O}(N)$ |

Storage excludes the input and output vectors. The solver costs below
exclude preconditioning but include overlap-operator assembly and, for
iterative methods, $m$ steps. The value of $m$ can depend on both the
solver and the preconditioner. For a complete iterative solve, the
preconditioner's setup cost is added once, and its application cost is
incurred on each step. Total storage combines the operator and
preconditioner storage. The banded and stencil bounds assume fixed $R$.

| Overlap method | Solver Time | Operator storage |
|----------------|-------------|------------------|
| direct (`MatVecMethod.DIRECT`) | $\mathcal{O}(N^3)$ | $\mathcal{O}(N^2)$ |
| dense block FFT (`MatVecMethod.TOEPLITZ_MATMUL` or `MatVecMethod.TOEPLITZ_EINSUM`) | $\mathcal{O}(N^{3/2}\log N+mN^{3/2})$ | $\mathcal{O}(N^{3/2})$ |
| banded block FFT (`MatVecMethod.TOEPLITZ_BANDED`) | $\mathcal{O}((m+1)N\log N)$ | $\mathcal{O}(N)$ |
| Gaussian stencil (`MatVecMethod.GAUSSIAN_STENCIL`) | $\mathcal{O}((m+1)N)$ | $\mathcal{O}(N)$ |

### Signal Reconstruction
Signal reconstruction synthesises the frequency-domain signal from the
coefficients obtained after solving the overlap system:
```math
  \epsilon_{\mathrm{rec}}(\omega)
  = \sum_{n,m} q_{nm}\alpha_{\omega_n t_m}(\omega).
```
The same direct, factorised, and FFT-based strategies apply, with the
corresponding computational complexities of signal projection.


## API Reference

### Class `VonNeumannTransform`

`VonNeumannTransform(npoints: int, omega_min: float, omega_max: float)`

Initialise a Von Neumann Transform instance.

- **Parameters:**
  - `npoints`: Number of points in the frequency domain signal.
  - `omega_min`: Minimum angular frequency.
  - `omega_max`: Maximum angular frequency.

```
transform(
    signal: np.ndarray,
    basis_method: BasisMethod = BasisMethod.FFT,
    matvec_method: MatVecMethod = MatVecMethod.TOEPLITZ_MATMUL,
    solver_method: SolverMethod = SolverMethod.CG,
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: int = 1000,
    precond_method: PrecondMethod = PrecondMethod.AUTO,
) -> np.ndarray
```

Computes the von Neumann representation of the signal.

- Parameters:
    - signal (np.ndarray): Input signal in the frequency domain.
    - basis_method (BasisMethod): Method to compute the projection
        of the signal onto the basis functions.
        Possible values are:
        - `BasisMethod.DIRECT`: Directly compute the projection
          by precomputing and storing the basis functions.
        - `BasisMethod.FACTORISE`: Use the factorisation of the basis
          functions to compute the projection.
        - `BasisMethod.FFT`: Use the FFT to compute the projection.
    - matvec_method (MatVecMethod): Method to compute the overlap matrix.
        Possible values are:
        - `MatVecMethod.DIRECT`: Directly assemble the overlap matrix.
           Requires `SolverMethod.DIRECT`.
        - `MatVecMethod.TOEPLITZ_MATMUL`: Use the Toeplitz structure
          to compute the matrix-vector product.
        - `MatVecMethod.TOEPLITZ_EINSUM`: Use the Toeplitz structure
          to compute the matrix-vector product with einsum.
        - `MatVecMethod.TOEPLITZ_BANDED`: Retain near-diagonal entries
          of the Fourier-domain blocks.
        - `MatVecMethod.GAUSSIAN_STENCIL`: Apply a local overlap stencil
          with a sparse matrix.
        These non-direct methods require an iterative solver
        (`SolverMethod.CG`, `SolverMethod.BICGSTAB`, or `SolverMethod.LGMRES`).
    - precond_method (PrecondMethod): Select the preconditioner independently.
        `AUTO` chooses the method-matched default; see
        [`PrecondMethod`](#precondmethod) for the exact mapping and explicit
        choices. Direct solves do not use a preconditioner.
    - solver_method (SolverMethod): Method to solve the linear system.
        - `SolverMethod.DIRECT`: Use a direct solver. Requires
          `MatVecMethod.DIRECT`.
        - `SolverMethod.CG`: Use the conjugate gradient method.
        - `SolverMethod.BICGSTAB`: Use the biconjugate gradient
          stabilised method.
        - `SolverMethod.LGMRES`: Use the LGMRES method.
        The iterative solvers require
        `MatVecMethod.TOEPLITZ_MATMUL`, `MatVecMethod.TOEPLITZ_EINSUM`,
        `MatVecMethod.TOEPLITZ_BANDED`, or `MatVecMethod.GAUSSIAN_STENCIL`.
    - rtol, atol (float): Relative and absolute tolerances for the
        iterative solver.
    - maxiter (int): Maximum number of iterations for the iterative
        solver.

- Returns:
    - q_nm (np.ndarray): Von Neumann coefficients, solution of the
        linear system S * q_nm = alpha_nm.

```
inverse_transform(
    q_nm: np.ndarray,
    method: BasisMethod = BasisMethod.FFT,
) -> np.ndarray
```

Reconstructs the signal from the von Neumann coefficients.

- Parameters:
    - q_nm (np.ndarray): Von Neumann coefficients.
    - method (BasisMethod): Method to reconstruct the signal.
        Possible values are:
        - `BasisMethod.DIRECT`: Directly reconstruct the signal
          by precomputing and storing the basis functions.
        - `BasisMethod.FACTORISE`: Use the factorisation of the basis
          functions to reconstruct the signal.
        - `BasisMethod.FFT`: Use the FFT to reconstruct the signal.
- Returns:
    - signal (np.ndarray): Reconstructed signal in the frequency domain.

### Enums

#### `BasisMethod`

Selects how basis functions are handled in the projection and reconstruction:

- `BasisMethod.DIRECT`: Precompute and store the basis functions.
- `BasisMethod.FACTORISE`: Use the factorisation of the basis functions.
- `BasisMethod.FFT`: Use the FFT to compute the projection and reconstruction.

#### `MatVecMethod`

Selects how the overlap operator is applied:

- `MatVecMethod.DIRECT`: Directly assemble the overlap matrix and multiply.
- `MatVecMethod.TOEPLITZ_MATMUL`: Use the Toeplitz structure to compute
  the matrix-vector product.
- `MatVecMethod.TOEPLITZ_EINSUM`: Use the Toeplitz structure to compute
  the matrix-vector product with einsum.
- `MatVecMethod.TOEPLITZ_BANDED`: Use the block FFT with banded
  Fourier-domain blocks.
- `MatVecMethod.GAUSSIAN_STENCIL`: Apply a local Gaussian stencil
  directly, with no FFT in the matrix-vector product.

#### `PrecondMethod`

Selects the preconditioner for an iterative solver:

- `PrecondMethod.AUTO`: Use the dense circulant preconditioner for
  `TOEPLITZ_MATMUL` and `TOEPLITZ_EINSUM`, the banded circulant preconditioner
  for `TOEPLITZ_BANDED`, and incomplete Cholesky for `GAUSSIAN_STENCIL`.
- `PrecondMethod.NONE`: Use the identity preconditioner.
- `PrecondMethod.CIRCULANT_DENSE`: Factorise dense Fourier blocks.
- `PrecondMethod.CIRCULANT_BANDED`: Factorise banded Fourier blocks.
- `PrecondMethod.INCOMPLETE_CHOLESKY`: Factorise the Gaussian stencil
  without fill, then apply two sparse triangular solves.

#### `SolverMethod`

Selects the linear solver for overlap inversion:

- `SolverMethod.DIRECT`: Use a direct solver.
- `SolverMethod.CG`: Use the conjugate gradient method.
- `SolverMethod.BICGSTAB`: Use the biconjugate gradient stabilised method.
- `SolverMethod.LGMRES`: Use the LGMRES method.

## License
Distributed under the Apache License 2.0.
See `LICENSE` for more information.
