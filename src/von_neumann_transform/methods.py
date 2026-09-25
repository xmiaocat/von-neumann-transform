from enum import Enum, auto


class BasisMethod(Enum):
    DIRECT = auto()
    FACTORISE = auto()
    FFT = auto()


class MatVecMethod(Enum):
    DIRECT = auto()
    TOEPLITZ_MATMUL = auto()
    TOEPLITZ_EINSUM = auto()
    TOEPLITZ_BANDED = auto()
    GAUSSIAN_STENCIL = auto()


class PrecondMethod(Enum):
    AUTO = auto()
    NONE = auto()
    CIRCULANT_DENSE = auto()
    CIRCULANT_BANDED = auto()
    INCOMPLETE_CHOLESKY = auto()


class SolverMethod(Enum):
    DIRECT = auto()
    CG = auto()
    BICGSTAB = auto()
    LGMRES = auto()
