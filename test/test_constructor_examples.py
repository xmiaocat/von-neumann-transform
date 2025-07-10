from pathlib import Path
import pytest

from von_neumann_transform import VonNeumannTransform

DATA_DIR = Path(__file__).parent / "data"

NPOINTS_LIST = [16, 256, 4096]
NPOINTS_LIST_TYPEERROR = [16.0, 256.0, 4096.0]
NPOINTS_LIST_VALUEERROR = [-16, -256, -4096]
OMEGA_LIST = [
    (0.0, 1.0),
    (0.0, 10.0),
    (0.0, 100.0),
    (0.5, 1.0),
    (5.0, 10.0),
    (50.0, 100.0),
]
OMEGA_LIST_NEGATIVE = [
    (-1.0, 0.0),
    (-1.0, 1.0),
    (-1.0, -0.5),
]
OMEGA_LIST_REVERSE = [
    (1.0, 0.0),
    (5.0, 1.0),
    (10.0, 5.0),
]


def test_constructor_examples():
    for npoints in NPOINTS_LIST:
        for omega in OMEGA_LIST:
            w_min, w_max = omega
            vnt = VonNeumannTransform(npoints, w_min, w_max)
            assert isinstance(vnt, VonNeumannTransform)
            assert vnt.npoints == npoints
            assert vnt.w_min == w_min
            assert vnt.w_max == w_max


def test_constructor_npoints_type_error_examples():
    for npoints in NPOINTS_LIST_TYPEERROR:
        with pytest.raises(TypeError):
            VonNeumannTransform(npoints, 0.0, 1.0)


def test_constructor_npoints_value_error_examples():
    for npoints in NPOINTS_LIST_VALUEERROR:
        with pytest.raises(ValueError):
            VonNeumannTransform(npoints, 0.0, 1.0)


def test_constructor_omega_negative_examples():
    for omega in OMEGA_LIST_NEGATIVE:
        w_min, w_max = omega
        with pytest.raises(ValueError):
            VonNeumannTransform(16, w_min, w_max)


def test_constructor_omega_reverse_examples():
    for omega in OMEGA_LIST_REVERSE:
        w_min, w_max = omega
        with pytest.raises(ValueError):
            VonNeumannTransform(16, w_min, w_max)
