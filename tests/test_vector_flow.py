"""Unit tests for multi-angle 2D vector Doppler flow estimation."""

import math
import numpy as np
import pytest

from sono_doppler.doppler.vector_flow import VectorDopplerEstimator


def test_vector_doppler_reconstruction():
    theta = math.radians(10.0)  # 10 degree steer angle
    prf = 6000.0
    f0 = 5.0e6
    c = 1540.0

    estimator = VectorDopplerEstimator(
        steer_angle_rad=theta,
        prf=prf,
        center_frequency=f0,
        spatial_kernel=None,
    )

    # True 2D velocity vector: vx = 0.40 m/s (lateral), vz = 0.15 m/s (axial)
    vx_true = 0.40
    vz_true = 0.15

    # Complementary angle Doppler shifts:
    # fd1 (-theta):
    fd1 = (f0 / c) * (-vx_true * math.sin(theta) + vz_true * (1.0 + math.cos(theta)))
    # fd2 (+theta):
    fd2 = (f0 / c) * (+vx_true * math.sin(theta) + vz_true * (1.0 + math.cos(theta)))

    nt = 64
    t = np.arange(nt) / prf

    ens_neg = np.exp(1j * 2.0 * np.pi * fd1 * t)[np.newaxis, np.newaxis, :]
    ens_pos = np.exp(1j * 2.0 * np.pi * fd2 * t)[np.newaxis, np.newaxis, :]

    result = estimator.estimate_vector_flow(ens_neg, ens_pos, power_threshold_db=None)

    assert np.isclose(result.vx[0, 0], vx_true, rtol=1e-3)
    assert np.isclose(result.vz[0, 0], vz_true, rtol=1e-3)

    speed_expected = math.sqrt(vx_true**2 + vz_true**2)
    assert np.isclose(result.magnitude[0, 0], speed_expected, rtol=1e-3)

    angle_expected = math.atan2(vz_true, vx_true)
    assert np.isclose(result.angle[0, 0], angle_expected, rtol=1e-3)
