"""Unit tests for Kasai 2D autocorrelation Doppler velocity estimation."""

import numpy as np
import pytest

from sono_doppler.doppler.kasai import KasaiDopplerEstimator


def test_kasai_doppler_frequency_accuracy():
    prf = 5000.0  # 5 kHz PRF
    f0 = 5.0e6  # 5 MHz center frequency
    c = 1540.0
    estimator = KasaiDopplerEstimator(prf=prf, center_frequency=f0, spatial_kernel=None)

    # Known target velocity: v_true = 0.25 m/s (axial)
    v_true = 0.25
    # Expected Doppler shift: fd = 2 * f0 * v / c
    fd_expected = (2.0 * f0 * v_true) / c  # ~1623.38 Hz

    # Generate pure complex sinusoid ensemble at this frequency
    nt = 64
    t = np.arange(nt) / prf
    phase = 2.0 * np.pi * fd_expected * t

    # 1 pixel ensemble: shape (1, 1, nt)
    complex_signal = np.exp(1j * phase)[np.newaxis, np.newaxis, :]

    result = estimator.estimate(complex_signal, power_threshold_db=None)

    assert np.isclose(result.mean_frequency[0, 0], fd_expected, rtol=1e-3)
    assert np.isclose(result.velocity[0, 0], v_true, rtol=1e-3)
    # Variance should be near zero for monochromatic signal
    assert result.variance[0, 0] < 1e-4


def test_kasai_negative_velocity():
    prf = 4000.0
    f0 = 5.0e6
    estimator = KasaiDopplerEstimator(prf=prf, center_frequency=f0, spatial_kernel=None)

    v_true = -0.15
    fd_expected = (2.0 * f0 * v_true) / 1540.0
    nt = 32
    t = np.arange(nt) / prf
    complex_signal = np.exp(1j * 2.0 * np.pi * fd_expected * t)[np.newaxis, np.newaxis, :]

    res = estimator.estimate(complex_signal, power_threshold_db=None)
    assert np.isclose(res.velocity[0, 0], v_true, rtol=1e-3)


def test_kasai_power_thresholding():
    prf = 4000.0
    f0 = 5.0e6
    estimator = KasaiDopplerEstimator(prf=prf, center_frequency=f0, spatial_kernel=None)

    nt = 16
    # High power signal at (0, 0), low power noise at (0, 1)
    sig = np.zeros((1, 2, nt), dtype=complex)
    sig[0, 0, :] = 10.0 * np.exp(1j * np.linspace(0, 4 * np.pi, nt))
    sig[0, 1, :] = 0.001 * np.exp(1j * np.linspace(0, 4 * np.pi, nt))

    res = estimator.estimate(sig, power_threshold_db=20.0)
    assert res.velocity[0, 0] != 0.0
    assert res.velocity[0, 1] == 0.0  # Masked due to low power
