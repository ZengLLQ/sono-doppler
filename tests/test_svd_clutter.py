"""Unit tests for Casorati SVD clutter filtering and wall filters."""

import numpy as np
import pytest

from sono_doppler.filtering.svd_clutter import CasoratiSVDFilter
from sono_doppler.filtering.wall_filter import IIRWallFilter, PolynomialWallFilter


def test_svd_clutter_energy_separation():
    # Simulate ensemble with high-amplitude static clutter + low-amplitude dynamic signal
    M = 100  # spatial points
    nt = 32  # slow-time pulses

    # Static high-rank tissue clutter (rank 2)
    rng = np.random.default_rng(42)
    u_clutter = rng.normal(0, 1, size=(M, 2))
    v_clutter = rng.normal(0, 1, size=(nt, 2))
    clutter = (u_clutter @ v_clutter.T) * 100.0  # High energy

    # Moving blood signal (rank 4, orthogonal frequencies)
    t = np.arange(nt)
    v_blood = np.column_stack([np.sin(2 * np.pi * 0.25 * t), np.cos(2 * np.pi * 0.25 * t)])
    u_blood = rng.normal(0, 1, size=(M, 2))
    blood = (u_blood @ v_blood.T) * 2.0  # ~34 dB weaker

    total_matrix = clutter + blood

    svd_filter = CasoratiSVDFilter(clutter_rank=2)
    filtered, metrics = svd_filter.filter_ensemble(total_matrix)

    assert filtered.shape == (M, nt)
    # Clutter rejection ratio should be high (> 20 dB)
    assert metrics.clutter_rejection_ratio_db > 20.0
    # Filtered blood matrix should have substantially reduced clutter
    norm_clutter_orig = np.linalg.norm(clutter)
    norm_filtered = np.linalg.norm(filtered)
    assert norm_filtered < 0.1 * norm_clutter_orig


def test_iir_wall_filter():
    # Test high-pass filtering of constant/DC component
    nt = 64
    signal = np.ones((10, nt)) + 0.1 * np.sin(np.linspace(0, 8 * np.pi, nt))
    iir = IIRWallFilter(cutoff_normalized=0.1, order=2)
    filtered = iir.filter_ensemble(signal, axis=-1)

    assert filtered.shape == (10, nt)
    # The DC mean should be attenuated to near zero
    assert np.all(np.abs(np.mean(filtered, axis=-1)) < 0.05)


def test_polynomial_wall_filter():
    nt = 32
    # Quadratic slow-drift clutter + fast oscillation
    t = np.linspace(-1, 1, nt)
    drift = 5.0 * t**2 + 2.0 * t + 10.0
    signal = drift[np.newaxis, :] + 0.5 * np.cos(np.linspace(0, 10 * np.pi, nt))[np.newaxis, :]

    poly = PolynomialWallFilter(poly_order=2)
    filtered = poly.filter_ensemble(signal, axis=-1)

    assert filtered.shape == (1, nt)
    # Mean of filtered signal should be close to zero
    assert abs(float(np.mean(filtered))) < 0.1
