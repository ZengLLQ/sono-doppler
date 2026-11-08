"""Unit tests for Delay-and-Sum beamforming and B-mode envelope detection."""

import numpy as np
import pytest

from sono_doppler.beamforming.delay_and_sum import DelayAndSumBeamformer, PlaneWaveConfig
from sono_doppler.beamforming.envelope import EnvelopeDetector
from sono_doppler.transducer import TransducerConfig


def test_delay_and_sum_geometry():
    transducer = TransducerConfig(num_elements=32, pitch=0.0003, sampling_rate=20.0e6)
    x_grid = np.linspace(-0.005, 0.005, 16)
    z_grid = np.linspace(0.010, 0.020, 20)

    beamformer = DelayAndSumBeamformer(transducer, x_grid, z_grid, f_number=1.5, window="hanning")
    tau_rx, weights = beamformer.compute_receive_delays_and_apodization()

    assert tau_rx.shape == (20, 16, 32)
    assert weights.shape == (20, 16, 32)
    assert np.all(tau_rx > 0)
    # Weights should sum to 1.0 along element axis (or 0 if outside aperture)
    weight_sums = np.sum(weights, axis=-1)
    assert np.all(np.isclose(weight_sums, 1.0) | np.isclose(weight_sums, 0.0))


def test_beamform_synthetic_point_target():
    transducer = TransducerConfig(num_elements=32, pitch=0.0003, sampling_rate=20.0e6)
    x_grid = np.linspace(-0.004, 0.004, 17)
    z_grid = np.linspace(0.012, 0.018, 25)

    beamformer = DelayAndSumBeamformer(transducer, x_grid, z_grid)

    # Point target at (x=0, z=15mm)
    z_target = 0.015
    num_samples = 600
    t = np.arange(num_samples) / transducer.sampling_rate
    c = 1540.0

    # Simulate raw RF received by elements for normal incidence plane wave (theta=0)
    rf_data = np.zeros((num_samples, transducer.num_elements))
    _, pulse = transducer.generate_pulse(duration=4e-7)

    for n, x_elem in enumerate(transducer.element_positions_x):
        # Two-way travel time: tau_tx = z/c, tau_rx = sqrt(x^2 + z^2)/c
        tau = (z_target + np.sqrt(x_elem**2 + z_target**2)) / c
        sample_idx = int(round(tau * transducer.sampling_rate))
        if 0 <= sample_idx < num_samples - len(pulse):
            rf_data[sample_idx : sample_idx + len(pulse), n] += pulse

    img = beamformer.beamform_single_angle(rf_data, angle=0.0)
    assert img.shape == (25, 17)

    # Peak response should be near target center (x=0, z=15mm)
    center_x_idx = 8  # middle of 17 points
    # Find z index closest to 0.015
    center_z_idx = int(np.argmin(np.abs(z_grid - z_target)))

    max_z_idx, max_x_idx = np.unravel_index(np.argmax(np.abs(img)), img.shape)
    assert abs(max_x_idx - center_x_idx) <= 1
    assert abs(max_z_idx - center_z_idx) <= 2


def test_envelope_detection_and_bmode():
    env_det = EnvelopeDetector(dynamic_range_db=40.0)
    # Synthetic RF carrier
    t = np.linspace(0, 1e-5, 200)
    carrier = np.sin(2.0 * np.pi * 5.0e6 * t) * np.exp(-((t - 5e-6) / 1e-6) ** 2)

    env = env_det.extract_envelope(carrier)
    assert np.all(env >= 0.0)
    assert np.isclose(np.max(env), 1.0, atol=0.1)

    bmode = env_det.bmode_image(carrier[:, np.newaxis], axis=0, normalize_unit=True)
    assert np.all(bmode >= 0.0)
    assert np.all(bmode <= 1.0)
