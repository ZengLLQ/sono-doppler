"""Unit tests for transducer geometry and acoustic excitation pulse."""

import numpy as np
import pytest

from sono_doppler.transducer import TransducerConfig, UltrasoundPhysics


def test_transducer_physics_constants():
    assert UltrasoundPhysics.sound_speed == 1540.0
    assert UltrasoundPhysics.blood_density == 1060.0


def test_transducer_geometry():
    cfg = TransducerConfig(num_elements=64, pitch=0.0003, center_frequency=5.0e6)
    assert len(cfg.element_positions_x) == 64
    assert np.isclose(cfg.wavelength, 1540.0 / 5.0e6)
    assert np.isclose(cfg.element_positions_x[0], -cfg.element_positions_x[-1])
    assert np.isclose(cfg.total_aperture, 63 * 0.0003)


def test_gabor_pulse_generation():
    cfg = TransducerConfig(center_frequency=5.0e6, sampling_rate=40.0e6)
    t, pulse = cfg.generate_pulse()

    assert len(t) == len(pulse)
    assert np.isclose(np.max(np.abs(pulse)), 1.0)
    assert np.isclose(t[len(t) // 2], 0.0, atol=1e-10)
    # Symmetry check: Gabor pulse is symmetric around t=0
    assert np.isclose(pulse[0], pulse[-1], atol=1e-2)
