"""Unit tests for clinical carotid flow phantom and stenosis modeling."""

import numpy as np
import pytest

from sono_doppler.phantoms.flow_phantom import CarotidFlowPhantom, StenosisProfile


def test_stenosis_lumen_narrowing():
    profile = StenosisProfile(diameter_reduction=0.70, plaque_center_x=0.0, plaque_length=0.006)
    phantom = CarotidFlowPhantom(vessel_radius_m=0.0035, stenosis=profile)

    r_normal = phantom.get_vessel_radius_at(-0.015)
    r_throat = phantom.get_vessel_radius_at(0.0)

    # At normal segment, radius is full ~3.5 mm
    assert np.isclose(r_normal, 0.0035, atol=1e-4)
    # At throat, radius is reduced by 70% -> 30% of normal
    assert np.isclose(r_throat, 0.0035 * 0.30, atol=1e-4)


def test_velocity_field_stenosis_jet():
    profile = StenosisProfile(diameter_reduction=0.60, plaque_center_x=0.0)
    phantom = CarotidFlowPhantom(
        depth_m=0.020,
        vessel_radius_m=0.0035,
        tilt_angle_deg=0.0,  # Horizontal for simple test
        mean_inflow_velocity=0.30,
        stenosis=profile,
    )

    x_grid = np.array([-0.010, 0.0, 0.010])
    z_grid = np.array([0.020])  # Vessel centerline

    vx, vz, mask = phantom.evaluate_velocity_field(x_grid, z_grid)

    assert np.all(mask)  # Centerline is inside lumen
    # Due to continuity, velocity at throat (x=0) must be significantly higher than at inflow (x=-0.010)
    v_inflow = vx[0, 0]
    v_throat = vx[0, 1]
    assert v_throat > 2.0 * v_inflow


def test_complex_ensemble_generation():
    phantom = CarotidFlowPhantom(
        depth_m=0.020,
        vessel_radius_m=0.0035,
        tilt_angle_deg=15.0,
    )
    x_grid = np.linspace(-0.01, 0.01, 16)
    z_grid = np.linspace(0.015, 0.025, 16)

    ensemble, vx, vz = phantom.generate_complex_ensemble(
        x_grid=x_grid,
        z_grid=z_grid,
        prf=3000.0,
        num_frames=16,
        seed=1,
    )

    assert ensemble.shape == (16, 16, 16)
    assert np.iscomplexobj(ensemble)
    assert vx.shape == (16, 16)
    assert vz.shape == (16, 16)
