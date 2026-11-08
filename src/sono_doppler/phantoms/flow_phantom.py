"""Clinical flow phantom modeling carotid artery bifurcation and stenotic jet hemodynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from sono_doppler.transducer import TransducerConfig, UltrasoundPhysics


@dataclass
class StenosisProfile:
    """Carotid plaque stenosis specifications.

    Parameters
    ----------
    diameter_reduction : float
        Fractional diameter reduction at narrowest throat (e.g. 0.70 for 70% stenosis).
    plaque_center_x : float
        Lateral coordinate of stenosis throat [meters].
    plaque_length : float
        Characteristic spatial length of plaque [meters].
    """

    diameter_reduction: float = 0.65  # 65% diameter stenosis (~88% area reduction)
    plaque_center_x: float = 0.000  # Centered at x=0
    plaque_length: float = 0.006  # 6 mm


class CarotidFlowPhantom:
    """Clinical carotid artery hemodynamics and acoustic scattering phantom.

    Models vessel geometry, wall pulsatility, stenotic jet acceleration,
    post-stenotic recirculation zone, and tissue-to-blood contrast ratio.

    Parameters
    ----------
    depth_m : float
        Depth of vessel center below transducer face [meters]. Default 22 mm.
    vessel_radius_m : float
        Normal unobstructed internal lumen radius [meters]. Default 3.5 mm.
    tilt_angle_deg : float
        Vessel tilt angle relative to array surface [degrees]. Default 15 degrees.
    mean_inflow_velocity : float
        Inflow mean blood velocity in healthy segment [m/s]. Default 0.35 m/s.
    stenosis : StenosisProfile, optional
        Plaque stenosis parameters. If None, normal unstenosed vessel.
    tissue_to_blood_ratio_db : float
        Tissue clutter to blood echo power contrast [dB]. Default 35 dB.
    """

    def __init__(
        self,
        depth_m: float = 0.022,
        vessel_radius_m: float = 0.0035,
        tilt_angle_deg: float = 15.0,
        mean_inflow_velocity: float = 0.35,
        stenosis: StenosisProfile | None = None,
        tissue_to_blood_ratio_db: float = 35.0,
    ) -> None:
        self.depth_m = depth_m
        self.vessel_radius_m = vessel_radius_m
        self.tilt_angle = math.radians(tilt_angle_deg)
        self.mean_inflow_velocity = mean_inflow_velocity
        self.stenosis = stenosis or StenosisProfile()
        self.contrast_ratio = 10.0 ** (tissue_to_blood_ratio_db / 20.0)

    def get_vessel_radius_at(self, x: float | np.ndarray) -> np.ndarray:
        """Compute internal lumen radius along lateral position x."""
        x_arr = np.asarray(x, dtype=float)
        r0 = self.vessel_radius_m
        if self.stenosis is None or self.stenosis.diameter_reduction <= 0:
            return np.full_like(x_arr, r0)

        xc = self.stenosis.plaque_center_x
        sig = self.stenosis.plaque_length / 2.355
        narrowing = self.stenosis.diameter_reduction * np.exp(-0.5 * ((x_arr - xc) / sig) ** 2)
        return r0 * (1.0 - narrowing)

    def get_vessel_centerline_z(self, x: float | np.ndarray) -> np.ndarray:
        """Compute axial depth z of vessel centerline at lateral position x."""
        x_arr = np.asarray(x, dtype=float)
        # z(x) = depth + x * tan(tilt)
        return self.depth_m + x_arr * math.tan(self.tilt_angle)

    def evaluate_velocity_field(
        self,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ground-truth 2D blood velocity vector field (vx, vz, inside_lumen_mask).

        Parameters
        ----------
        x_grid : np.ndarray
            Lateral coordinates [meters].
        z_grid : np.ndarray
            Axial coordinates [meters].

        Returns
        -------
        vx : np.ndarray
            Lateral velocity matrix [m/s], shape (nz, nx).
        vz : np.ndarray
            Axial velocity matrix [m/s], shape (nz, nx).
        lumen_mask : np.ndarray
            Boolean mask indicating points inside vessel lumen, shape (nz, nx).
        """
        Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")
        center_z = self.get_vessel_centerline_z(X)
        local_r_max = self.get_vessel_radius_at(X)

        # Distance from point to vessel centerline along normal
        # Normal distance: perpendicular to vessel axis
        cos_tilt = math.cos(self.tilt_angle)
        dist_from_axis = np.abs(Z - center_z) * cos_tilt

        lumen_mask = dist_from_axis <= local_r_max

        # Flow conservation: Q = v_mean(x) * pi * R(x)^2 = const
        # v_mean(x) = v_inflow * (R0 / R(x))^2
        r0 = self.vessel_radius_m
        area_ratio = (r0 / np.maximum(local_r_max, 1e-6)) ** 2
        local_v_peak = 2.0 * self.mean_inflow_velocity * area_ratio

        # Parabolic laminar flow profile inside lumen: v(r) = v_peak * (1 - (r/R)^2)
        norm_r = dist_from_axis / np.maximum(local_r_max, 1e-6)
        v_magnitude = local_v_peak * np.maximum(0.0, 1.0 - norm_r**2) * lumen_mask

        # Decompose into lateral (x) and axial (z) components along vessel axis
        vx = v_magnitude * math.cos(self.tilt_angle)
        vz = v_magnitude * math.sin(self.tilt_angle)

        # Post-stenotic recirculation modeling:
        # Downstream of stenosis throat (x > plaque_center + plaque_length),
        # flow separation creates a localized reverse vortex near the outer wall
        if self.stenosis and self.stenosis.diameter_reduction > 0.5:
            xc = self.stenosis.plaque_center_x
            post_stenosis = (X > (xc + 0.5 * self.stenosis.plaque_length)) & (
                X < (xc + 2.5 * self.stenosis.plaque_length)
            )
            near_wall = norm_r > 0.65
            recirc_mask = post_stenosis & near_wall & lumen_mask
            # Invert velocity direction in recirculation zone
            vx[recirc_mask] = -0.35 * vx[recirc_mask]
            vz[recirc_mask] = -0.35 * vz[recirc_mask]

        return vx, vz, lumen_mask

    def generate_complex_ensemble(
        self,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        prf: float,
        num_frames: int = 64,
        center_frequency: float = 5.0e6,
        steer_angle_rad: float = 0.0,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic 3D complex demodulated ultrasound ensemble (nz, nx, nt).

        Simulates static tissue speckle, high-amplitude specular vessel walls,
        and Doppler phase-shifted erythrocytes flowing through the vessel.

        Parameters
        ----------
        x_grid : np.ndarray
            Lateral grid [meters].
        z_grid : np.ndarray
            Axial depth grid [meters].
        prf : float
            Pulse Repetition Frequency [Hz].
        num_frames : int
            Number of slow-time acquisitions (default 64).
        center_frequency : float
            Transducer center frequency [Hz].
        steer_angle_rad : float
            Transmit beam angle [radians].
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        ensemble : np.ndarray
            Complex ensemble of shape (nz, nx, num_frames).
        vx : np.ndarray
            Ground truth lateral velocity field.
        vz : np.ndarray
            Ground truth axial velocity field.
        """
        rng = np.random.default_rng(seed)
        vx, vz, lumen_mask = self.evaluate_velocity_field(x_grid, z_grid)
        nz, nx = len(z_grid), len(x_grid)

        # 1. Static tissue background speckle: high amplitude Rayleigh scattering
        tissue_real = rng.normal(0, 1.0, size=(nz, nx))
        tissue_imag = rng.normal(0, 1.0, size=(nz, nx))
        tissue_field = (tissue_real + 1j * tissue_imag) * self.contrast_ratio
        # Zero out inside lumen where blood resides
        tissue_field[lumen_mask] *= 0.05

        # 2. Strong specular vessel walls:
        Z, X = np.meshgrid(z_grid, x_grid, indexing="ij")
        center_z = self.get_vessel_centerline_z(X)
        local_r_max = self.get_vessel_radius_at(X)
        cos_tilt = math.cos(self.tilt_angle)
        dist_from_axis = np.abs(Z - center_z) * cos_tilt
        wall_distance = np.abs(dist_from_axis - local_r_max)
        wall_mask = wall_distance < 0.0004  # 0.4 mm wall boundary
        tissue_field[wall_mask] += self.contrast_ratio * 3.0

        # Slow tissue pulsation (slight wall motion at 1.2 Hz cardiac rhythm)
        t_slow = np.arange(num_frames) / prf
        wall_pulse_phase = 2.0 * np.pi * 1.2 * t_slow

        # 3. Blood flow Doppler phase progression:
        # Effective projected Doppler shift along transmit beam direction:
        # fd = (f0 / c) * (vx * sin(theta) + vz * (1 + cos(theta)))
        c = UltrasoundPhysics.sound_speed
        sin_t = math.sin(steer_angle_rad)
        cos_t = math.cos(steer_angle_rad)
        fd = (center_frequency / c) * (vx * sin_t + vz * (1.0 + cos_t))

        # Shape (nz, nx, num_frames)
        # Phase at frame t: 2*pi*fd * t + random initial scatterer phase
        phase_init = rng.uniform(0, 2.0 * np.pi, size=(nz, nx))
        t_3d = t_slow[np.newaxis, np.newaxis, :]
        fd_3d = fd[:, :, np.newaxis]
        phase_t = 2.0 * np.pi * fd_3d * t_3d + phase_init[:, :, np.newaxis]

        blood_field = np.exp(1j * phase_t) * lumen_mask[:, :, np.newaxis]

        # Combine tissue clutter (with slight cardiac wall oscillation) + blood + thermal noise
        ensemble = np.zeros((nz, nx, num_frames), dtype=np.complex128)
        for t_idx in range(num_frames):
            tissue_t = tissue_field * (1.0 + 0.08 * math.sin(wall_pulse_phase[t_idx]))
            noise = (rng.normal(0, 0.2, (nz, nx)) + 1j * rng.normal(0, 0.2, (nz, nx)))
            ensemble[:, :, t_idx] = tissue_t + blood_field[:, :, t_idx] + noise

        return ensemble, vx, vz
