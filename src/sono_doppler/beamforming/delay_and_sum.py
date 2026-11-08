"""Plane-wave ultrafast ultrasound Delay-and-Sum (DAS) beamforming."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from sono_doppler.transducer import TransducerConfig, UltrasoundPhysics


@dataclass
class PlaneWaveConfig:
    """Steered plane-wave transmit sequence configuration.

    Parameters
    ----------
    angles : list[float] or np.ndarray
        List of plane-wave transmit steering angles [radians].
    prf : float
        Pulse Repetition Frequency (slow-time sampling rate) [Hz].
    """

    angles: np.ndarray = None
    prf: float = 5000.0  # 5 kHz PRF for ultrafast Doppler

    def __post_init__(self) -> None:
        if self.angles is None:
            # Default to 3-angle compounding (-5 deg, 0 deg, +5 deg)
            self.angles = np.radians(np.array([-5.0, 0.0, 5.0]))
        else:
            self.angles = np.asarray(self.angles, dtype=float)

    @property
    def num_angles(self) -> int:
        return len(self.angles)


class DelayAndSumBeamformer:
    """GPU/CPU vector-optimized Delay-and-Sum beamformer for plane-wave ultrasound.

    Parameters
    ----------
    transducer : TransducerConfig
        Transducer physical parameters.
    x_grid : np.ndarray
        Lateral pixel coordinates for beamformed image grid [meters].
    z_grid : np.ndarray
        Axial depth coordinates for beamformed image grid [meters].
    f_number : float
        Dynamic receive aperture F-number (z / active_width). Default 1.2.
    window : str
        Apodization window type: 'hanning', 'hamming', or 'boxcar'.
    """

    def __init__(
        self,
        transducer: TransducerConfig,
        x_grid: np.ndarray,
        z_grid: np.ndarray,
        f_number: float = 1.2,
        window: str = "hanning",
    ) -> None:
        self.transducer = transducer
        self.x_grid = np.asarray(x_grid, dtype=float)
        self.z_grid = np.asarray(z_grid, dtype=float)
        self.f_number = f_number
        self.window = window.lower()

        self.nz = len(self.z_grid)
        self.nx = len(self.x_grid)
        self.c = UltrasoundPhysics.sound_speed
        self.fs = self.transducer.sampling_rate

        # 2D coordinate meshgrids: shape (nz, nx)
        self.Z, self.X = np.meshgrid(self.z_grid, self.x_grid, indexing="ij")
        self.x_elements = self.transducer.element_positions_x  # (num_elem,)

    def compute_receive_delays_and_apodization(self) -> tuple[np.ndarray, np.ndarray]:
        """Precompute receive travel times and apodization weights.

        Returns
        -------
        tau_rx : np.ndarray
            Shape (nz, nx, num_elem) containing receive travel times in seconds.
        weights : np.ndarray
            Shape (nz, nx, num_elem) containing normalized apodization weights.
        """
        num_elem = len(self.x_elements)
        # Broadcast (nz, nx, 1) and (1, 1, num_elem)
        X_3d = self.X[:, :, np.newaxis]
        Z_3d = self.Z[:, :, np.newaxis]
        elem_x = self.x_elements[np.newaxis, np.newaxis, :]

        # Distance from (x, z) to element (x_elem, 0)
        dist_rx = np.sqrt((X_3d - elem_x) ** 2 + Z_3d**2)
        tau_rx = dist_rx / self.c

        # Dynamic aperture calculation: max lateral distance for F-number
        # active_width = z / f_number -> max_dist = z / (2 * f_number)
        max_dist = np.maximum(Z_3d / (2.0 * self.f_number), self.transducer.pitch)
        lateral_offset = np.abs(X_3d - elem_x)

        in_aperture = lateral_offset <= max_dist

        if self.window == "boxcar":
            weights = in_aperture.astype(float)
        elif self.window == "hamming":
            norm_pos = lateral_offset / (max_dist + 1e-12)
            weights = (0.54 + 0.46 * np.cos(np.pi * norm_pos)) * in_aperture
        else:  # hanning (default)
            norm_pos = lateral_offset / (max_dist + 1e-12)
            weights = 0.5 * (1.0 + np.cos(np.pi * norm_pos)) * in_aperture

        # Normalize weights across elements for each pixel
        weight_sum = np.sum(weights, axis=-1, keepdims=True)
        weight_sum[weight_sum == 0] = 1.0
        weights = weights / weight_sum

        return tau_rx, weights

    def beamform_single_angle(
        self,
        rf_data: np.ndarray,
        angle: float,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Delay-and-Sum beamform a single plane-wave acquisition.

        Parameters
        ----------
        rf_data : np.ndarray
            Raw RF channel signal matrix of shape (num_samples, num_elem).
        angle : float
            Steering angle of transmit plane wave [radians].
        t0 : float
            Start time of sampling window relative to excitation [seconds].

        Returns
        -------
        beamformed_image : np.ndarray
            2D complex/real beamformed RF matrix of shape (nz, nx).
        """
        tau_rx, weights = self.compute_receive_delays_and_apodization()

        # Plane-wave transmit delay: tau_tx(x, z) = (x*sin(theta) + z*cos(theta)) / c
        tau_tx = (self.X * math.sin(angle) + self.Z * math.cos(angle)) / self.c
        tau_tx_3d = tau_tx[:, :, np.newaxis]

        # Total two-way propagation delay
        tau_total = tau_tx_3d + tau_rx

        # Convert continuous delay to discrete sample indices
        sample_indices = (tau_total - t0) * self.fs
        num_samples, num_elem = rf_data.shape

        # Linear interpolation of channel RF data
        idx_floor = np.floor(sample_indices).astype(np.int64)
        idx_ceil = idx_floor + 1
        frac = sample_indices - idx_floor

        # Mask valid in-bounds samples
        valid_mask = (idx_floor >= 0) & (idx_ceil < num_samples)

        # Safe indexing by clipping
        idx_floor_safe = np.clip(idx_floor, 0, num_samples - 1)
        idx_ceil_safe = np.clip(idx_ceil, 0, num_samples - 1)

        # Gather channel RF values using advanced indexing
        elem_indices = np.arange(num_elem)[np.newaxis, np.newaxis, :]
        val_floor = rf_data[idx_floor_safe, elem_indices]
        val_ceil = rf_data[idx_ceil_safe, elem_indices]

        interpolated_rf = (1.0 - frac) * val_floor + frac * val_ceil
        interpolated_rf = interpolated_rf * valid_mask * weights

        # Sum across receive transducer elements
        beamformed_image = np.sum(interpolated_rf, axis=-1)
        return beamformed_image

    def beamform_compounded(
        self,
        rf_ensemble: list[np.ndarray] | np.ndarray,
        pw_config: PlaneWaveConfig,
        t0: float = 0.0,
    ) -> np.ndarray:
        """Coherent Plane-Wave Compounding (CPWC) across multiple steering angles.

        Parameters
        ----------
        rf_ensemble : list of np.ndarray, or np.ndarray of shape (num_angles, num_samples, num_elem)
            RF data for each plane-wave angle.
        pw_config : PlaneWaveConfig
            Angles and sequence metadata.
        t0 : float
            Sampling start time offset [seconds].

        Returns
        -------
        compounded_image : np.ndarray
            Coherently summed beamformed RF matrix of shape (nz, nx).
        """
        accumulated = np.zeros((self.nz, self.nx), dtype=float)
        for i, angle in enumerate(pw_config.angles):
            angle_rf = rf_ensemble[i]
            beamformed = self.beamform_single_angle(angle_rf, angle, t0=t0)
            accumulated += beamformed

        return accumulated / pw_config.num_angles
