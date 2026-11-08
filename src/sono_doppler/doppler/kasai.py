"""Kasai 2D complex autocorrelation Doppler frequency, velocity, and turbulence estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from sono_doppler.transducer import UltrasoundPhysics


@dataclass
class DopplerResult:
    """Quantitative results of Doppler autocorrelation estimation."""

    mean_frequency: np.ndarray  # Mean Doppler frequency [Hz]
    velocity: np.ndarray  # Estimated axial velocity [m/s]
    power: np.ndarray  # Doppler signal power / amplitude squared
    variance: np.ndarray  # Flow turbulence / normalized spectral variance [0, 1]
    nyquist_velocity: float  # Maximum unambiguous velocity [m/s]


class KasaiDopplerEstimator:
    """Kasai autocorrelation estimator for color flow and pulsed-wave Doppler.

    Parameters
    ----------
    prf : float
        Pulse Repetition Frequency [Hz].
    center_frequency : float
        Transducer ultrasound center frequency [Hz].
    sound_speed : float
        Acoustic speed in tissue [m/s]. Default 1540 m/s.
    spatial_kernel : tuple[int, int], optional
        Spatial smoothing boxcar kernel size (axial_size, lateral_size).
        If provided, smoothes autocorrelation estimates over local speckle cells.
    """

    def __init__(
        self,
        prf: float,
        center_frequency: float,
        sound_speed: float = UltrasoundPhysics.sound_speed,
        spatial_kernel: tuple[int, int] | None = (3, 3),
    ) -> None:
        self.prf = float(prf)
        self.f0 = float(center_frequency)
        self.c = float(sound_speed)
        self.spatial_kernel = spatial_kernel

    @property
    def nyquist_velocity(self) -> float:
        """Maximum detectable axial velocity without Doppler aliasing [m/s]."""
        return (self.c * self.prf) / (4.0 * self.f0)

    def estimate(
        self,
        complex_ensemble: np.ndarray,
        power_threshold_db: float | None = 25.0,
    ) -> DopplerResult:
        """Estimate mean Doppler velocity, power, and variance.

        Parameters
        ----------
        complex_ensemble : np.ndarray
            3D complex analytic/demodulated ensemble of shape (nz, nx, nt),
            where nt is the slow-time temporal dimension.
        power_threshold_db : float, optional
            Threshold below max power in dB for masking velocity in non-flow regions.

        Returns
        -------
        result : DopplerResult
            Contains 2D velocity, frequency, power, and variance maps.
        """
        if complex_ensemble.ndim != 3:
            raise ValueError(f"Expected 3D complex ensemble, got shape {complex_ensemble.shape}")

        nz, nx, nt = complex_ensemble.shape
        if nt < 2:
            raise ValueError(f"Ensemble length nt={nt} must be >= 2 for lag-1 autocorrelation.")

        # Lag-1 temporal autocorrelation: R1 = sum_t (z[t+1] * conj(z[t]))
        z_curr = complex_ensemble[:, :, :-1]
        z_next = complex_ensemble[:, :, 1:]
        r1 = np.mean(z_next * np.conj(z_curr), axis=-1)

        # Lag-0 autocorrelation (average signal power): R0 = sum_t (|z[t]|^2)
        r0 = np.mean(np.abs(complex_ensemble) ** 2, axis=-1)

        # Optional spatial kernel filtering to reduce acoustic speckle noise
        if self.spatial_kernel is not None:
            kz, kx = self.spatial_kernel
            from scipy.ndimage import uniform_filter

            r1_real = uniform_filter(r1.real, size=(kz, kx))
            r1_imag = uniform_filter(r1.imag, size=(kz, kx))
            r1 = r1_real + 1j * r1_imag
            r0 = uniform_filter(r0, size=(kz, kx))

        # Mean Doppler angular phase shift: omega_d in [-pi, pi]
        # Angle of lag-1 autocorrelation
        omega_d = np.angle(r1)

        # Mean Doppler frequency in Hz
        # fd = (PRF / (2*pi)) * omega_d
        mean_freq = (self.prf / (2.0 * math.pi)) * omega_d

        # Axial velocity from Doppler equation: v_z = (c * fd) / (2 * f0)
        velocity = (self.c * mean_freq) / (2.0 * self.f0)

        # Normalized spectral variance: var = 1 - |R1| / R0
        # In pure single-frequency flow, |R1| = R0 -> var = 0
        # In turbulence/noise, |R1| << R0 -> var -> 1
        eps = 1e-12
        r1_mag = np.abs(r1)
        variance = np.clip(1.0 - (r1_mag / (r0 + eps)), 0.0, 1.0)

        # Mask regions below power threshold
        if power_threshold_db is not None:
            max_p = np.max(r0)
            if max_p > 0:
                p_db = 10.0 * np.log10(np.maximum(r0 / max_p, 1e-12))
                mask = p_db < -abs(power_threshold_db)
                velocity[mask] = 0.0
                mean_freq[mask] = 0.0
                variance[mask] = 0.0

        return DopplerResult(
            mean_frequency=mean_freq,
            velocity=velocity,
            power=r0,
            variance=variance,
            nyquist_velocity=self.nyquist_velocity,
        )
