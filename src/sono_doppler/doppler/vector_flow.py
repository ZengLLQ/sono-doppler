"""Multi-angle steered plane-wave 2D vector Doppler flow estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np

from sono_doppler.doppler.kasai import KasaiDopplerEstimator
from sono_doppler.transducer import UltrasoundPhysics


@dataclass
class VectorFlowResult:
    """Quantitative results of 2D Vector Doppler velocity field."""

    vx: np.ndarray  # Lateral velocity component [m/s]
    vz: np.ndarray  # Axial velocity component [m/s]
    magnitude: np.ndarray  # Total velocity speed |v| [m/s]
    angle: np.ndarray  # Flow direction angle in radians [-pi, pi]
    power: np.ndarray  # Doppler power
    turbulence: np.ndarray  # Spectral variance


class VectorDopplerEstimator:
    """2D Vector velocity flow estimator using complementary plane-wave steering.

    Synthesizes transmit beams at symmetric steer angles (+theta, -theta)
    to overcome Doppler beam-to-flow angle blindness and reconstruct true 2D velocity vectors.

    Parameters
    ----------
    steer_angle_rad : float
        Plane-wave steering angle magnitude [radians] (e.g. 10 deg = 0.1745 rad).
    prf : float
        Pulse Repetition Frequency per angle [Hz].
    center_frequency : float
        Transducer center frequency [Hz].
    sound_speed : float
        Speed of sound in tissue [m/s]. Default 1540 m/s.
    spatial_kernel : tuple[int, int], optional
        Spatial smoothing window size (axial, lateral).
    """

    def __init__(
        self,
        steer_angle_rad: float,
        prf: float,
        center_frequency: float,
        sound_speed: float = UltrasoundPhysics.sound_speed,
        spatial_kernel: tuple[int, int] | None = (3, 3),
    ) -> None:
        self.theta = abs(float(steer_angle_rad))
        if self.theta < 1e-4:
            raise ValueError("Steering angle must be strictly positive (e.g. >= 0.05 rad).")

        self.prf = float(prf)
        self.f0 = float(center_frequency)
        self.c = float(sound_speed)
        self.kasai = KasaiDopplerEstimator(
            prf=self.prf,
            center_frequency=self.f0,
            sound_speed=self.c,
            spatial_kernel=spatial_kernel,
        )

    def estimate_vector_flow(
        self,
        ensemble_neg_angle: np.ndarray,
        ensemble_pos_angle: np.ndarray,
        power_threshold_db: float | None = 20.0,
    ) -> VectorFlowResult:
        """Reconstruct 2D velocity vectors from complementary steer angles.

        Parameters
        ----------
        ensemble_neg_angle : np.ndarray
            Complex ensemble acquired at angle -theta, shape (nz, nx, nt).
        ensemble_pos_angle : np.ndarray
            Complex ensemble acquired at angle +theta, shape (nz, nx, nt).
        power_threshold_db : float, optional
            Power threshold in dB to mask non-vessel regions.

        Returns
        -------
        result : VectorFlowResult
            Reconstructed vx, vz, magnitude, angle, and power maps.
        """
        # Estimate Doppler frequency shifts for each steered ensemble
        res_neg = self.kasai.estimate(ensemble_neg_angle, power_threshold_db=None)
        res_pos = self.kasai.estimate(ensemble_pos_angle, power_threshold_db=None)

        fd1 = res_neg.mean_frequency  # -theta
        fd2 = res_pos.mean_frequency  # +theta

        # Closed-form inversion:
        # fd1 = (f0 / c) * (-vx * sin(theta) + vz * (1 + cos(theta)))
        # fd2 = (f0 / c) * (+vx * sin(theta) + vz * (1 + cos(theta)))
        #
        # fd2 - fd1 = (2 * f0 / c) * vx * sin(theta)
        # -> vx = (c * (fd2 - fd1)) / (2 * f0 * sin(theta))
        #
        # fd2 + fd1 = (2 * f0 / c) * vz * (1 + cos(theta))
        # -> vz = (c * (fd2 + fd1)) / (2 * f0 * (1 + cos(theta)))

        sin_t = math.sin(self.theta)
        cos_t = math.cos(self.theta)

        vx = (self.c * (fd2 - fd1)) / (2.0 * self.f0 * sin_t)
        vz = (self.c * (fd2 + fd1)) / (2.0 * self.f0 * (1.0 + cos_t))

        magnitude = np.sqrt(vx**2 + vz**2)
        angle = np.arctan2(vz, vx)

        # Average power and turbulence
        power = 0.5 * (res_neg.power + res_pos.power)
        turbulence = 0.5 * (res_neg.variance + res_pos.variance)

        # Apply power threshold mask
        if power_threshold_db is not None:
            max_p = np.max(power)
            if max_p > 0:
                p_db = 10.0 * np.log10(np.maximum(power / max_p, 1e-12))
                mask = p_db < -abs(power_threshold_db)
                vx[mask] = 0.0
                vz[mask] = 0.0
                magnitude[mask] = 0.0
                angle[mask] = 0.0
                turbulence[mask] = 0.0

        return VectorFlowResult(
            vx=vx,
            vz=vz,
            magnitude=magnitude,
            angle=angle,
            power=power,
            turbulence=turbulence,
        )
