"""Ultrasound transducer geometry, physical constants, and acoustic pulse modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class UltrasoundPhysics:
    """Acoustic constants in biological soft tissue."""

    sound_speed: float = 1540.0  # Mean speed of sound in soft tissue [m/s]
    blood_attenuation: float = 0.5  # Acoustic attenuation [dB / (cm * MHz)]
    blood_density: float = 1060.0  # Density of blood [kg / m^3]


@dataclass
class TransducerConfig:
    """Linear array ultrasound transducer configuration.

    Parameters
    ----------
    num_elements : int
        Number of piezoelectric transducer elements.
    pitch : float
        Center-to-center spacing between adjacent elements [meters].
    center_frequency : float
        Center acoustic carrier frequency [Hz].
    bandwidth_fraction : float
        Fractional -6 dB pulse bandwidth (e.g. 0.70 for 70%).
    sampling_rate : float
        RF analog-to-digital converter sampling frequency [Hz].
    """

    num_elements: int = 64
    pitch: float = 0.0003  # 0.3 mm pitch
    center_frequency: float = 5.0e6  # 5.0 MHz
    bandwidth_fraction: float = 0.70
    sampling_rate: float = 40.0e6  # 40 MHz

    @property
    def wavelength(self) -> float:
        """Acoustic wavelength in soft tissue [meters]."""
        return UltrasoundPhysics.sound_speed / self.center_frequency

    @property
    def element_positions_x(self) -> np.ndarray:
        """Lateral positions of each transducer element centered at x=0 [meters]."""
        indices = np.arange(self.num_elements) - (self.num_elements - 1) / 2.0
        return indices * self.pitch

    @property
    def total_aperture(self) -> float:
        """Total transducer array active lateral width [meters]."""
        return (self.num_elements - 1) * self.pitch

    def generate_pulse(self, duration: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Generate a Gabor wavelet transmit excitation pulse.

        Parameters
        ----------
        duration : float, optional
            Total temporal window length [seconds]. If None, auto-calculated
            from bandwidth.

        Returns
        -------
        t : np.ndarray
            Time vector centered at t=0 [seconds].
        pulse : np.ndarray
            Normalized acoustic pressure waveform.
        """
        f0 = self.center_frequency
        bw_hz = f0 * self.bandwidth_fraction
        # Standard deviation in time domain from Gaussian FWHM
        sigma_t = 1.0 / (2.0 * math.pi * (bw_hz / 2.355))

        if duration is None:
            duration = 6.0 * sigma_t

        dt = 1.0 / self.sampling_rate
        num_samples = int(math.ceil(duration / dt))
        if num_samples % 2 == 0:
            num_samples += 1

        t = (np.arange(num_samples) - (num_samples - 1) // 2) * dt
        envelope = np.exp(-0.5 * (t / sigma_t) ** 2)
        carrier = np.cos(2.0 * math.pi * f0 * t)
        pulse = envelope * carrier
        # Normalize peak amplitude to unity
        pulse = pulse / np.max(np.abs(pulse))
        return t, pulse
