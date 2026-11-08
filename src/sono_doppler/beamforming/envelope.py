"""Ultrasound envelope detection, analytic signal computation, and logarithmic compression."""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert


class EnvelopeDetector:
    """Hilbert envelope detector and logarithmic dynamic range compressor.

    Parameters
    ----------
    dynamic_range_db : float
        Visual display dynamic range in decibels (e.g., 50.0 dB).
    """

    def __init__(self, dynamic_range_db: float = 50.0) -> None:
        self.dynamic_range_db = float(dynamic_range_db)

    def compute_analytic_signal(self, rf_data: np.ndarray, axis: int = 0) -> np.ndarray:
        """Compute the complex analytic signal via Hilbert transform.

        Parameters
        ----------
        rf_data : np.ndarray
            Real RF data array.
        axis : int
            Axial propagation axis along which to apply the Hilbert transform.

        Returns
        -------
        analytic_signal : np.ndarray
            Complex analytic signal where real is RF and imag is Hilbert transform.
        """
        return hilbert(rf_data, axis=axis)

    def extract_envelope(self, rf_data: np.ndarray, axis: int = 0) -> np.ndarray:
        """Extract the high-frequency acoustic pulse envelope.

        Parameters
        ----------
        rf_data : np.ndarray
            Real or complex RF data.
        axis : int
            Axial propagation axis.

        Returns
        -------
        envelope : np.ndarray
            Non-negative amplitude envelope.
        """
        if np.iscomplexobj(rf_data):
            return np.abs(rf_data)
        analytic = self.compute_analytic_signal(rf_data, axis=axis)
        return np.abs(analytic)

    def log_compress(
        self,
        envelope: np.ndarray,
        reference_max: float | None = None,
    ) -> np.ndarray:
        """Logarithmically compress the envelope into decibel display units.

        Parameters
        ----------
        envelope : np.ndarray
            Non-negative acoustic envelope.
        reference_max : float, optional
            Normalization maximum value. If None, uses max of `envelope`.

        Returns
        -------
        b_mode_db : np.ndarray
            Log-compressed values normalized in [0, dynamic_range_db].
        """
        if reference_max is None:
            reference_max = float(np.max(envelope))
        if reference_max <= 0:
            return np.zeros_like(envelope)

        normalized = np.maximum(envelope / reference_max, 1e-12)
        db_values = 20.0 * np.log10(normalized)
        # Shift and clip to [0, dynamic_range_db]
        compressed = np.clip(db_values + self.dynamic_range_db, 0.0, self.dynamic_range_db)
        return compressed

    def bmode_image(
        self,
        rf_image: np.ndarray,
        axis: int = 0,
        normalize_unit: bool = True,
    ) -> np.ndarray:
        """End-to-end B-mode image generation from beamformed RF data.

        Parameters
        ----------
        rf_image : np.ndarray
            2D beamformed RF matrix.
        axis : int
            Axial dimension (typically axis 0).
        normalize_unit : bool
            If True, normalizes output to [0, 1].

        Returns
        -------
        bmode : np.ndarray
            Compressed B-mode raster.
        """
        envelope = self.extract_envelope(rf_image, axis=axis)
        compressed = self.log_compress(envelope)
        if normalize_unit and self.dynamic_range_db > 0:
            return compressed / self.dynamic_range_db
        return compressed
