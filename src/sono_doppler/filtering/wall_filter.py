"""Traditional high-pass IIR and polynomial wall filters for Doppler clutter rejection."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


class IIRWallFilter:
    """High-pass Butterworth wall filter applied along the slow-time axis.

    Parameters
    ----------
    cutoff_normalized : float
        Normalized cutoff frequency as fraction of Nyquist (fc / (PRF/2)).
        Typical values are 0.05 to 0.15.
    order : int
        Filter order (default 4).
    """

    def __init__(self, cutoff_normalized: float = 0.08, order: int = 4) -> None:
        self.cutoff_normalized = float(np.clip(cutoff_normalized, 0.001, 0.999))
        self.order = int(order)
        self.b, self.a = butter(self.order, self.cutoff_normalized, btype="high")

    def filter_ensemble(self, ensemble: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply zero-phase high-pass filtering along slow-time axis.

        Parameters
        ----------
        ensemble : np.ndarray
            Input ultrasound ensemble.
        axis : int
            Slow-time axis (default -1).

        Returns
        -------
        filtered : np.ndarray
            High-pass filtered ensemble.
        """
        nt = ensemble.shape[axis]
        # filtfilt requires padlen < nt
        padlen = min(3 * max(len(self.a), len(self.b)), nt - 1)
        if padlen <= 0:
            return ensemble

        if np.iscomplexobj(ensemble):
            real_filt = filtfilt(self.b, self.a, ensemble.real, axis=axis, padlen=padlen)
            imag_filt = filtfilt(self.b, self.a, ensemble.imag, axis=axis, padlen=padlen)
            return real_filt + 1j * imag_filt

        return filtfilt(self.b, self.a, ensemble, axis=axis, padlen=padlen)


class PolynomialWallFilter:
    """Legendre polynomial regression clutter filter along slow time.

    Projects each slow-time ensemble onto low-order orthogonal polynomials
    to model and subtract slow tissue motion.

    Parameters
    ----------
    poly_order : int
        Order of polynomial to subtract (e.g. 1 for linear trend, 2 for quadratic).
    """

    def __init__(self, poly_order: int = 2) -> None:
        self.poly_order = poly_order

    def filter_ensemble(self, ensemble: np.ndarray, axis: int = -1) -> np.ndarray:
        """Subtract polynomial tissue motion trend."""
        nt = ensemble.shape[axis]
        t = np.linspace(-1.0, 1.0, nt)
        # Build Vandermonde basis matrix (nt, order + 1)
        V = np.vander(t, N=self.poly_order + 1, increasing=True)
        # Projection matrix P_clutter = V * (V^T V)^-1 * V^T
        # Blood = (I - P_clutter) * y
        Q, _ = np.linalg.qr(V)
        # Q is orthonormal basis: P_clutter = Q @ Q.T
        P_clutter = Q @ Q.T

        # Apply along slow time
        # Move axis to last, reshape to (M, nt), multiply by (I - P_clutter)
        moved = np.moveaxis(ensemble, axis, -1)
        shape_moved = moved.shape
        flat = moved.reshape((-1, nt))

        # flat @ (I - P_clutter).T = flat - flat @ P_clutter
        filtered_flat = flat - flat @ P_clutter
        filtered_moved = filtered_flat.reshape(shape_moved)
        return np.moveaxis(filtered_moved, -1, axis)
