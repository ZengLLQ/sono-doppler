"""Singular Value Decomposition (SVD) Casorati spatiotemporal clutter filtering."""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass
class SVDMetrics:
    """Diagnostic metrics from SVD clutter separation."""

    total_energy: float
    clutter_energy: float
    blood_energy: float
    noise_energy: float
    clutter_rejection_ratio_db: float
    singular_values: np.ndarray
    clutter_cutoff_rank: int
    noise_cutoff_rank: int


class CasoratiSVDFilter:
    """Spatiotemporal SVD clutter filter using the Casorati matrix formulation.

    Separates slow-time ultrasound ensemble into tissue clutter, blood flow,
    and thermal noise subspaces via eigen-decomposition of the space-time correlation.

    Parameters
    ----------
    clutter_rank : int or float, optional
        Rank cutoff below which singular components are attributed to tissue clutter.
        If int >= 1, specifies exact number of singular vectors to discard.
        If float between 0.0 and 1.0, specifies fraction of total ranks.
        If None, estimated adaptively from singular value spectrum slope.
    noise_rank : int or float, optional
        Rank cutoff above which singular components are attributed to electronic noise.
        If None, all components above clutter_rank are retained.
    """

    def __init__(
        self,
        clutter_rank: int | float | None = 6,
        noise_rank: int | float | None = None,
    ) -> None:
        self.clutter_rank = clutter_rank
        self.noise_rank = noise_rank

    def filter_ensemble(
        self,
        ensemble: np.ndarray,
    ) -> tuple[np.ndarray, SVDMetrics]:
        """Filter a 3D slow-time complex ultrasound frame ensemble.

        Parameters
        ----------
        ensemble : np.ndarray
            Complex or real ultrasound ensemble of shape (nz, nx, nt) or (M, nt),
            where nt is the number of slow-time acquisitions.

        Returns
        -------
        blood_ensemble : np.ndarray
            Clutter-filtered blood ensemble with same shape as input.
        metrics : SVDMetrics
            Spectral diagnostic metrics including Clutter Rejection Ratio.
        """
        original_shape = ensemble.shape
        if ensemble.ndim == 3:
            nz, nx, nt = original_shape
            # Flatten spatial dimensions into Casorati matrix M x nt
            casorati = ensemble.reshape((nz * nx, nt))
        elif ensemble.ndim == 2:
            casorati = ensemble
            nt = casorati.shape[1]
        else:
            raise ValueError(f"Expected 2D or 3D ensemble, got shape {original_shape}")

        # Compute economy/thin SVD: C = U * diag(S) * V^H
        # U: (M, nt), s: (nt,), Vh: (nt, nt)
        U, s, Vh = np.linalg.svd(casorati, full_matrices=False)

        # Determine cutoff ranks
        r_clutter = self._determine_clutter_rank(s, nt)
        r_noise = self._determine_noise_rank(s, nt)

        if r_clutter >= r_noise:
            r_clutter = max(1, r_noise - 1)

        # Compute energy partitions
        s_squared = s**2
        total_energy = float(np.sum(s_squared))
        clutter_energy = float(np.sum(s_squared[:r_clutter]))
        blood_energy = float(np.sum(s_squared[r_clutter:r_noise]))
        noise_energy = float(np.sum(s_squared[r_noise:])) if r_noise < nt else 0.0

        if blood_energy + noise_energy > 0:
            crr_db = 10.0 * math.log10((clutter_energy + 1e-12) / (blood_energy + noise_energy + 1e-12))
        else:
            crr_db = 0.0

        # Blood subspace reconstruction: C_blood = U[:, r_c:r_n] * diag(s[r_c:r_n]) * Vh[r_c:r_n, :]
        U_blood = U[:, r_clutter:r_noise]
        s_blood = s[r_clutter:r_noise]
        Vh_blood = Vh[r_clutter:r_noise, :]

        # Scaled matrix multiplication
        casorati_blood = (U_blood * s_blood[np.newaxis, :]) @ Vh_blood

        # Reshape to original dimensions
        blood_ensemble = casorati_blood.reshape(original_shape)

        metrics = SVDMetrics(
            total_energy=total_energy,
            clutter_energy=clutter_energy,
            blood_energy=blood_energy,
            noise_energy=noise_energy,
            clutter_rejection_ratio_db=crr_db,
            singular_values=s,
            clutter_cutoff_rank=r_clutter,
            noise_cutoff_rank=r_noise,
        )

        return blood_ensemble, metrics

    def _determine_clutter_rank(self, s: np.ndarray, nt: int) -> int:
        """Resolve clutter cutoff rank."""
        if self.clutter_rank is None:
            # Adaptive detection: Find knee in log singular value spectrum
            # Tissue clutter has steep drop, noise/blood is flatter
            log_s = np.log(np.maximum(s, 1e-12))
            diffs = np.diff(log_s)
            # Find point where slope flattens significantly
            min_rank = max(1, int(0.05 * nt))
            max_rank = min(nt - 2, int(0.40 * nt))
            if max_rank > min_rank:
                best_rank = min_rank + int(np.argmax(diffs[min_rank:max_rank])) + 1
                return best_rank
            return max(1, int(0.10 * nt))

        if isinstance(self.clutter_rank, float):
            return max(1, min(nt - 1, int(round(self.clutter_rank * nt))))
        return max(1, min(nt - 1, int(self.clutter_rank)))

    def _determine_noise_rank(self, s: np.ndarray, nt: int) -> int:
        """Resolve noise cutoff rank."""
        if self.noise_rank is None:
            return nt

        if isinstance(self.noise_rank, float):
            return max(1, min(nt, int(round(self.noise_rank * nt))))
        return max(1, min(nt, int(self.noise_rank)))
