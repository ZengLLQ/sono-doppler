"""ASCII terminal visualization for ultrasound B-mode, Color Doppler, and duplex spectra."""

from __future__ import annotations

import math
import numpy as np


class AsciiDopplerVisualizer:
    """Terminal ASCII renderer for clinical ultrasound flow visualization."""

    @staticmethod
    def render_color_doppler_map(
        velocity_map: np.ndarray,
        v_max: float,
        width: int = 50,
        height: int = 18,
    ) -> str:
        """Render 2D color Doppler velocity map using pure ASCII characters.

        Forward flow towards transducer is shown with '+', '^', '#'.
        Reverse flow away from transducer is shown with '-', '=', '~'.
        Zero flow is shown with '.'.

        Parameters
        ----------
        velocity_map : np.ndarray
            2D axial velocity matrix in m/s.
        v_max : float
            Maximum color velocity display scale [m/s].
        width : int
            ASCII column width.
        height : int
            ASCII row height.

        Returns
        -------
        output : str
            Formatted ASCII representation.
        """
        nz, nx = velocity_map.shape
        lines = []

        # Header
        lines.append(f"+{'-' * (width + 2)}+")
        lines.append(f"|  COLOR DOPPLER VELOCITY MAP  (Scale: +/-{v_max:.2f} m/s) {' ' * max(0, width - 48)}|")
        lines.append(f"+{'-' * (width + 2)}+")

        row_indices = np.linspace(0, nz - 1, height).astype(int)
        col_indices = np.linspace(0, nx - 1, width).astype(int)

        sampled = velocity_map[np.ix_(row_indices, col_indices)]

        # Character ramp for positive (forward/red) and negative (reverse/blue) flow
        pos_chars = [".", "+", "*", "#", "@"]
        neg_chars = [".", "-", "=", "%", "&"]

        for r in range(height):
            row_str = []
            for c in range(width):
                val = sampled[r, c]
                norm = val / (v_max + 1e-12)

                if abs(val) < 0.05 * v_max:
                    char = "."
                elif norm > 0:
                    idx = min(len(pos_chars) - 1, int(round(norm * (len(pos_chars) - 1))))
                    char = pos_chars[idx]
                else:
                    idx = min(len(neg_chars) - 1, int(round(abs(norm) * (len(neg_chars) - 1))))
                    char = neg_chars[idx]
                row_str.append(char)
            lines.append(f"| {''.join(row_str)} |")

        lines.append(f"+{'-' * (width + 2)}+")
        lines.append("| Legend: [+] Forward (red)  [-] Reverse (blue)  [#/@] High Jet  [.] Wall/No-Flow |")
        return "\n".join(lines)

    @staticmethod
    def render_spectral_doppler(
        time_points: np.ndarray,
        velocities: np.ndarray,
        width: int = 54,
        height: int = 12,
    ) -> str:
        """Render a spectral Doppler velocity-time waveform trace.

        Parameters
        ----------
        time_points : np.ndarray
            Time array in seconds.
        velocities : np.ndarray
            Centerline velocity over time [m/s].
        width : int
            ASCII width.
        height : int
            ASCII height.

        Returns
        -------
        output : str
            ASCII waveform.
        """
        lines = []
        lines.append(f"+{'-' * (width + 2)}+")
        lines.append(f"|  SPECTRAL DOPPLER VELOCITY WAVEFORM (m/s vs Time) {' ' * max(0, width - 48)}|")
        lines.append(f"+{'-' * (width + 2)}+")

        v_min = float(min(0.0, np.min(velocities)))
        v_max = float(max(0.1, np.max(velocities) * 1.1))

        # Subsample along width
        idx_t = np.linspace(0, len(velocities) - 1, width).astype(int)
        v_samples = velocities[idx_t]

        for h in range(height - 1, -1, -1):
            level = v_min + (h / (height - 1)) * (v_max - v_min)
            row = []
            for w in range(width):
                val = v_samples[w]
                if abs(val - level) < (v_max - v_min) / (height * 1.5):
                    row.append("*")
                elif val > level and level >= 0:
                    row.append(":")
                elif val < level and level < 0:
                    row.append(":")
                elif abs(level) < (v_max - v_min) / (height * 2.0):
                    row.append("-")  # Zero baseline
                else:
                    row.append(" ")
            lines.append(f"| {level:5.2f} |{''.join(row)}|")

        lines.append(f"+{'-' * 8}+{'-' * width}+")
        lines.append(f"| Time   | 0.0s{' ' * (width - 12)}{time_points[-1]:.2f}s |")
        lines.append(f"+{'-' * (width + 10)}+")
        return "\n".join(lines)

    @staticmethod
    def format_hemodynamics_report(
        peak_systolic_velocity: float,
        end_diastolic_velocity: float,
        mean_velocity: float,
        clutter_rejection_ratio_db: float,
        stenosis_grade: str,
    ) -> str:
        """Format a clinical hemodynamics and stenosis evaluation report."""
        psv_cms = peak_systolic_velocity * 100.0
        edv_cms = end_diastolic_velocity * 100.0
        ri = (peak_systolic_velocity - end_diastolic_velocity) / (peak_systolic_velocity + 1e-12)

        lines = [
            "+---------------------------------------------------------+",
            "|      CAROTID DUPLEX HEMODYNAMIC DIAGNOSTIC REPORT       |",
            "+---------------------------------------------------------+",
            f"| Peak Systolic Velocity (PSV) : {psv_cms:8.1f} cm/s ({peak_systolic_velocity:.2f} m/s)      |",
            f"| End Diastolic Velocity (EDV) : {edv_cms:8.1f} cm/s ({end_diastolic_velocity:.2f} m/s)      |",
            f"| Mean Spatial Velocity (Vmean): {mean_velocity*100.0:8.1f} cm/s ({mean_velocity:.2f} m/s)      |",
            f"| Resistive Index (RI)         : {ri:8.2f}                          |",
            f"| Clutter Rejection Ratio (CRR): {clutter_rejection_ratio_db:8.1f} dB                        |",
            "+---------------------------------------------------------+",
            f"| Clinical Stenosis Diagnostic : {stenosis_grade:<24} |",
            "+---------------------------------------------------------+",
        ]
        return "\n".join(lines)
