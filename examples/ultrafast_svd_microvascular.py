"""Example: Ultrafast ultrasound microvascular flow imaging with SVD clutter rejection."""

import time
import numpy as np

from sono_doppler.doppler.kasai import KasaiDopplerEstimator
from sono_doppler.filtering.svd_clutter import CasoratiSVDFilter
from sono_doppler.filtering.wall_filter import IIRWallFilter
from sono_doppler.phantoms.flow_phantom import CarotidFlowPhantom, StenosisProfile
from sono_doppler.transducer import TransducerConfig


def run_microvascular_svd_comparison():
    print("=" * 65)
    print(" ULTRAFAST SVD CLUTTER FILTERING FOR MICROVASCULAR FLOW ")
    print("=" * 65)

    transducer = TransducerConfig(num_elements=64, center_frequency=7.5e6)  # High frequency 7.5 MHz
    prf = 3000.0
    frames = 48

    x_grid = np.linspace(-0.010, 0.010, 32)
    z_grid = np.linspace(0.015, 0.025, 32)

    phantom = CarotidFlowPhantom(
        depth_m=0.020,
        vessel_radius_m=0.0025,
        tilt_angle_deg=10.0,
        mean_inflow_velocity=0.20,
        tissue_to_blood_ratio_db=40.0,  # 40 dB tissue clutter dominance
    )

    print("\nSynthesizing high clutter ultrasound ensemble (Tissue/Blood contrast: 40 dB)...")
    ensemble, _, _ = phantom.generate_complex_ensemble(
        x_grid, z_grid, prf=prf, num_frames=frames, center_frequency=transducer.center_frequency, seed=77
    )

    print("\n[Method 1] Traditional High-Pass IIR Butterworth Wall Filter:")
    t0 = time.perf_counter()
    iir = IIRWallFilter(cutoff_normalized=0.10, order=4)
    b_iir = iir.filter_ensemble(ensemble)
    t_iir = (time.perf_counter() - t0) * 1000.0

    p_iir = np.mean(np.abs(b_iir) ** 2, axis=-1)
    print(f"  Execution time: {t_iir:.2f} ms")
    print(f"  Residual mean power: {float(np.mean(p_iir)):.2e}")

    print("\n[Method 2] Spatiotemporal Casorati SVD Clutter Filter:")
    t0 = time.perf_counter()
    svd = CasoratiSVDFilter(clutter_rank=5)
    b_svd, metrics = svd.filter_ensemble(ensemble)
    t_svd = (time.perf_counter() - t0) * 1000.0

    print(f"  Execution time: {t_svd:.2f} ms")
    print(f"  Clutter Rejection Ratio (CRR): {metrics.clutter_rejection_ratio_db:.1f} dB")
    print(f"  Tissue Clutter Energy : {metrics.clutter_energy:.2e}")
    print(f"  Extracted Blood Energy: {metrics.blood_energy:.2e}")

    # Estimate Doppler on SVD filtered blood
    kasai = KasaiDopplerEstimator(prf=prf, center_frequency=transducer.center_frequency)
    res = kasai.estimate(b_svd, power_threshold_db=15.0)

    print(f"\nSVD Micro-Doppler successfully extracted blood flow with peak velocity: {np.max(np.abs(res.velocity))*100:.1f} cm/s")


if __name__ == "__main__":
    run_microvascular_svd_comparison()
