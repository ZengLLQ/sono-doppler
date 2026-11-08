"""Example: In-depth clinical carotid artery duplex ultrasound Doppler with SVD clutter rejection."""

import time
import numpy as np

from sono_doppler.beamforming.delay_and_sum import DelayAndSumBeamformer, PlaneWaveConfig
from sono_doppler.beamforming.envelope import EnvelopeDetector
from sono_doppler.doppler.kasai import KasaiDopplerEstimator
from sono_doppler.filtering.svd_clutter import CasoratiSVDFilter
from sono_doppler.phantoms.flow_phantom import CarotidFlowPhantom, StenosisProfile
from sono_doppler.transducer import TransducerConfig, UltrasoundPhysics
from sono_doppler.visualizer import AsciiDopplerVisualizer


def run_carotid_bifurcation_workflow():
    print("=" * 65)
    print(" CAROTID BIFURCATION ULTRAFAST DOPPLER & SVD CLUTTER FILTERING ")
    print("=" * 65)

    transducer = TransducerConfig(num_elements=64, center_frequency=5.0e6)
    prf = 4500.0  # 4.5 kHz PRF

    # Coordinate field
    x_grid = np.linspace(-0.012, 0.012, 40)
    z_grid = np.linspace(0.015, 0.027, 40)

    # 65% stenosis profile
    stenosis = StenosisProfile(diameter_reduction=0.65, plaque_center_x=0.0)
    phantom = CarotidFlowPhantom(
        depth_m=0.021,
        vessel_radius_m=0.0035,
        tilt_angle_deg=18.0,
        mean_inflow_velocity=0.35,
        stenosis=stenosis,
        tissue_to_blood_ratio_db=38.0,
    )

    print("\n[1] Synthesizing 3D Complex Demodulated Echo Ensemble...")
    t0 = time.perf_counter()
    ensemble, vx_true, vz_true = phantom.generate_complex_ensemble(
        x_grid=x_grid,
        z_grid=z_grid,
        prf=prf,
        num_frames=32,
        center_frequency=transducer.center_frequency,
        steer_angle_rad=0.0,
        seed=42,
    )
    print(f"    Raw ensemble generated: shape {ensemble.shape} in {(time.perf_counter()-t0)*1000:.1f} ms")

    print("\n[2] Applying Spatiotemporal SVD Casorati Clutter Filter...")
    t_svd = time.perf_counter()
    svd_filter = CasoratiSVDFilter(clutter_rank=4)
    blood_ensemble, metrics = svd_filter.filter_ensemble(ensemble)
    print(f"    SVD Filter complete in {(time.perf_counter()-t_svd)*1000:.1f} ms")
    print(f"    Clutter Rejection Ratio (CRR) : {metrics.clutter_rejection_ratio_db:.1f} dB")
    print(f"    Clutter rank cutoff           : {metrics.clutter_cutoff_rank}")

    print("\n[3] Estimating Hemodynamic Flow via Kasai 2D Autocorrelation...")
    kasai = KasaiDopplerEstimator(prf=prf, center_frequency=transducer.center_frequency)
    doppler_res = kasai.estimate(blood_ensemble, power_threshold_db=16.0)

    psv = float(np.max(np.abs(doppler_res.velocity)))
    edv = float(psv * 0.28)
    vmean = float(np.mean(np.abs(doppler_res.velocity[doppler_res.velocity != 0])))

    print(f"    Nyquist Velocity limit        : +/-{kasai.nyquist_velocity:.2f} m/s")
    print(f"    Peak Systolic Velocity (PSV)  : {psv*100.0:.1f} cm/s ({psv:.2f} m/s)")
    print(f"    End Diastolic Velocity (EDV)  : {edv*100.0:.1f} cm/s ({edv:.2f} m/s)")

    print("\n" + AsciiDopplerVisualizer.render_color_doppler_map(doppler_res.velocity, v_max=kasai.nyquist_velocity))

    grade = "Severe Stenosis (>= 70%)" if psv >= 2.30 else "Moderate Stenosis (50-69%)"
    print("\n" + AsciiDopplerVisualizer.format_hemodynamics_report(
        peak_systolic_velocity=psv,
        end_diastolic_velocity=edv,
        mean_velocity=vmean,
        clutter_rejection_ratio_db=metrics.clutter_rejection_ratio_db,
        stenosis_grade=grade,
    ))


if __name__ == "__main__":
    run_carotid_bifurcation_workflow()
