"""Command-line interface for SonoDoppler ultrasound Doppler simulator."""

from __future__ import annotations

import time
import click
import numpy as np

from sono_doppler.beamforming.delay_and_sum import DelayAndSumBeamformer, PlaneWaveConfig
from sono_doppler.doppler.kasai import KasaiDopplerEstimator
from sono_doppler.filtering.svd_clutter import CasoratiSVDFilter
from sono_doppler.filtering.wall_filter import IIRWallFilter, PolynomialWallFilter
from sono_doppler.phantoms.flow_phantom import CarotidFlowPhantom, StenosisProfile
from sono_doppler.transducer import TransducerConfig, UltrasoundPhysics
from sono_doppler.visualizer import AsciiDopplerVisualizer


@click.group()
def main() -> None:
    """SonoDoppler: Ultrasound Beamforming, SVD Clutter Filtering & Color Doppler CLI."""
    pass


@main.command()
@click.option("--frames", default=32, help="Number of slow-time ensemble frames.")
@click.option("--stenosis", default=0.65, help="Carotid stenosis diameter reduction fraction (0.0 to 0.85).")
def demo(frames: int, stenosis: float) -> None:
    """Run an end-to-end ultrafast Doppler flow simulation."""
    click.echo(f"Initializing Carotid Ultrasound Simulation with {stenosis*100:.0f}% Stenosis...")

    transducer = TransducerConfig(num_elements=64, center_frequency=5.0e6)
    prf = 4000.0  # 4 kHz PRF

    # Image coordinates: 40 mm lateral x 20 mm depth window
    x_grid = np.linspace(-0.015, 0.015, 48)
    z_grid = np.linspace(0.014, 0.030, 48)

    profile = StenosisProfile(diameter_reduction=stenosis)
    phantom = CarotidFlowPhantom(
        depth_m=0.022,
        vessel_radius_m=0.0035,
        tilt_angle_deg=20.0,
        mean_inflow_velocity=0.35,
        stenosis=profile,
        tissue_to_blood_ratio_db=35.0,
    )

    click.echo("Generating 3D complex ultrasound slow-time ensemble...")
    t_start = time.perf_counter()
    ensemble, vx_true, vz_true = phantom.generate_complex_ensemble(
        x_grid=x_grid,
        z_grid=z_grid,
        prf=prf,
        num_frames=frames,
        center_frequency=transducer.center_frequency,
        steer_angle_rad=0.0,
        seed=101,
    )
    gen_time = time.perf_counter() - t_start
    click.echo(f"Generated ensemble {ensemble.shape} in {gen_time*1000.0:.1f} ms.")

    click.echo("Applying Casorati Spatiotemporal SVD Clutter Filter...")
    t_svd = time.perf_counter()
    svd_filter = CasoratiSVDFilter(clutter_rank=4)
    blood_ensemble, svd_metrics = svd_filter.filter_ensemble(ensemble)
    svd_time = time.perf_counter() - t_svd
    click.echo(
        f"SVD completed in {svd_time*1000.0:.1f} ms | "
        f"CRR: {svd_metrics.clutter_rejection_ratio_db:.1f} dB | "
        f"Clutter Rank Cutoff: {svd_metrics.clutter_cutoff_rank}"
    )

    click.echo("Estimating Color Flow Doppler via Kasai 2D Autocorrelation...")
    kasai = KasaiDopplerEstimator(
        prf=prf,
        center_frequency=transducer.center_frequency,
        spatial_kernel=(3, 3),
    )
    doppler_res = kasai.estimate(blood_ensemble, power_threshold_db=18.0)

    # Diagnostic hemodynamics
    psv = float(np.max(np.abs(doppler_res.velocity)))
    edv = float(psv * 0.30)  # Simulated diastole
    vmean = float(np.mean(np.abs(doppler_res.velocity[doppler_res.velocity != 0]))) if np.any(doppler_res.velocity != 0) else 0.0

    if psv >= 2.30:
        grade = "Severe Stenosis (>= 70%)"
    elif psv >= 1.25:
        grade = "Moderate Stenosis (50-69%)"
    else:
        grade = "Normal / Mild (< 50%)"

    click.echo("\n" + AsciiDopplerVisualizer.render_color_doppler_map(doppler_res.velocity, v_max=kasai.nyquist_velocity))
    click.echo("\n" + AsciiDopplerVisualizer.format_hemodynamics_report(
        peak_systolic_velocity=psv,
        end_diastolic_velocity=edv,
        mean_velocity=vmean,
        clutter_rejection_ratio_db=svd_metrics.clutter_rejection_ratio_db,
        stenosis_grade=grade,
    ))


@main.command()
def stenosis() -> None:
    """Compare hemodynamics between healthy carotid and 70% severe stenosis."""
    click.echo("===============================================================")
    click.echo(" CLINICAL CAROTID ARTERY STENOSIS HEMODYNAMICS COMPARISON ")
    click.echo("===============================================================")

    prf = 5000.0
    f0 = 5.0e6
    x_grid = np.linspace(-0.015, 0.015, 36)
    z_grid = np.linspace(0.016, 0.028, 36)

    cases = [
        ("Normal Unobstructed Carotid", 0.0),
        ("Moderate Carotid Stenosis (50%)", 0.50),
        ("Severe Carotid Stenosis (70%)", 0.70),
    ]

    for title, sten_val in cases:
        profile = StenosisProfile(diameter_reduction=sten_val)
        phantom = CarotidFlowPhantom(
            depth_m=0.022,
            vessel_radius_m=0.0035,
            tilt_angle_deg=20.0,
            mean_inflow_velocity=0.35,
            stenosis=profile,
        )
        ensemble, _, _ = phantom.generate_complex_ensemble(
            x_grid, z_grid, prf=prf, num_frames=32, center_frequency=f0, seed=42
        )
        svd_filter = CasoratiSVDFilter(clutter_rank=4)
        blood, metrics = svd_filter.filter_ensemble(ensemble)

        kasai = KasaiDopplerEstimator(prf=prf, center_frequency=f0)
        res = kasai.estimate(blood, power_threshold_db=15.0)

        psv = float(np.max(np.abs(res.velocity)))
        edv = float(psv * 0.28)
        ri = (psv - edv) / (psv + 1e-12)

        click.echo(f"\nScenario: {title}")
        click.echo(f"  Diameter Reduction   : {sten_val*100.0:.0f}%")
        click.echo(f"  Peak Systolic (PSV)  : {psv*100.0:6.1f} cm/s ({psv:.2f} m/s)")
        click.echo(f"  End Diastolic (EDV)  : {edv*100.0:6.1f} cm/s ({edv:.2f} m/s)")
        click.echo(f"  Resistive Index (RI) : {ri:6.2f}")
        click.echo(f"  Clutter Rejection    : {metrics.clutter_rejection_ratio_db:6.1f} dB")


@main.command()
def benchmark() -> None:
    """Benchmark SVD clutter filter against traditional IIR and Polynomial wall filters."""
    click.echo("Benchmarking Clutter Rejection Filters (Casorati SVD vs IIR vs Polynomial)...")
    prf = 4000.0
    f0 = 5.0e6
    x_grid = np.linspace(-0.015, 0.015, 36)
    z_grid = np.linspace(0.016, 0.028, 36)

    phantom = CarotidFlowPhantom(
        depth_m=0.022,
        vessel_radius_m=0.0035,
        tilt_angle_deg=20.0,
        mean_inflow_velocity=0.35,
        stenosis=StenosisProfile(0.60),
        tissue_to_blood_ratio_db=35.0,
    )
    ensemble, _, _ = phantom.generate_complex_ensemble(
        x_grid, z_grid, prf=prf, num_frames=48, center_frequency=f0, seed=42
    )

    # 1. Casorati SVD
    t0 = time.perf_counter()
    svd = CasoratiSVDFilter(clutter_rank=5)
    b_svd, m_svd = svd.filter_ensemble(ensemble)
    t_svd = (time.perf_counter() - t0) * 1000.0

    # 2. IIR Wall Filter
    t0 = time.perf_counter()
    iir = IIRWallFilter(cutoff_normalized=0.08, order=4)
    b_iir = iir.filter_ensemble(ensemble)
    t_iir = (time.perf_counter() - t0) * 1000.0

    # 3. Polynomial Filter
    t0 = time.perf_counter()
    poly = PolynomialWallFilter(poly_order=2)
    b_poly = poly.filter_ensemble(ensemble)
    t_poly = (time.perf_counter() - t0) * 1000.0

    click.echo("\n+---------------------+-------------------+----------------+")
    click.echo("| Filter Algorithm    | Execution Time    | CRR / Energy   |")
    click.echo("+---------------------+-------------------+----------------+")
    click.echo(f"| Casorati SVD        | {t_svd:13.2f} ms   | {m_svd.clutter_rejection_ratio_db:10.1f} dB   |")
    click.echo(f"| IIR Butterworth     | {t_iir:13.2f} ms   |       N/A      |")
    click.echo(f"| Polynomial (deg=2)  | {t_poly:13.2f} ms   |       N/A      |")
    click.echo("+---------------------+-------------------+----------------+")
    click.echo("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
