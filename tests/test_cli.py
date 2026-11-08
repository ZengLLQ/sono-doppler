"""Unit tests for Click CLI commands."""

from click.testing import CliRunner
import pytest

from sono_doppler.cli import main


def test_cli_demo():
    runner = CliRunner()
    result = runner.invoke(main, ["demo", "--frames", "16", "--stenosis", "0.50"])
    assert result.exit_code == 0
    assert "COLOR DOPPLER VELOCITY MAP" in result.output
    assert "CAROTID DUPLEX HEMODYNAMIC DIAGNOSTIC REPORT" in result.output


def test_cli_stenosis():
    runner = CliRunner()
    result = runner.invoke(main, ["stenosis"])
    assert result.exit_code == 0
    assert "CLINICAL CAROTID ARTERY STENOSIS HEMODYNAMICS COMPARISON" in result.output
    assert "Normal Unobstructed Carotid" in result.output
    assert "Severe Carotid Stenosis (70%)" in result.output


def test_cli_benchmark():
    runner = CliRunner()
    result = runner.invoke(main, ["benchmark"])
    assert result.exit_code == 0
    assert "Casorati SVD" in result.output
    assert "Benchmark completed successfully." in result.output
