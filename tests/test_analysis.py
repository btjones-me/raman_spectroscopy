import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from raman_spectroscopy import (
    AnalysisSettings,
    analyse_spectrum,
    average_spectra,
    load_spectrum,
)
from raman_spectroscopy.analysis import fit_peaks, sum_lorentzians, validate_spectrum

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Raman Spectroscopy/CZTS_data/CZTS_111116/B21/B21_1.txt"


def test_matches_original_dissertation_reference():
    metadata = json.loads((ROOT / "tests/fixtures/b21_1_legacy.json").read_text())
    assert sha256(SAMPLE.read_bytes()).hexdigest() == metadata["sample_sha256"]
    legacy = ROOT / "Raman Spectroscopy/PythonCode/raman_analysis_clean.py"
    assert sha256(legacy.read_bytes()).hexdigest() == metadata["script_sha256"]
    reference = np.load(ROOT / "tests/fixtures/b21_1_legacy.npz")
    x, raw = load_spectrum(SAMPLE)
    before = raw.copy()
    result = analyse_spectrum(x, raw)
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_allclose(result.processed, reference["processed"], atol=1e-9)
    np.testing.assert_allclose(result.fitted, reference["fitted"], atol=2e-5, rtol=2e-4)
    expected = reference["parameters"].reshape(-1, 3)
    expected[:, 2] = abs(expected[:, 2])
    expected = expected[np.argsort(expected[:, 0])]
    np.testing.assert_allclose(result.parameters, expected, atol=0.02, rtol=0.002)
    assert np.all(result.parameters[:, 2] > 0)
    assert result.covariance.shape == (15, 15)


def test_known_overlapping_lorentzians_recovered():
    x = np.linspace(200, 450, 1000)
    expected = np.array([[300, 5, 7], [324, 3, 6]])
    y = sum_lorentzians(x, *expected.ravel())
    initial = [[298, 4, -9], [326, 2, 8]]
    params, covariance, messages = fit_peaks(x, y, initial)
    np.testing.assert_allclose(params, expected, atol=1e-5)
    np.testing.assert_allclose(sum_lorentzians(x, *params.ravel()), y, atol=1e-6)
    assert np.isfinite(covariance).all()
    assert not messages


def test_full_pipeline_recovers_synthetic_peaks():
    x = np.linspace(200, 450, 1024)
    y = sum_lorentzians(x, 280, 5, 4, 360, 3, 7)
    result = analyse_spectrum(x, y, AnalysisSettings(n_peaks=2, baseline="none"))
    np.testing.assert_allclose(result.parameters[:, 0], [280, 360], atol=0.01)
    np.testing.assert_allclose(result.parameters[:, 2], [4, 7], atol=0.01)
    assert result.rmse < 1e-5


@pytest.mark.parametrize(
    "x,y",
    [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3, 4], [1, 2, 3]),
        ([1, 2, 2, 4], [1, 2, 3, 4]),
        ([1, 3, 2, 4], [1, 2, 3, 4]),
        ([1, 2, 3, 4], [1, 2, np.nan, 4]),
        ([1, 2, 3, np.inf], [1, 2, 3, 4]),
    ],
)
def test_invalid_spectra_rejected(x, y):
    with pytest.raises(ValueError):
        validate_spectrum(x, y)


def test_flat_spectrum_rejected():
    with pytest.raises(ValueError, match="Constant"):
        analyse_spectrum(np.arange(100), np.ones(100))


@pytest.mark.parametrize(
    "settings",
    [
        {"n_peaks": 0},
        {"baseline": "unknown"},
        {"baseline_degree": 3},
        {"initial_width": 0},
        {"normalisation": float("nan")},
        {"max_evaluations": -1},
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        AnalysisSettings(**settings)


def test_averaging_rejects_missing_and_mismatched_grids():
    x, y = load_spectrum(SAMPLE)
    with pytest.raises(ValueError, match="No spectra"):
        average_spectra([])
    with pytest.raises(ValueError, match="identical"):
        average_spectra([(x, y), (x + 1, y)])
    axis, mean = average_spectra([(x, y), (x, y)])
    np.testing.assert_array_equal(axis, x)
    np.testing.assert_allclose(mean, analyse_spectrum(x, y).processed)


def test_average_is_not_normalised_twice():
    x = np.linspace(200, 450, 1024)
    first = sum_lorentzians(x, 280, 5, 4, 360, 3, 7)
    second = sum_lorentzians(x, 280, 3, 4, 360, 5, 7)
    axis, mean = average_spectra([(x, first), (x, second)], AnalysisSettings(baseline="none"))
    result = analyse_spectrum(
        axis, mean, AnalysisSettings(n_peaks=2, baseline="none", normalisation=None)
    )
    np.testing.assert_array_equal(result.processed, mean)
    assert mean.max() < 9.5
    assert result.rmse < 1e-5


def test_width_sign_covariance_transformation():
    # The same fit started on either side of the width symmetry must produce
    # the same positive widths and uncertainty correlations after conversion.
    x = np.linspace(200, 450, 1024)
    rng = np.random.default_rng(7)
    y = sum_lorentzians(x, 300, 5, 7) + rng.normal(0, 0.02, len(x))
    positive, cov_positive, _ = fit_peaks(x, y, [298, 4, 9])
    negative, cov_negative, _ = fit_peaks(x, y, [298, 4, -9])
    np.testing.assert_allclose(positive, negative, rtol=1e-6)
    np.testing.assert_allclose(cov_positive, cov_negative, rtol=1e-3, atol=1e-10)
