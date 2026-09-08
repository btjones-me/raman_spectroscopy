"""Explicit data loading and reproducible, machine-readable result exports."""

import csv
import json
import platform
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .analysis import validate_spectrum


def load_spectrum(path, delimiter=None, skiprows=0):
    """Read exactly two numeric columns: Raman shift (cm^-1), intensity.

    CSV defaults to comma-separated, other extensions to whitespace-separated.
    Header lines must be explicitly skipped. '#' comments are accepted.
    """
    path = Path(path)
    delimiter = delimiter or ("," if path.suffix.lower() == ".csv" else None)
    try:
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if data.shape[1] != 2:
        raise ValueError(f"{path}: expected exactly two columns, got {data.shape[1]}")
    return validate_spectrum(data[:, 0], data[:, 1])


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_result(result, directory, source=None, input_options=None):
    """Write into a new directory; never overwrite an earlier analysis.

    JSON records settings, covariance, versions and the input SHA-256. Non-finite
    uncertainty values are null in JSON (not an estimate of zero uncertainty).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "peaks.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "centre_cm-1",
                "amplitude_au",
                "hwhm_cm-1",
                "fwhm_cm-1",
                "centre_stderr_cm-1",
                "amplitude_stderr_au",
                "hwhm_stderr_cm-1",
            ]
        )
        for (centre, amplitude, width), errors in zip(result.parameters, result.standard_errors):
            writer.writerow([centre, amplitude, width, 2 * width, *errors])
    np.savetxt(
        directory / "spectrum.csv",
        np.column_stack(
            (
                result.x,
                result.raw,
                result.baseline,
                result.processed,
                result.fitted,
                result.residuals,
            )
        ),
        delimiter=",",
        comments="",
        header="raman_shift_cm-1,raw_intensity,baseline,processed_au,fit_au,residual_au",
    )
    metadata = {
        "source": str(source) if source else None,
        "source_sha256": sha256(Path(source).read_bytes()).hexdigest() if source else None,
        "input_options": input_options or {},
        "settings": asdict(result.settings),
        "rmse_au": result.rmse,
        "warnings": result.warnings,
        "covariance": result.covariance.tolist(),
        "covariance_order": "centre, amplitude, HWHM for each peak in peaks.csv order",
        "uncertainty_note": (
            "Local linear covariance, relative sigma=2; not calibrated experimental uncertainty"
        ),
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "czts-raman-analysis",
                "numpy",
                "scipy",
                "PeakUtils",
                "matplotlib",
            )
        },
    }
    (directory / "analysis.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n"
    )
    return directory
