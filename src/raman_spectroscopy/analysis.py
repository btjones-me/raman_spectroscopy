"""Numerical analysis without file access or plotting.

Defaults retain the dissertation algorithm. Widths are HWHM, not FWHM;
normalised amplitudes are arbitrary units, not raw detector counts.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import peakutils
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks_cwt


@dataclass(frozen=True)
class AnalysisSettings:
    n_peaks: int = 5
    baseline: str = "legacy"
    baseline_degree: int = 2
    normalisation: float | None = 9.5
    initial_width: float = 10.0
    max_evaluations: int = 14000

    def __post_init__(self):
        for name in ("n_peaks", "max_evaluations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.baseline not in {"legacy", "polynomial", "none"}:
            raise ValueError("baseline must be legacy, polynomial or none")
        if not isinstance(self.baseline_degree, int) or self.baseline_degree < 0:
            raise ValueError("baseline_degree must be a non-negative integer")
        if self.baseline == "legacy" and self.baseline_degree != 2:
            raise ValueError("legacy baseline uses degree 2; select polynomial to change it")
        for name in ("normalisation", "initial_width"):
            value = getattr(self, name)
            if name == "normalisation" and value is None:
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class AnalysisResult:
    x: np.ndarray
    raw: np.ndarray
    baseline: np.ndarray
    processed: np.ndarray
    fitted: np.ndarray
    parameters: np.ndarray  # rows: centre, amplitude, positive HWHM
    covariance: np.ndarray
    settings: AnalysisSettings
    warnings: list[str] = field(default_factory=list)

    @property
    def residuals(self):
        return self.processed - self.fitted

    @property
    def rmse(self):
        return float(np.sqrt(np.mean(self.residuals**2)))

    @property
    def standard_errors(self):
        with np.errstate(invalid="ignore"):
            return np.sqrt(np.diag(self.covariance)).reshape(-1, 3)


def validate_spectrum(x, y):
    """Copy finite, strictly monotonic two-column data; preserve axis direction."""
    x, y = np.array(x, dtype=float, copy=True), np.array(y, dtype=float, copy=True)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 4:
        raise ValueError("A spectrum needs matching one-dimensional arrays with at least 4 points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spectrum contains non-finite values")
    steps = np.diff(x)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError(
            "Raman shifts must be strictly increasing or decreasing, without duplicates"
        )
    return x, y


def correct_baseline(x, y, settings):
    """Return corrected intensity and effective baseline in original intensity units.

    Legacy mode adds 0.001*x**2 - 0.08*x + 5 before the degree-2
    PeakUtils baseline, exactly as in the dissertation script. Polynomial
    mode applies PeakUtils directly and is an explicit methodological change.
    """
    if settings.baseline == "none":
        return y.copy(), np.zeros_like(y)
    if settings.baseline_degree >= len(y):
        raise ValueError("Baseline degree must be smaller than the number of points")
    working = y + np.polyval([0.001, -0.08, 5], x) if settings.baseline == "legacy" else y
    corrected = working - peakutils.baseline(working, settings.baseline_degree)
    return corrected, y - corrected


def lorentzian(x, centre, amplitude, width):
    """A Lorentzian with peak height amplitude and half width at half maximum."""
    return amplitude * width**2 / ((x - centre) ** 2 + width**2)


def sum_lorentzians(x, *parameters):
    total = np.zeros_like(x, dtype=float)
    for centre, amplitude, width in np.asarray(parameters).reshape(-1, 3):
        total += lorentzian(x, centre, amplitude, width)
    return total


def detect_peaks(x, y, n_peaks):
    """Return up to n strongest CWT peaks; wavelet widths are in sample points."""
    indices = find_peaks_cwt(y, np.arange(1, 50, 0.5))
    return np.array(sorted(indices, key=lambda index: y[index])[-n_peaks:], dtype=int)


def fit_peaks(x, y, initial_parameters, max_evaluations=14000):
    """Fit the legacy unconstrained sum and retain covariance and warning messages.

    Canonicalise width signs after fitting (the curve is invariant to sign),
    transforming covariance consistently. This does not introduce new bounds.
    """
    if len(x) <= np.size(initial_parameters):
        raise ValueError("Not enough data points for the requested number of fit parameters")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", OptimizeWarning)
        parameters, covariance = curve_fit(
            sum_lorentzians,
            x,
            y,
            p0=np.ravel(initial_parameters),
            maxfev=max_evaluations,
            sigma=np.full(len(y), 2.0),
        )
    messages = [str(item.message) for item in caught]
    if not np.isfinite(parameters).all():
        raise ValueError("Fit produced non-finite parameters")
    signs = np.ones_like(parameters)
    signs[2::3] = np.where(parameters[2::3] < 0, -1.0, 1.0)
    parameters *= signs
    covariance = covariance * signs[:, None] * signs[None, :]
    # Stable peak ordering for tables and comparison, including covariance axes.
    order = np.argsort(parameters[::3])
    indices = (order[:, None] * 3 + np.arange(3)).ravel()
    parameters, covariance = parameters[indices], covariance[np.ix_(indices, indices)]
    if not np.isfinite(covariance).all():
        messages.append("Covariance is not finite; parameter uncertainty is unreliable")
    if np.any(parameters[1::3] < 0):
        messages.append("Fit includes negative amplitudes; inspect before interpretation")
    if np.any((parameters[::3] < min(x)) | (parameters[::3] > max(x))):
        messages.append("Fit includes centres outside the measured range")
    if np.any(parameters[2::3] == 0):
        messages.append("Fit includes a zero width; inspect before interpretation")
    return parameters.reshape(-1, 3), covariance, messages


def analyse_spectrum(x, y, settings=None):
    """Baseline-correct, normalise, detect and fit peaks without modifying input."""
    settings = settings or AnalysisSettings()
    x, raw = validate_spectrum(x, y)
    if np.ptp(raw) == 0:
        raise ValueError("Constant-intensity spectrum has no peaks to analyse")
    corrected, baseline = correct_baseline(x, raw, settings)
    maximum = np.max(corrected)
    if not np.isfinite(corrected).all() or maximum <= 0:
        raise ValueError("Baseline-corrected spectrum has no finite positive maximum")
    processed = (
        corrected.copy()
        if settings.normalisation is None
        else settings.normalisation * corrected / maximum
    )
    peaks = detect_peaks(x, processed, settings.n_peaks)
    if not len(peaks):
        raise ValueError("No peaks detected; inspect data and baseline settings")
    initial = np.column_stack(
        (x[peaks], processed[peaks], np.full(len(peaks), settings.initial_width))
    )
    parameters, covariance, messages = fit_peaks(x, processed, initial, settings.max_evaluations)
    if len(peaks) < settings.n_peaks:
        messages.append(f"Requested {settings.n_peaks} peaks but detected only {len(peaks)}")
    steps = np.abs(np.diff(x))
    if not np.allclose(steps, np.median(steps), rtol=0.05):
        messages.append("Non-uniform Raman shift spacing: CWT widths are in sample points")
    fitted = sum_lorentzians(x, *parameters.ravel())
    if not np.isfinite(fitted).all():
        raise ValueError("Fit produced a non-finite curve")
    return AnalysisResult(
        x, raw, baseline, processed, fitted, parameters, covariance, settings, messages
    )


def average_spectra(spectra, settings=None):
    """Average individually corrected/normalised repeats on identical x grids.

    Returns x and mean intensity. Fit with baseline='none', normalisation=None
    to avoid processing the mean twice.
    No interpolation or silent averaging of mismatched axes is performed.
    """
    settings = settings or AnalysisSettings()
    spectra = [validate_spectrum(x, y) for x, y in spectra]
    if not spectra:
        raise ValueError("No spectra to average")
    x = spectra[0][0]
    processed = []
    for axis, raw in spectra:
        if not np.array_equal(axis, x):
            raise ValueError("Repeated spectra must have identical Raman shift grids")
        corrected, _ = correct_baseline(axis, raw, settings)
        if np.ptp(raw) == 0 or not np.isfinite(corrected).all() or corrected.max() <= 0:
            raise ValueError("A repeated spectrum cannot be normalised")
        processed.append(
            corrected
            if settings.normalisation is None
            else settings.normalisation * corrected / corrected.max()
        )
    return x.copy(), np.mean(processed, axis=0)
