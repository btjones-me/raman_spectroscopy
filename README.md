# Raman spectrum analysis

Baseline correction, automatic peak detection and fitting of a **sum of Lorentzian peaks** to Raman spectra. Analyse a single spectrum or a folder of measurements and export peak tables, diagnostic plots and reproducible settings.

Originally developed by Benjamin Jones for the 2016 Durham University Physics Master's dissertation, *Developing Sustainable Materials (CZTS) for Thin-Film Solar Devices Using Raman and Photoluminescence Spectroscopy*, supervised by Dr Douglas Halliday.

**Scope:** this is research software for spectral peak fitting. It does not automatically identify compounds or establish phase composition. Peak assignments require reference spectra, acquisition context and researcher interpretation.

[![Checks](https://github.com/btjones-me/raman_spectroscopy/actions/workflows/tests.yml/badge.svg)](https://github.com/btjones-me/raman_spectroscopy/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/btjones-me/raman_spectroscopy/blob/master/notebooks/quickstart.ipynb)

## Quick start

Requires Python 3.11 or newer. CI checks Python 3.11–3.13 on Windows, macOS and Linux. Run these commands from a terminal:

```sh
git clone https://github.com/btjones-me/raman_spectroscopy.git
cd raman_spectroscopy
python -m venv .venv
```

Activate the environment:

```sh
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and analyse one bundled measurement:

```sh
python -m pip install -e .
raman-analyse "Raman Spectroscopy/CZTS_data/CZTS_111116/B21/B21_1.txt" --output results/example
```

The terminal reports success or a specific error. Open `results/example/B21_1.txt.analysis/fit.png` to inspect the measured spectrum, estimated baseline, fit components and residuals. Results also include:

| File | Contents |
| --- | --- |
| `peaks.csv` | Centres, heights, HWHM, FWHM and approximate standard errors |
| `spectrum.csv` | Input, effective baseline, processed spectrum, fit and residuals |
| `analysis.json` | Settings, covariance, warnings, software versions and input checksum |
| `summary.json` | Batch success/failure records, at the output root |

Output directories must be **new**. Use another output name for a rerun; previous results are never overwritten.

For a browser-based walkthrough, open the [notebook](notebooks/quickstart.ipynb) or use the Colab button above. It demonstrates the bundled example, analysis settings, plots, file upload and downloads. Colab runs your uploaded data on Google's infrastructure; a local notebook is also supported.

## Your own measurements

Provide exactly two columns: **Raman shift in cm⁻¹**, then **intensity**. Whitespace-separated `.txt` and comma-separated `.csv` files are supported. Values must be finite, with a strictly increasing or decreasing shift axis. Duplicate shifts, unsorted data and constant-intensity spectra are rejected. Data are not silently sorted or resampled.

```text
250.0  102.3
250.5  105.1
251.0  110.2
251.5  106.8
```

This illustrates the format, not a sufficient dataset for a five-peak fit. Use full measured spectra. For a CSV with one header row:

```sh
raman-analyse my-spectrum.csv --skiprows 1 --peaks 3 --output results/my-spectrum
```

For every `.txt` file in a directory and its subdirectories:

```sh
raman-analyse "Raman Spectroscopy/CZTS_data" --output results/all-samples --max-evaluations 100000
```

The original evaluation limit is 14,000. The full-dataset example and CI explicitly use 100,000: B24_6 can exhaust the original limit on some platforms. Convergence can vary with numerical libraries and hardware; increasing the limit may help, but does not make a fit physically valid. The chosen limit is recorded in each result.

Use `--pattern "*.csv"` for a CSV batch, `--delimiter ";"` for semicolon-separated data, and `--no-plots` for tables only. Batch output preserves relative folders and full filenames to avoid name collisions. A failed spectrum is recorded and other files continue; the command exits nonzero if any file fails. See `raman-analyse --help` for settings.

## Analysis choices and limitations

The defaults retain the original individual-spectrum workflow:

1. Add the historical quadratic `0.001*x² - 0.08*x + 5`, then subtract a degree-2 PeakUtils baseline. The effective baseline export accounts for that added quadratic.
2. Scale the corrected maximum to 9.5 arbitrary units.
3. Detect peaks with SciPy CWT, using widths from 1 to 49.5 **sample points**, and retain up to the five strongest.
4. Fit an unconstrained sum of Lorentzians, starting each HWHM at 10 cm⁻¹.

`--baseline polynomial` applies PeakUtils directly to the measured intensities; `--baseline-degree` controls its degree. `--baseline none` skips baseline correction. **These are explicit methodological alternatives, not validated improvements.** The original baseline's misleading `degree` argument has not been carried over: legacy mode always uses degree 2.

Fitted widths are reported as positive HWHM; FWHM is twice HWHM. Taking the absolute width after optimisation preserves the original curve exactly because the model squares width. Covariance is transformed with the signs and peak ordering. Amplitudes may still be negative, centres may leave the measured range, and overlapping peaks may not be uniquely identifiable. Such results require inspection; successful optimisation does not establish scientific validity.

Standard errors come from the local fit covariance with the original constant relative weighting (`sigma=2`, SciPy's default `absolute_sigma=False`). They are **not calibrated experimental uncertainties** and exclude preprocessing, model-selection and baseline uncertainty. Non-finite covariance is flagged and encoded as `null` in JSON. Normalised intensity is labelled a.u., not detector counts. CWT uses sample spacing, so results may depend on acquisition resolution and axis direction; non-uniform spacing triggers a warning.

No compound library, smoothing, cosmic-ray removal, instrument response correction, automatic model selection or constrained optimisation is performed. Inspect residuals and compare with domain references before interpreting fitted peaks.

## Python use

```python
from raman_spectroscopy import AnalysisSettings, analyse_spectrum, load_spectrum, save_result
from raman_spectroscopy.plotting import plot_result

x, y = load_spectrum("measurement.txt")
result = analyse_spectrum(x, y, AnalysisSettings(n_peaks=3))
print(result.parameters)  # rows: centre, amplitude, positive HWHM
print(result.warnings)
save_result(result, "results/python-example", source="measurement.txt")
figure = plot_result(result)
figure.savefig("results/python-example/fit.png")
```

`average_spectra` corrects and normalises each repeat before averaging, requiring identical shift grids. To fit that mean without correcting or scaling it twice:

```python
from raman_spectroscopy import average_spectra

x, mean = average_spectra([load_spectrum("repeat1.txt"), load_spectrum("repeat2.txt")])
result = analyse_spectrum(x, mean, AnalysisSettings(n_peaks=4, baseline="none", normalisation=None))
```

For averaged results, retain the individual input paths, preprocessing settings and checksums separately: the single-spectrum exporter does not construct multi-input provenance automatically.

## Sample data and historical reference

The repository contains **43 two-column spectra**, each with 1,024 points, grouped into samples B21–B27 under `Raman Spectroscopy/CZTS_data/CZTS_111116`. See [sample provenance and limitations](docs/sample-data.md).

The original [dissertation script](Raman%20Spectroscopy/PythonCode/raman_analysis_clean.py) and historical plots remain unchanged for comparison. That script has Windows-specific paths and historical dependencies; use the package above for new work. The new package omits the unused `lmfit` dependency and unused experimental functions.

![Historical fitted spectrum](Raman%20Spectroscopy/ExamplePlots/figure_5.png)

## Validation and contributions

```sh
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests
ruff format --check src tests
```

Alternatively, `uv sync --locked --extra dev --python 3.12` reproduces the committed dependency resolution, then `uv run pytest -q`. CI also exercises dependency installation across operating systems and Python versions.

Tests compare the default pipeline against a captured result from the unchanged dissertation script, recover synthetic peaks of known position and width, and cover invalid input, averaging, exports and batch failures. This protects implementation behaviour; it does not validate chemical assignments or general performance on other instruments and materials.

Please [open an issue](https://github.com/btjones-me/raman_spectroscopy/issues) with your command, Python/package versions, error and a small shareable example. Suggestions and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose algorithm changes.

## Citation and licence

MIT licensed; see [LICENSE](LICENSE). Use GitHub's “Cite this repository” entry or [CITATION.cff](CITATION.cff), and record the commit and analysis settings used. No DOI has been assigned here. A versioned release archived through Zenodo is a future publishing step.
