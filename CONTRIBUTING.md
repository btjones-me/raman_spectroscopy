# Contributing

Install the development dependencies with `python -m pip install -e ".[dev]"`, then run `pytest -q`, `ruff check src tests`, and `ruff format --check src tests`. Run the CLI on the bundled data as described in the README before proposing a release.

Keep numerical analysis independent of file access and plotting. Preserve input arrays, report invalid data explicitly, and include a regression test when fixing a numerical or workflow bug.

Propose scientific changes (baseline correction, peak selection, constraints, weighting or normalisation) separately from mechanical refactors. Explain the assumptions and compare before/after results on synthetic spectra and representative measurements. Do not regenerate reference fixtures merely to make changed behaviour pass.

The historical dissertation script and sample data are reference material. Avoid modifying them as part of general cleanup. Do not include private spectra or identifying laboratory information in issues without permission.

To check the notebook locally, install the development dependencies and execute it with Jupyter or nbclient from the repository root. It uses the local checkout when available. The Colab install path uses GitHub's default branch and becomes usable when the package/notebook changes are merged there.
