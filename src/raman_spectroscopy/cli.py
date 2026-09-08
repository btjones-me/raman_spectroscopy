"""Command-line batch analysis with explicit failure reporting."""

import argparse
import json
import sys
from pathlib import Path

from .analysis import AnalysisSettings, analyse_spectrum
from .io import load_spectrum, save_result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit Lorentzian peaks to two-column Raman spectra")
    parser.add_argument(
        "input", type=Path, help="Spectrum file or directory (searched recursively)"
    )
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "--pattern", default="*.txt", help="Directory file pattern (default: *.txt)"
    )
    parser.add_argument("--peaks", type=int, default=5)
    parser.add_argument("--baseline", choices=["legacy", "polynomial", "none"], default="legacy")
    parser.add_argument("--baseline-degree", type=int, default=2)
    parser.add_argument("--normalisation", type=float, default=9.5)
    parser.add_argument("--initial-width", type=float, default=10)
    parser.add_argument("--max-evaluations", type=int, default=14000)
    parser.add_argument(
        "--delimiter", help="Column separator; inferred for CSV, whitespace otherwise"
    )
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true", help="Write data and metadata only")
    args = parser.parse_args(argv)
    try:
        settings = AnalysisSettings(
            args.peaks,
            args.baseline,
            args.baseline_degree,
            args.normalisation,
            args.initial_width,
            args.max_evaluations,
        )
        if args.skiprows < 0:
            raise ValueError("skiprows must be non-negative")
        if args.input.is_file():
            files = [args.input]
        elif args.input.is_dir():
            files = sorted(path for path in args.input.rglob(args.pattern) if path.is_file())
        else:
            raise ValueError(f"Input does not exist: {args.input}")
        if not files:
            raise ValueError(f"No files match {args.pattern} in {args.input}")
        args.output.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        from .plotting import plot_result
    entries = []
    for path in files:
        relative = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        destination = args.output / relative.parent / (relative.name + ".analysis")
        try:
            result = analyse_spectrum(*load_spectrum(path, args.delimiter, args.skiprows), settings)
            save_result(
                result,
                destination,
                source=path,
                input_options={"delimiter": args.delimiter, "skiprows": args.skiprows},
            )
            if not args.no_plots:
                figure = plot_result(result, title=str(relative))
                try:
                    figure.savefig(destination / "fit.png", dpi=150)
                finally:
                    plt.close(figure)
            entries.append(
                {
                    "source": str(path),
                    "status": "ok",
                    "output": str(destination),
                    "rmse_au": result.rmse,
                    "warnings": result.warnings,
                }
            )
            print(f"OK {relative}: {len(result.parameters)} peaks; RMSE {result.rmse:.4g}")
        except (ValueError, RuntimeError, OSError, FloatingPointError) as error:
            entries.append({"source": str(path), "status": "failed", "error": str(error)})
            print(f"FAILED {relative}: {error}", file=sys.stderr)
    (args.output / "summary.json").write_text(json.dumps(entries, indent=2) + "\n")
    failed = sum(entry["status"] == "failed" for entry in entries)
    print(f"{len(entries) - failed}/{len(entries)} succeeded. Results: {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
