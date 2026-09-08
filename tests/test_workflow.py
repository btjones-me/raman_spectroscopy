import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from raman_spectroscopy import analyse_spectrum, load_spectrum, save_result
from raman_spectroscopy.cli import main

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "Raman Spectroscopy/CZTS_data/CZTS_111116/B21/B21_1.txt"


def test_csv_headers_and_malformed_input(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("shift,intensity\n1,2\n2,4\n3,3\n4,1\n")
    x, y = load_spectrum(path, skiprows=1)
    np.testing.assert_array_equal(x, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="Cannot read"):
        load_spectrum(path)
    path.write_text("1 2 3\n2 3 4\n3 4 5\n4 5 6\n")
    with pytest.raises(ValueError, match="two columns"):
        load_spectrum(path, delimiter=" ")


def test_exports_roundtrip_and_refuse_overwrite(tmp_path):
    result = analyse_spectrum(*load_spectrum(SAMPLE))
    output = tmp_path / "result"
    save_result(result, output, source=SAMPLE)
    metadata = json.loads((output / "analysis.json").read_text())
    assert metadata["settings"]["baseline"] == "legacy"
    assert len(metadata["source_sha256"]) == 64
    assert "scipy" in metadata["versions"]
    table = np.loadtxt(output / "peaks.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(table[:, :3], result.parameters)
    np.testing.assert_allclose(table[:, 3], 2 * result.parameters[:, 2])
    spectrum = np.loadtxt(output / "spectrum.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(spectrum[:, -1], result.residuals)
    with pytest.raises(FileExistsError):
        save_result(result, output)


def test_cli_from_unrelated_directory_with_plot(tmp_path):
    output = tmp_path / "result"
    process = subprocess.run(
        [sys.executable, "-m", "raman_spectroscopy.cli", str(SAMPLE), "--output", str(output)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary[0]["status"] == "ok"
    png = Path(summary[0]["output"]) / "fit.png"
    assert png.read_bytes().startswith(b"\x89PNG")


def test_batch_reports_failure_and_keeps_other_results(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "valid.txt").write_bytes(SAMPLE.read_bytes())
    (inputs / "bad.txt").write_text("bad data")
    output = tmp_path / "output"
    assert main([str(inputs), "--output", str(output), "--no-plots"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert sorted(item["status"] for item in summary) == ["failed", "ok"]
    with pytest.raises(SystemExit) as error:
        main([str(inputs), "--output", str(output)])
    assert error.value.code == 2


def test_empty_folder_fails_without_creating_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path), "--output", str(output)])
    assert error.value.code == 2
    assert not output.exists()
