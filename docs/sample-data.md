# Bundled measurements

These files accompanied Benjamin Jones's Durham University Master's dissertation on CZTS thin-film solar materials, as described in the original README. The repository provides 43 measurements grouped into B21–B27, with 1,024 numeric pairs per file. Each pair is Raman shift and intensity; the original plots label the shift in cm⁻¹. The bundled axes descend from approximately 834.788 to 212.178 cm⁻¹. Repeats within each sample share the same axis.

The data are useful for trying the workflow and checking software regressions. They are not a labelled benchmark: the repository does not provide ground-truth peak assignments, calibrated error estimates, or sufficient acquisition metadata to establish instrument settings, laser wavelength for each file, calibration or sample preparation. A date-like folder name is not treated as verified acquisition metadata. The maintainer should supply this information from original lab records before expanding scientific claims.

The existing repository-level MIT licence is retained unchanged. No additional permission claims about third-party or institutional material are introduced by this cleanup.

`tests/fixtures/b21_1_legacy.npz` captures processed intensity, fitted curve and original signed parameters for B21_1. Its adjacent JSON records the source commit, file hashes, dependency versions and generation method. It was generated from original function definitions, not the refactored package. Tests canonicalise parameter signs/order only for comparison, and allow small numerical optimiser differences across supported environments.
