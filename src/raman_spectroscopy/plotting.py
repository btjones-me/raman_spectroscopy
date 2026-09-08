"""Optional plotting, separate from the numerical pipeline."""

from .analysis import lorentzian


def plot_result(result, title="Raman spectrum"):
    """Return a figure with baseline, fit components and residuals; never show it."""
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, layout="constrained")
    axes[0].plot(result.x, result.raw, label="Measured")
    axes[0].plot(result.x, result.baseline, label="Effective baseline")
    axes[0].set_ylabel("Input intensity")
    axes[0].set_title(title)
    axes[0].legend()
    axes[1].plot(result.x, result.processed, color="black", label="Processed")
    axes[1].plot(result.x, result.fitted, "r--", label="Fit")
    for number, parameters in enumerate(result.parameters, start=1):
        axes[1].plot(result.x, lorentzian(result.x, *parameters), alpha=0.7, label=f"Peak {number}")
    axes[1].set_ylabel("Intensity (a.u.)")
    axes[1].legend(ncol=2)
    axes[2].plot(result.x, result.residuals)
    axes[2].axhline(0, color="grey", linewidth=0.7)
    axes[2].set_ylabel("Residual (a.u.)")
    axes[2].set_xlabel("Raman shift (cm⁻¹)")
    return figure
