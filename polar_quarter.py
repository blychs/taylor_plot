import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


r = np.array([1.0, -0.52, 0.25, 0.48, 0.33, 0.38, 0.43, 0.42])
std = np.array([3.90, 3.47, 6.95, 5.82, 3.91, 1.87, 3.82, 3.64]) / 3.90


def taylor_correls(
    r: npt.NDArray[np.float64],
    std: npt.NDArray[np.float64],
) -> None:
    """Plot a Taylor diagram.

    The correlation coefficients are converted to polar angles, while the
    normalized standard deviations are used as radii. The first element of
    each array represents the reference dataset at correlation and normalized
    standard deviation equal to one. Dotted contours show normalized centered
    RMSE values.

    The angular domain extends from 90 to 180 degrees only when ``r`` contains
    at least one negative correlation.

    Args:
        r: One-dimensional correlation coefficients in the interval [-1, 1].
        std: One-dimensional standard deviations normalized by the reference
            standard deviation. Must have the same length as ``r``.

    Raises:
        ValueError: If a correlation lies outside [-1, 1] or a standard
            deviation is negative.
    """

    if np.any((r < -1.0) | (r > 1.0)):
        raise ValueError("correlations must be in the interval [-1, 1]")
    if np.any(std < 0.0):
        raise ValueError("standard deviations cannot be negative")

    colors = ["k", "b", "r", "m", "tab:orange", "maroon", "g", "y"]
    fig, ax = plt.subplots(
        subplot_kw={
            "projection": "polar",
        }
    )
    positive_rlocs = np.concatenate(
        [np.arange(0.0, 1.0, 0.1).round(1), np.array([0.95, 0.99, 1.0])]
    )
    has_negative_correlation = bool(np.any(r < 0.0))
    theta_max = np.pi if has_negative_correlation else np.pi / 2
    if has_negative_correlation:
        rlocs = np.array(
            [
                -1.0,
                -0.99,
                -0.95,
                -0.9,
                -0.8,
                -0.6,
                -0.4,
                -0.2,
                0.0,
                0.2,
                0.4,
                0.6,
                0.8,
                0.9,
                0.95,
                0.99,
                1.0,
            ]
        )
    else:
        rlocs = positive_rlocs

    theta = np.arccos(r)

    lines, labels = ax.set_thetagrids(
        np.rad2deg(np.arccos(rlocs)), labels=rlocs, zorder=0
    )

    rmse_angular_samples = 600 if has_negative_correlation else 300
    rmse_theta = np.linspace(0.0, theta_max, rmse_angular_samples)
    rmse_std = np.linspace(0.0, 2.0, 300)
    rmse_theta_grid, rmse_std_grid = np.meshgrid(rmse_theta, rmse_std)
    rmse_grid = np.sqrt(
        1.0 + rmse_std_grid**2 - 2.0 * rmse_std_grid * np.cos(rmse_theta_grid)
    )
    rmse_contours = ax.contour(
        rmse_theta_grid,
        rmse_std_grid,
        rmse_grid,
        levels=np.arange(
            0.25, 3.01 if has_negative_correlation else 2.01, 0.25
        ),
        colors="0.45",
        linestyles=":",
        linewidths=0.9,
        zorder=1,
    )
    ax.clabel(rmse_contours, inline=True, fontsize=8, fmt="%g")

    ax.scatter(theta[1:], std[1:], c=colors[1:], zorder=10)
    reference_angular_samples = 400 if has_negative_correlation else 200
    ax.plot(
        np.linspace(0.0, theta_max, reference_angular_samples),
        np.ones(reference_angular_samples),
        color="black",
        linewidth=2.0,
        linestyle="--",
    )
    ax.scatter(
        theta[:1], std[:1], c=colors[:1], marker="*", s=70, clip_on=False, zorder=100
    )
    ax.set_rmax(2)
    ax.set_thetamin(0.0)
    ax.set_thetamax(np.rad2deg(theta_max))
    plt.show()


if __name__ == "__main__":
    taylor_correls(r, std)
