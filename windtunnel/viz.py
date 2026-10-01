"""Plotting helpers (matplotlib). (Owner: Aryan)"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from windtunnel import contract as C  # noqa: E402

X = np.arange(C.NX)
Y = np.arange(C.NY)


def _obstacle(ax, mask):
    ax.contourf(X, Y, mask.astype(float), levels=[0.5, 1.5], colors=["#444"])


def _panel(ax, data, mask, title, cmap, vmin=None, vmax=None, streamlines=None):
    img = np.ma.masked_where(mask, data)
    if vmin is None and vmax is None:
        # Clip to the 1st-99th percentile so a few inlet-corner cells don't wash out the map.
        vmin, vmax = np.percentile(np.asarray(data)[~mask], [1, 99])
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    if streamlines is not None:
        u, v = streamlines
        ax.streamplot(X, Y, np.ma.masked_where(mask, u), np.ma.masked_where(mask, v),
                      color="white", linewidth=0.6, density=1.2, arrowsize=0.6)
    _obstacle(ax, mask)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, C.NX - 0.5)
    ax.set_ylim(-0.5, C.NY - 0.5)
    plt.colorbar(im, ax=ax, fraction=0.023, pad=0.02)


def plot_fields(fields, mask, title=None):
    """Velocity magnitude with streamlines, and pressure, for one sample. Returns a Figure."""
    mask = np.asarray(mask, bool)
    u, v, p = fields
    fig, axes = plt.subplots(2, 1, figsize=(7, 6.4), constrained_layout=True)
    _panel(axes[0], np.hypot(u, v) / C.U0, mask, "|U| / U0 with streamlines", "viridis",
           streamlines=(u, v))
    _panel(axes[1], p, mask, "pressure coefficient p (gauge, outlet reference)", "RdBu_r")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def plot_comparison(pred, true, mask, title=None):
    """Surrogate vs ground truth side by side with an error map. Returns a Figure."""
    mask = np.asarray(mask, bool)
    fig, axes = plt.subplots(2, 3, figsize=(15, 4.8), constrained_layout=True)
    for row, (name, f) in enumerate([("|U| / U0", lambda x: np.hypot(x[0], x[1]) / C.U0),
                                     ("p", lambda x: x[2])]):
        t, q = f(true), f(pred)
        lo, hi = (float(v) for v in np.percentile(t[~mask], [1, 99]))
        cmap = "viridis" if row == 0 else "RdBu_r"
        _panel(axes[row, 0], t, mask, f"ground truth (LBM): {name}", cmap, lo, hi)
        _panel(axes[row, 1], q, mask, f"surrogate: {name}", cmap, lo, hi)
        _panel(axes[row, 2], np.abs(q - t), mask, f"|error|: {name}", "magma")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def plot_error_hist(errors, title, xlabel):
    """errors: {label: 1D array}. Returns a Figure."""
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    allv = np.concatenate([np.asarray(v) for v in errors.values() if len(v)])
    bins = np.linspace(0, np.percentile(allv, 99) if allv.size else 1, 30)
    for label, v in errors.items():
        ax.hist(np.clip(v, 0, bins[-1]), bins=bins, alpha=0.55, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("samples")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    return fig
