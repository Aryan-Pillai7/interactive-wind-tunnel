"""Plotting helpers (matplotlib). (Owner: Aryan)"""


def plot_fields(fields, mask, title=None):
    """Velocity magnitude, pressure and streamlines for one sample. Returns a Figure."""
    raise NotImplementedError


def plot_comparison(pred, true, mask):
    """Surrogate vs ground truth side by side with an error map. Returns a Figure."""
    raise NotImplementedError
