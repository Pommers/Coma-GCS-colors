# plot helpers
import numpy as np

def fit_line(x, y, order=1):
    """
    Simple linear fit for plotting only.
    Returns xfit, yfit or (None, None) if insufficient data.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 2:
        return None, None

    p = np.polyfit(x, y, order)
    xfit = np.linspace(np.min(x), np.max(x), 200)
    yfit = np.polyval(p, xfit)
    return xfit, yfit

    