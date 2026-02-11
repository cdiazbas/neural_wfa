import numpy as np


def cder(x, y):
    """
    Computes the derivatives of Stokes vectors (y) with respect to wavelength
    (x).

    Uses centered derivatives formula adapted for non-equidistant grids.

    Args:
        x (np.ndarray): 1D wavelength array of length nlam.
        y (np.ndarray): 4D/nD data array (..., nlam).
                        Original code expects (ny, nx, nStokes, nlam).

    Returns:
        np.ndarray: Derivative array of same shape as y.
    """
    y = np.ascontiguousarray(y)  # ensure contiguity if needed

    # Check dimensions to ensure strict compatibility with legacy usage
    if y.ndim != 4:
        raise ValueError(
            f"cder expects 4D input array (N1, N2, N_Stokes, N_wav), got {y.ndim}D"
        )

    N1, N2, nstokes, nlam = y.shape
    yp = np.zeros((N1, N2, nlam), dtype="float32")

    odx = x[1] - x[0]
    # Use Stokes I (index 0 along axis 2)
    body = (y[:, :, 0, 1] - y[:, :, 0, 0]) / odx
    yp[:, :, 0] = body

    for ii in range(1, nlam - 1):
        dx = x[ii + 1] - x[ii]
        dy = (y[:, :, 0, ii + 1] - y[:, :, 0, ii]) / dx

        yp[:, :, ii] = (odx * dy + dx * body) / (dx + odx)

        odx = dx
        body = dy

    yp[:, :, -1] = body
    return yp
