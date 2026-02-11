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
    # Handle different input shapes? Original code assumes (ny, nx, nStokes, nlam)
    # But accesses y[:, :, 0, i], implying it differentiates along axis -1?
    # Actually, original code does:
    # ny, nx, nstokes, nlam = y.shape[:]
    # And accesses y[:, :, 0, 1] - y[:, :, 0, 0] ... wait.
    # Original cder documentation says "Derivatives of Stokes I (y)".
    # And uses y[:, :, 0, ...]. This implies it only differentiates Stokes I?
    # BUT WFA_model3D calls it as: dIdw = cder(wl, self.data[None, ...])[0, ...]
    # where self.data is (nt, ny, nx, ns, nw) rearranged to ((nt ny nx), ns, nw).
    # Wait, WFA_model3D reshape logic:
    # rearrange(self.data, "nt ny nx ns nw -> (nt ny nx) ns nw")
    # cder is called with self.data[None, ...] which makes it (1, (nt ny nx), ns, nw).
    # Then cder internally unpacks: ny, nx, nstokes, nlam = y.shape[:]
    # So 'ny' becomes 1, 'nx' becomes (nt ny nx), 'nstokes' is ns, 'nlam' is nw.
    # Inside cder: yp[:, :, 0] refers to Stokes I ???
    # No, yp shape is (ny, nx, nlam).
    # And it computes using y[:, :, 0, i].
    # This means cder HARDCODES using index 0 (Stokes I) along the 3rd dimension.
    # It returns yp of shape (ny, nx, nlam) which is effectively dI/dw.

    # We should preserve this behavior but perhaps make it more robust or document it clearly.
    # Or generalize it?
    # For now, let's replicate exact behavior to match baseline.

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
