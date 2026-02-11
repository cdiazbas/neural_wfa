import numpy as np


def potential_extrapolation(Bz, zz=[0.0], pixel=[0.1, 0.1]):
    """
    Computes a potential extrapolation from the observed vertical field. It
    uses a fast implementation in the Fourier space.

    Parameters
    ----------
    Bz : ndarray
        2D array of dimensions (ny, nx) in Gauss
    zz : ndarray
        1D array of dimensions (nz) in Mm
    pixel : ndarray
        1D array of dimensions (2) with the pixel size in arcsec

    Returns
    -------
    ndarray
       3D array of dimensions (nx,ny,nz,3) with the magnetic field vector [Bx, By, Bz]

    :Authors:
        Ported from the IDL /ssw/packages/nlfff/idl/FFF.pro (by Yuhong Fan) by Carlos Diaz (ISP-SU 2020)
        and simplified to the vertical case.
    """

    # Simplifications to the pure vertical case:
    alpha = 0.0
    axx = 1.0
    axy = 0.0
    ayx = 0.0
    ayy = 1.0

    Nx1, Ny1 = Bz.shape
    nz = len(zz)
    dcterm = np.sum(Bz) / (float(Nx1) * float(Ny1))  # Flux imbalance
    Bz_norm = Bz - dcterm

    cxx = axx
    cxy = ayx
    cyx = axy
    cyy = ayy
    fa = np.fft.fft2(Bz_norm) / (
        Bz_norm.shape[0] * Bz_norm.shape[1]
    )  # Normalization IDL fft
    fa[0, 0] = 0.0  # make sure net flux is zero

    kxi = np.array(
        [
            2
            * np.pi
            / Nx1
            * np.roll(np.arange(Nx1) - int((Nx1 - 1) / 2), int(-(Nx1 - 1) / 2))
        ]
        * Ny1
    )
    kyi = (
        2
        * np.pi
        / Ny1
        * np.roll(np.arange(Ny1) - int((Ny1 - 1) / 2), int(-(Ny1 - 1) / 2)).T
    )
    kyi = np.matmul(np.expand_dims(kyi, axis=-1), np.ones((1, Nx1)))
    dxi, dyi = pixel[0], pixel[1]
    dxi = abs((149e3 / 206265.0) * dxi)
    dyi = abs((149e3 / 206265.0) * dyi)  # pixel size in Mm.
    kxi = kxi / dxi
    kyi = kyi / dyi  # radians per Mm
    kx = cxx * kxi + cyx * kyi
    ky = cxy * kxi + cyy * kyi
    kz2 = kx**2 + ky**2
    k = np.sqrt(kz2 - alpha**2.0)
    kl2 = np.zeros_like(kz2) + kz2 * 1j
    kl2[0, 0] = 1.0  # [0,0] is singular, do not divide by zero
    nx, ny = Bz.shape

    # Computing the vector field:
    iphihat0 = fa / kl2
    nz = len(zz)
    eps = 1e-10
    B = np.zeros((nx, ny, nz, 3))
    for iz in range(0, nz):
        iphihat = iphihat0 * np.exp(-k * zz[iz])
        fbx = (k * kx - alpha * ky) * iphihat
        fby = (k * ky + alpha * kx) * iphihat
        fbz = complex(0.0, 1.0) * (kz2) * iphihat
        B[:, :, iz, 0] = np.flipud(
            np.fliplr(np.real(np.fft.fft2(fbx, axes=(-1, -2))) + eps)
        )
        B[:, :, iz, 1] = np.flipud(
            np.fliplr(np.real(np.fft.fft2(fby, axes=(-1, -2))) + eps)
        )
        B[:, :, iz, 2] = np.flipud(
            np.fliplr(np.real(np.fft.fft2(fbz, axes=(-1, -2))) + eps)
        )

    # Flux balance back
    B[:, :, :, 2] = B[:, :, :, 2] + dcterm
    return B


def make_square(Bz_numpy):
    """
    Embed the Bz_numpy into a square matrix for the potential extrapolation.
    """
    if Bz_numpy.shape[0] > Bz_numpy.shape[1]:
        Bz_numpy = Bz_numpy.T

    Bz_sq = np.zeros((max(Bz_numpy.shape), max(Bz_numpy.shape)))
    # Insert the smaller matrix into the middle of the larger matrix:
    Bz_sq[
        Bz_sq.shape[0] // 2 - Bz_numpy.shape[0] // 2 : Bz_sq.shape[0] // 2
        + Bz_numpy.shape[0] // 2,
        :,
    ] = Bz_numpy
    # Repeat the last row in the empty space:
    Bz_sq[0 : Bz_sq.shape[0] // 2 - Bz_numpy.shape[0] // 2, :] = Bz_numpy[0, :]
    Bz_sq[Bz_sq.shape[0] // 2 + Bz_numpy.shape[0] // 2 :, :] = Bz_numpy[-1, :]

    if Bz_numpy.shape[0] > Bz_numpy.shape[1]:
        Bz_sq = Bz_sq.T

    return Bz_sq


def embed_potential_extrapolation(Bz_numpy, offset=0.0):
    """
    Embed the Bz_numpy into a square matrix, and then apply the potential
    extrapolation.

    Output: Bz, 1D_Btrans, phiB at two heights after adding the offset to phiB.
    """
    Bz_sq = make_square(Bz_numpy[:, :])

    # Crop to the original shize:
    newB = potential_extrapolation(Bz_sq, zz=[0.0, 1.0], pixel=[1.0, 1.0])[
        Bz_sq.shape[0] // 2 - Bz_numpy.shape[0] // 2 : Bz_sq.shape[0] // 2
        + Bz_numpy.shape[0] // 2,
        :,
        :,
    ]

    # Given that the output is Bx,By,Bz, and I want Bz,Btrans,phiB, I need to rearrange the output:
    newB_rearranged = np.zeros_like(newB)
    newB_rearranged[:, :, :, 0] = newB[:, :, :, 2]
    newB_rearranged[:, :, :, 1] = np.sqrt(
        newB[:, :, :, 0] ** 2.0 + newB[:, :, :, 1] ** 2
    )
    newB_rearranged[:, :, :, 2] = np.arctan2(
        newB[:, :, :, 1], newB[:, :, :, 0]
    )  # Fixed atan to atan2 for stability

    # Add an offset
    newB_rearranged[:, :, :, 2] += offset

    # Make the azimuth between 0 and 180 degrees:
    newB_rearranged[:, :, :, 2][newB_rearranged[:, :, :, 2] < 0] += np.pi
    newB_rearranged[:, :, :, 2] = np.mod(newB_rearranged[:, :, :, 2], np.pi)

    return newB_rearranged
