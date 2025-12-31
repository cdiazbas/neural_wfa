import astropy.io.fits as fits
import numpy as np
import os

def writefits(filename: str, data: np.ndarray, header=None, overwrite: bool = True):
    """
    Writes a NumPy array to a FITS file.
    
    Args:
        filename (str): Output path.
        data (np.ndarray): Data to save.
        header: Optional FITS header.
        overwrite (bool): Whether to overwrite existing file.
    """
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f"File {filename} already exists.")
        
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(filename, overwrite=overwrite)

def readfits(filename: str) -> np.ndarray:
    """
    Reads data from a FITS file.
    """
    with fits.open(filename) as hdul:
        return hdul[0].data.copy()
