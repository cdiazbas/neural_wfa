import numpy as np
import torch.nn as nn
import torch

# =================================================================
class line:
    """
    Class line is used to store the atomic data of spectral lines. We use this
    class as input for the WFA routines below.
    Usage: 
        lin = line(8542)
    """
    def __init__(self, cw=8542, verbose=False):

        self.larm = 4.668645048281451e-13
        
        if(cw == 8542):
            self.j1 = 2.5; self.j2 = 1.5; self.g1 = 1.2; self.g2 = 1.33; self.cw = 8542.091
        elif(cw == 6301):
            self.j1 = 2.0; self.j2 = 2.0; self.g1 = 1.84; self.g2 = 1.50; self.cw = 6301.4995
        elif(cw == 6302):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.49; self.g2 = 0.0; self.cw = 6302.4931
        elif(cw == 8468):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 2.50; self.g2 = 2.49; self.cw = 8468.4059
        elif(cw == 6173):
            self.j1 = 1.0; self.j2 = 0.0; self.g1 = 2.50; self.g2 = 0.0; self.cw = 6173.3340
        elif(cw == 5173):
            self.j1 = 1.0; self.j2 = 1.0; self.g1 = 1.50; self.g2 = 2.0; self.cw = 5172.6843
        elif(cw == 5896):
            self.j1 = 0.5; self.j2 = 0.5; self.g1 = 2.00; self.g2 = 2.0/3.0; self.cw = 5895.9242
        else:
            print("line::init: ERROR, line not implemented")
            self.j1 = 0.0; self.j2 = 0.0; self.g1 = 0.0; self.g2 = 0.0; self.cw = 0.0
            return

        j1 = self.j1; j2 = self.j2; g1 = self.g1; g2 = self.g2
        
        d = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        self.geff = 0.5 * (g1 + g2) + 0.25 * (g1 - g2) * d;
        ss = j1 * (j1 + 1.0) + j2 * (j2 + 1.0);
        dd = j1 * (j1 + 1.0) - j2 * (j2 + 1.0);
        gd = g1 - g2;
        self.Gg = (self.geff * self.geff) - (0.0125  * gd * gd * (16.0 * ss - 7.0 * dd * dd - 4.0));

        if verbose:
            print("line::init: cw={0}, geff={1}, Gg={2}".format(self.cw, self.geff, self.Gg))


# =================================================================
def cder(x, y):
    """
    function cder computes the derivatives of Stokes I (y)
    
    Input: 
            x: 1D wavelength array
            y: 4D data array (ny, nx, nStokes, nw)
            Use the usual centered derivatives formula for non-equidistant grids.
    """
    ny, nx, nstokes, nlam = y.shape[:]
    yp = np.zeros((ny, nx, nlam), dtype='float32')
    
    odx = x[1]-x[0]; ody = (y[:,:,0,1] - y[:,:,0,0]) / odx
    yp[:,:,0] = ody
    
    for ii in range(1,nlam-1):
        dx = x[ii+1] - x[ii]
        dy = (y[:,:,0,ii+1] - y[:,:,0,ii]) / dx
        
        yp[:,:,ii] = (odx * dy + dx * ody) / (dx + odx)
        
        odx = dx; ody = dy
    
    yp[:,:,-1] = ody    
    return yp



# =================================================================
class WFA_model(nn.Module):
    """ 
    Implements the WFA model by parametrizing the magnetic field as a neural field.
    """
    def __init__(self,data,wl,vdop= 0.035,mask=None, spectral_line=8542):
        super().__init__()
        self.lin = line(spectral_line)
        self.data = torch.from_numpy(np.array(data.astype(np.float32)))
        self.ny, self.nx, self.nStokes, self.nWav = self.data.shape
        # Wavelength relative to the center of the line
        self.wl = torch.from_numpy(np.array(wl.astype(np.float32)))
        dIdw = cder(wl, self.data)
        dIdw = dIdw.reshape(self.data.shape[0]*self.data.shape[1],dIdw.shape[2])
        self.dIdw = torch.from_numpy(np.array(dIdw.astype(np.float32)))
        self.scl = (1./(wl+1e-9))
        self.scl[np.abs(wl) <= vdop] = 0.0
        self.scl = torch.from_numpy(np.array(self.scl.astype(np.float32)))
        if mask is None: mask = range(self.data.shape[-1])
        self.mask = mask

        self.data_stokesQ = self.data[:,:,1,:].reshape(self.data.shape[0]*self.data.shape[1],self.data.shape[3])
        self.data_stokesU = self.data[:,:,2,:].reshape(self.data.shape[0]*self.data.shape[1],self.data.shape[3])
        self.data_stokesV = self.data[:,:,3,:].reshape(self.data.shape[0]*self.data.shape[1],self.data.shape[3])
        self.C = -4.67e-13 * self.lin.cw**2
        self.dIdwscl = self.dIdw * self.scl
        self.Vnorm = 1
        self.QUnorm = 1000

    def forward(self, params, index=None):
        Blos = params[:,0]*self.Vnorm
        BQ = params[:,1]*self.QUnorm
        BU = params[:,2]*self.QUnorm
        if index is None: index = range(0,len(self.dIdw))
        
        stokesV = self.C * self.lin.geff * Blos[:,None] * self.dIdw[index,:]
        Clp = 0.75 * self.C**2 * self.lin.Gg * self.dIdwscl[index,:] #Bt[:,None]**2
        stokesQ = Clp * BQ[:,None] #torch.cos(2*phiB)
        stokesU = Clp * BU[:,None] #torch.sin(2*phiB)
        return stokesQ, stokesU, stokesV

    def evaluate(self,params,weights=[1.0,1.0,1.0], index=None):
        stokesQ, stokesU, stokesV = self.forward(params,index=index)
        if index is None: index = range(0,len(self.dIdw))
        return weights[0]*torch.mean(torch.abs(self.data_stokesQ[index,:] - stokesQ)[:,self.mask]) + \
            weights[1]*torch.mean(torch.abs(self.data_stokesU[index,:] - stokesU)[:,self.mask]) + \
            weights[2]*torch.mean(torch.abs(self.data_stokesV[index,:] - stokesV)[:,self.mask])

    def initial_guess(self, inner=False, split=False):
        Blos = torch.sum(self.data_stokesV[:,self.mask]*self.dIdw[:,self.mask],dim=-1)/(self.C * self.lin.geff*torch.sum(self.dIdw[:,self.mask]**2,dim=-1))
        BtQ = (torch.sum(self.data_stokesQ[:,self.mask]*self.dIdwscl[:,self.mask],dim=-1)/(0.75 * self.C**2 * self.lin.Gg*torch.sum(self.dIdwscl[:,self.mask]**2,dim=-1)))
        BtU = (torch.sum(self.data_stokesU[:,self.mask]*self.dIdwscl[:,self.mask],dim=-1)/(0.75 * self.C**2 * self.lin.Gg*torch.sum(self.dIdwscl[:,self.mask]**2,dim=-1)))
        if inner is False:
            Bt = (BtQ**2 + BtU**2)**0.25
            phiB = 0.5 * torch.arctan2(BtU,BtQ)
            phiB[phiB < 0] += np.pi
            if split is False:
                return torch.stack((Blos,Bt,phiB),dim=-1)
            else:
                return Blos.reshape(self.nx,self.ny), Bt.reshape(self.nx,self.ny), phiB.reshape(self.nx,self.ny)
        else:
            return torch.stack((Blos,BtQ,BtU),dim=-1)


    def optimizeBlos(self,params, index=None):
        Blos = params[:,0]*self.Vnorm
        if index is None: index = range(0,len(self.dIdw))
        # move dIdw to same device as params:
        if self.dIdw.device != Blos.device:
            self.dIdw = self.dIdw.to(Blos.device)
            self.data_stokesV = self.data_stokesV.to(Blos.device)
        stokesV = self.C * self.lin.geff * Blos[:,None] * self.dIdw[index,:]
        return torch.mean(torch.abs(self.data_stokesV[index,:] - stokesV)[:,self.mask])


    def optimizeBQU(self,params, index=None):
        BQ = params[:,0]*self.QUnorm
        BU = params[:,1]*self.QUnorm
        if index is None: index = range(0,len(self.dIdw))
        Clp = 0.75 * self.C**2 * self.lin.Gg * self.dIdwscl[index,:] #Bt[:,None]**2
        stokesQ = Clp * BQ[:,None]
        stokesU = Clp * BU[:,None]
        return torch.mean(torch.abs(self.data_stokesQ[index,:] - stokesQ)[:,self.mask]) + \
            torch.mean(torch.abs(self.data_stokesU[index,:] - stokesU)[:,self.mask])





# =================================================================
class WFA_model3D(nn.Module):
    """ 
    Implements the WFA model by parametrizing the magnetic field as a neural field.
    """
    def __init__(self,data,wl,vdop= 0.035,mask=None, spectral_line=8542):
        super().__init__()
        self.lin = line(spectral_line)
        self.data = torch.from_numpy(np.array(data.astype(np.float32)))
        print('Data:',self.data.shape, 'should be in the format [(nt) ny nx ns nw]')
        print('Wav:',wl.shape,'should be in Angstroms relative to the center of the line')
        
        from einops import rearrange
        if len(self.data.shape) == 5: # nt ny nx ns nw
            self.nt, self.ny, self.nx, self.nStokes, self.nWav = self.data.shape
            self.data = rearrange(self.data,'nt ny nx ns nw -> (nt ny nx) ns nw')
        elif len(self.data.shape) == 4: # ny nx ns nw
            self.ny, self.nx, self.nStokes, self.nWav = self.data.shape
            self.data = rearrange(self.data,'ny nx ns nw -> (ny nx) ns nw')

        self.wl = torch.from_numpy(np.array(wl.astype(np.float32)))
        dIdw = cder(wl, self.data[None,...])[0,...]
        self.dIdw = torch.from_numpy(np.array(dIdw.astype(np.float32)))
        self.scl = (1./(wl+1e-9))
        self.scl[np.abs(wl) <= vdop] = 0.0
        self.scl = torch.from_numpy(np.array(self.scl.astype(np.float32)))
        if mask is None: mask = range(self.data.shape[-1])
        self.mask = mask

        self.data_stokesQ = self.data[:,1,:]
        self.data_stokesU = self.data[:,2,:]
        self.data_stokesV = self.data[:,3,:]
        self.C = -4.67e-13 * self.lin.cw**2
        self.dIdwscl = self.dIdw * self.scl
        self.Vnorm = 1#1000.0
        self.QUnorm = 1000#1e3#1e6
        
    def forward(self, params, index=None):
        Blos = params[:,0]*self.Vnorm
        BQ = params[:,1]*self.QUnorm
        BU = params[:,2]*self.QUnorm
        if index is None: index = range(0,len(self.dIdw))
        
        stokesV = self.C * self.lin.geff * Blos[:,None] * self.dIdw[index,:]
        Clp = 0.75 * self.C**2 * self.lin.Gg * self.dIdwscl[index,:]
        stokesQ = Clp * BQ[:,None]
        stokesU = Clp * BU[:,None]
        return stokesQ, stokesU, stokesV

    def evaluate(self,params,weights=[1.0,1.0,1.0], index=None):
        stokesQ, stokesU, stokesV = self.forward(params,index=index)
        if index is None: index = range(0,len(self.dIdw))
        return weights[0]*torch.mean(torch.abs(self.data_stokesQ[index,:] - stokesQ)[:,self.mask]) + \
            weights[1]*torch.mean(torch.abs(self.data_stokesU[index,:] - stokesU)[:,self.mask]) + \
            weights[2]*torch.mean(torch.abs(self.data_stokesV[index,:] - stokesV)[:,self.mask])

    def initial_guess(self, inner=False, split=False):
        Blos = torch.sum(self.data_stokesV[:,self.mask]*self.dIdw[:,self.mask],dim=-1)/(self.C * self.lin.geff*torch.sum(self.dIdw[:,self.mask]**2,dim=-1))
        BtQ = (torch.sum(self.data_stokesQ[:,self.mask]*self.dIdwscl[:,self.mask],dim=-1)/(0.75 * self.C**2 * self.lin.Gg*torch.sum(self.dIdwscl[:,self.mask]**2,dim=-1)))
        BtU = (torch.sum(self.data_stokesU[:,self.mask]*self.dIdwscl[:,self.mask],dim=-1)/(0.75 * self.C**2 * self.lin.Gg*torch.sum(self.dIdwscl[:,self.mask]**2,dim=-1)))
        if inner is False:
            Bt = (BtQ**2 + BtU**2)**0.25
            phiB = 0.5 * torch.arctan2(BtU,BtQ)
            phiB[phiB < 0] += np.pi
            if split is False:
                return torch.stack((Blos,Bt,phiB),dim=-1)
            else:
                return Blos, Bt, phiB
        else:
            return torch.stack((Blos,BtQ,BtU),dim=-1)


    def optimizeBlos(self,params, index=None, noise=0.0):
        Blos = params[:,0]*self.Vnorm
        if self.dIdw.device != params.device:
            self.dIdw = self.dIdw.to(Blos.device)
            self.data_stokesV = self.data_stokesV.to(Blos.device)
        if index is None: index = range(0,len(self.dIdw))
        stokesV = self.C * self.lin.geff * Blos[:,None] * self.dIdw[index,:]
        
        # Introduce regularization noise:
        if noise > 0.0:
            stokesV += torch.randn_like(stokesV)*noise
        
        return torch.mean(torch.abs(self.data_stokesV[index,:] - stokesV)[:,self.mask]**2.0)


    def optimizeBQU(self,params, index=None, noise=0.0):
        BQ = params[:,0]*self.QUnorm
        BU = params[:,1]*self.QUnorm
        if self.dIdwscl.device != params.device:
            self.dIdwscl = self.dIdwscl.to(params.device)
            self.data_stokesQ = self.data_stokesQ.to(params.device)
            self.data_stokesU = self.data_stokesU.to(params.device)
            
        
        if index is None: index = range(0,len(self.dIdwscl))
        Clp = 0.75 * self.C**2 * self.lin.Gg * self.dIdwscl[index,:]
        stokesQ = Clp * BQ[:,None]
        stokesU = Clp * BU[:,None]
        
        # Introduce regularization noise:
        if noise > 0.0:
            stokesQ += torch.randn_like(stokesQ)*noise
            stokesU += torch.randn_like(stokesU)*noise
        
        return torch.mean(torch.abs(self.data_stokesQ[index,:] - stokesQ)[:,self.mask]**2.0) + \
            torch.mean(torch.abs(self.data_stokesU[index,:] - stokesU)[:,self.mask]**2.0)



# =================================================================
def potential_extrapolation(Bz, zz=[0.0], pixel=[0.1,0.1]):
    """
    Computes a potential extrapolation from the observed vertical field.
    It uses a fast implementation in the Fourier space.

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
       3D array of dimensions (nx,ny,nz) with the magnetic field vector
    
    Example
    --------
    >>> a = readimage('Bz.fits')
    >>> Bvector = potential_extrapolation(Bz, zz=[0.0,1.0], pixel=[0.1,0.1])
    
    :Authors:
        Ported from the IDL /ssw/packages/nlfff/idl/FFF.pro (by Yuhong Fan) by Carlos Diaz (ISP-SU 2020)
        and simplified to the vertical case.
    
    """

    # Simplifications to the pure vertical case:
    b0=0.; pangle=0.; cmd=0.; lat=0.; alpha = 0.; bc = 0.; lc = 0.
    axx = 1.0; axy = 0.; axz = 0.; ayx = 0.
    ayy = 1.0; ayz = 0.; azx = 0.; azz = 1.0

    Nx1, Ny1 = Bz.shape
    nz = len(zz)
    dcterm = np.sum(Bz)/(float(Nx1)*float(Ny1))  # Flux imbalance
    Bz = Bz - dcterm

    cxx = axx ; cxy = ayx; cyx = axy; cyy = ayy
    fa = np.fft.fft2(Bz)/(Bz.shape[0]*Bz.shape[1]) # Normalization IDL fft
    fa[0,0] = 0.0   # make sure net flux is zero
    
    kxi = np.array([2*np.pi/Nx1*np.roll(np.arange(Nx1)- int((Nx1-1)/2),int(-(Nx1-1)/2))]*Ny1)
    kyi = 2*np.pi/Ny1*np.roll(np.arange(Ny1)- int((Ny1-1)/2),int(-(Ny1-1)/2)).T
    kyi = np.matmul( np.expand_dims(kyi,axis=-1), np.ones((1,Nx1)))
    dxi,dyi = pixel[0], pixel[1]
    dxi = abs((149e3/206265.0)*dxi)
    dyi = abs((149e3/206265.0)*dyi) # pixel size in Mm.
    kxi = kxi/dxi ; kyi = kyi/dyi # radians per Mm
    kx = cxx*kxi + cyx*kyi ; ky = cxy*kxi + cyy*kyi ; kz2 = (kx**2+ky**2) 
    k = np.sqrt(kz2-alpha**2.)
    kl2 = np.zeros_like(kz2) + kz2 * 1j
    kl2[0,0] = 1.0  # [0,0] is singular, do not divide by zero
    nx, ny = Bz.shape

    # Computing the vector field:
    iphihat0 = fa/kl2
    nz = len(zz);
    eps = 1e-10
    B = np.zeros((nx,ny,nz,3))
    for iz in range(0,nz):
        iphihat = iphihat0*np.exp(-k*zz[iz])
        fbx = (k*kx-alpha*ky)*iphihat
        fby = (k*ky+alpha*kx)*iphihat
        fbz = complex(0.,1.)*(kz2)*iphihat
        B[:,:,iz,0] = np.flipud(np.fliplr(np.real(np.fft.fft2(fbx,axes=(-1, -2))) +eps))
        B[:,:,iz,1] = np.flipud(np.fliplr(np.real(np.fft.fft2(fby,axes=(-1, -2))) +eps))
        B[:,:,iz,2] = np.flipud(np.fliplr(np.real(np.fft.fft2(fbz,axes=(-1, -2))) +eps))

    # Flux balance back
    B[:,:,:,2] = B[:,:,:,2] + dcterm
    return B



# =================================================================
def make_square(Bz_numpy):
    """ Embed the Bz_numpy into a square matrix for the potential extrapolation.
    """
    if Bz_numpy.shape[0] > Bz_numpy.shape[1]:
        Bz_numpy = Bz_numpy.T
    
    Bz_sq = np.zeros((max(Bz_numpy.shape),max(Bz_numpy.shape)))
    # Insert the smaller matrix into the middle of the larger matrix:
    Bz_sq[Bz_sq.shape[0]//2-Bz_numpy.shape[0]//2:Bz_sq.shape[0]//2+Bz_numpy.shape[0]//2,:] = Bz_numpy
    # Repeat the last row in the empty space:
    Bz_sq[0:Bz_sq.shape[0]//2-Bz_numpy.shape[0]//2,:] = Bz_numpy[0,:]
    Bz_sq[Bz_sq.shape[0]//2+Bz_numpy.shape[0]//2:,:] = Bz_numpy[-1,:]
    
    if Bz_numpy.shape[0] > Bz_numpy.shape[1]:
        Bz_sq = Bz_sq.T
    
    return Bz_sq


# =================================================================
def embed_potential_extrapolation(Bz_numpy, offset=0.0):
    """ 
    Embed the Bz_numpy into a square matrix, and then apply the potential extrapolation.
    Output: Bz,Btrans,phiB at two heights after adding the offset to phiB.
    """
    Bz_sq = make_square(Bz_numpy[:,:])
    
    # Crop to the original shize:
    newB = potential_extrapolation(Bz_sq, zz=[0.0,1.0], pixel=[1.0,1.0])[Bz_sq.shape[0]//2-Bz_numpy.shape[0]//2:Bz_sq.shape[0]//2+Bz_numpy.shape[0]//2,:,:]
    
    # Given that the output is Bx,By,Bz, and I want Bz,Btrans,phiB, I need to rearrange the output:
    newB_rearranged = np.zeros_like(newB)
    newB_rearranged[:,:,:,0] = newB[:,:,:,2]
    newB_rearranged[:,:,:,1] = np.sqrt(newB[:,:,:,0]**2.+newB[:,:,:,1]**2)
    newB_rearranged[:,:,:,2] = np.arctan(newB[:,:,:,1]/newB[:,:,:,0])

    # Add an offset of:
    newB_rearranged[:,:,:,2] += offset

    # Make the azimuth between 0 and 180 degrees:
    newB_rearranged[:,:,:,2][newB_rearranged[:,:,:,2] < 0] += np.pi
    return newB_rearranged



# =================================================================
def polar2bqu(B0,B1,B2):
    """ Blos, Btr, phiB -> Blos, BQ, BU
    """
    BQ = B1*B1 * np.cos(2*B2)
    BU = B1*B1 * np.sin(2*B2)
    return B0, BQ, BU

# =================================================================
def bqu2polar(B0,BQ,BU):
    """ Blos, BQ, BU -> Blos, Btr, phiB
    """
    Btr = np.sqrt(np.sqrt(BQ**2 + BU**2))
    phiB = 0.5*np.arctan2(BU,BQ)
    phiB[phiB < 0] += np.pi
    return B0, Btr, phiB


# =================================================================
def bqu2polar_cube(Bcube, split=False):
    """ Blos, BQ, BU -> Blos, Btr, phiB
    """
    Bcube = Bcube.copy()
    phiB = 0.5*np.arctan2(Bcube[:,:,2],Bcube[:,:,1])
    phiB[phiB < 0] += np.pi
    phiB[phiB > np.pi] -= np.pi

    Bt = np.sqrt(np.sqrt(Bcube[:,:,2]**2.+Bcube[:,:,1]**2))
    Bcube[:,:,1] = Bt
    Bcube[:,:,2] = phiB
    if split is False:
        return Bcube
    else:
        return Bcube[:,:,0], Bcube[:,:,1], Bcube[:,:,2]


