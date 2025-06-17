import numpy as np
import torch
from typing import Dict
import astropy.io.fits as fits
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange


import torch
import subprocess

def get_free_gpu():
    """
    Selects the GPU with the most free memory using nvidia-smi.

    Returns:
        torch.device: The selected GPU device or CPU if no GPU is available.
    """
    try:
        # Execute nvidia-smi command to get GPU memory info
        result = subprocess.check_output(
            [
                'nvidia-smi',
                '--query-gpu=index,memory.free',
                '--format=csv,nounits,noheader'
            ]
        )
        gpu_info = result.decode('utf-8').strip().split('\n')

        # Parse the GPU information
        gpus = []
        for info in gpu_info:
            idx, free = map(int, info.split(','))
            gpus.append((idx, free))

        if not gpus:
            print("No GPU found. Using CPU instead.")
            return torch.device("cpu")

        # Select GPU with the most free memory
        selected_gpu = max(gpus, key=lambda x: x[1])[0]
        selected_free_memory = max(gpus, key=lambda x: x[1])[1]
        print(f"Selected GPU {selected_gpu} with {selected_free_memory} MiB free memory.")
        
        return torch.device(f"cuda:{selected_gpu}")

    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}. Using CPU instead.")
        return torch.device("cpu")
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed. Using CPU instead.")
        return torch.device("cpu")
    except Exception as e:
        print(f"Unexpected error: {e}. Using CPU instead.")
        return torch.device("cpu")


# ====================================================================
def retrieve_index(coordinates_org, tt):
    """
    Retrieve the indices of the coordinates being evaluated.

    Args:
        coordinates (numpy.ndarray): The original coordinates array.
        tt (int): The time step or index to evaluate.

    Returns:
        numpy.ndarray: A matrix of indices corresponding to the evaluated coordinates.
    """
    # Generate a matrix of indices for the entire coordinates array
    indices = np.arange(coordinates_org[...,0].detach().numpy().size).reshape(coordinates_org[...,0].detach().numpy().shape)
    
    # Extract the indices for the given time step tt
    indices_tt = indices[tt, :, :]
    
    # Rearrange the indices to match the shape (ny * nx)
    indices_tt = rearrange(indices_tt, 'ny nx -> (ny nx)')
    
    return np.squeeze(indices_tt)


# ====================================================================
def even_shape(x):
    """Make sure that the size of the image is even"""
    # make the img has a even shape
    if x.shape[0] % 2 != 0:
        x = x[...,:-1,:]
    if x.shape[1] % 2 != 0:
        x = x[...,:,:-1]
    return x


# ====================================================================
def print_summary(coordinates,nfmodel):
    nxyt = coordinates.shape[0]
    model = nfmodel
    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Number of pixels:', nxyt)
    print('Number of pixels per parameter:', (nxyt)/(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# ====================================================================
# Function to convert a number to a string in scientific notation
def formating(value):
    formatted_str = "{:.0e}".format(value)
    formatted_str = formatted_str.replace("e-0", " \\times 10^{-")
    formatted_str += "}"
    if value == 0:
        formatted_str = "0"
    return formatted_str
    

# ====================================================================
class AttributeDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


# ====================================================================
def writefits(name, d):
    io = fits.PrimaryHDU(d)
    io.writeto(name, overwrite=True)


# ====================================================================
def add_colorbar(im, aspect=20, pad_fraction=0.5, nbins=5, **kwargs):
    """Add a vertical color bar to an image plot."""
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    from matplotlib import ticker
    cb = im.axes.figure.colorbar(im, cax=cax, **kwargs)
    tick_locator = ticker.MaxNLocator(nbins)
    cb.locator = tick_locator
    cb.update_ticks()
    return cb

# ====================================================================
def optimization(optimizer,niterations,parameters,model, xl, img):
    from tqdm import trange
    t = trange(niterations, leave=True)
    for loop in t:
        optimizer.zero_grad()        #reset gradients
        parameters, final_output = evaluate(parameters, xl, img, model)

        chi2loss = torch.mean(torch.abs(img - final_output))
        reguloss = + 1e-4*regu(parameters, 0,img) + 1e-0*regu2(parameters, 1,img) + 5e-4*regu_value(parameters,1,img,0.014)+ 1e-3*regu(parameters, 2,img)

        loss = chi2loss + reguloss

        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward

        t.set_postfix({'loss': loss.item(), 'chi2loss': chi2loss.item(), 'reguloss': reguloss.item()})

    parameters, final_output = evaluate(parameters,xl,img, model)
    outplot = parameters.detach().numpy().reshape(img.shape[1],img.shape[2],3)
    return outplot, final_output

# ====================================================================
def Gaussian_model(x, params):
    x = torch.from_numpy(np.array(x.astype(np.float32)))
    sigma = torch.abs(params[:,1])
    return torch.pow(10,params[:,0])*torch.exp(-(x-params[:,2])**2./(2*(sigma+1e-6)**2.))




# ====================================================================
def evaluate(out,xl,img,model):
    final_output = torch.tensor(())
    for ii in range(xl.shape[0]):
        final_output = torch.cat((final_output, model(xl[ii], out)), 0)
    final_output = torch.reshape(final_output, (img.shape[0],img.shape[1],img.shape[2]))
    return out, final_output


# ====================================================================
def regu(out, ii, img):
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(mout[:,:,:,ii]-output_smooth))


# ====================================================================
def regu2(out, ii,img):
    # Squared diference
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(output_smooth-mout[:,:,:,ii])**2.0)


# ====================================================================
def regu2_angle(out, ii, img, plot=False,sine=False):
    # Squared diference for angles
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))

    sin_output_smooth = F.conv2d(m(torch.sin(2*mout[:,:,:,ii])), weights,padding='valid')
    cos_output_smooth = F.conv2d(m(torch.cos(2*mout[:,:,:,ii])), weights,padding='valid')

    if plot is True:
        fig = plt.figure(figsize=(9*1.5,4.5))
        plt.subplot(131)
        plt.imshow(sin_output_smooth.detach().numpy()[0,:,:],cmap='RdGy',interpolation='nearest')
        plt.subplot(132)
        plt.imshow(torch.sin(2*mout[:,:,:,ii]).detach().numpy()[0,:,:],cmap='RdGy',interpolation='nearest')
        plt.subplot(133)
        plt.imshow(torch.abs(sin_output_smooth-torch.sin(2*mout[:,:,:,ii])).detach().numpy()[0,:,:],cmap='gray',interpolation='nearest')
        plt.tight_layout()
        plt.savefig('test.pdf',bbox_inches='tight')
        plt.close(fig)
    
        fig = plt.figure(figsize=(9*1.5,4.5))
        plt.subplot(131)
        plt.imshow(cos_output_smooth.detach().numpy()[0,:,:],cmap='RdGy',interpolation='nearest')
        plt.subplot(132)
        plt.imshow(torch.cos(2*mout[:,:,:,ii]).detach().numpy()[0,:,:],cmap='RdGy',interpolation='nearest')
        plt.subplot(133)
        plt.imshow(torch.abs(cos_output_smooth-torch.cos(2*mout[:,:,:,ii])).detach().numpy()[0,:,:],cmap='gray',interpolation='nearest')
        plt.tight_layout()
        plt.savefig('test2.pdf',bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(9*1.5,4.5))
        plt.subplot(131)
        plt.imshow(F.conv2d(m(torch.cos(mout[:,:,:,ii])), weights,padding='valid').detach().numpy()[0,:,:],cmap='RdGy',interpolation='nearest')
        plt.subplot(132)
        plt.imshow(torch.cos(mout[:,:,:,ii]).detach().numpy()[0,:,:],cmap='RdGy',interpolation='nearest')
        plt.subplot(133)
        plt.imshow(torch.abs(F.conv2d(m(torch.cos(mout[:,:,:,ii])), weights,padding='valid')-torch.cos(mout[:,:,:,ii])).detach().numpy()[0,:,:],cmap='gray',interpolation='nearest')
        plt.tight_layout()
        plt.savefig('test2a.pdf',bbox_inches='tight')
        plt.close(fig)
    
        plt.figure()
        plt.imshow(torch.abs(torch.abs(sin_output_smooth-torch.sin(2*mout[:,:,:,ii]))**2.0).detach().numpy()[0,:,:] + \
         torch.abs(torch.abs(cos_output_smooth-torch.cos(2*mout[:,:,:,ii]))**2.0).detach().numpy()[0,:,:],cmap='gray',interpolation='nearest')
        plt.colorbar()
        plt.savefig('test2b.pdf',bbox_inches='tight')

        fig = plt.figure(figsize=(9*1.5,4.5))
        plt.subplot(131)
        plt.imshow(torch.arctan(sin_output_smooth/cos_output_smooth).detach().numpy()[0,:,:],cmap='twilight',interpolation='nearest')
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(torch.arctan(torch.sin(2*mout[:,:,:,ii])/torch.cos(2*mout[:,:,:,ii])).detach().numpy()[0,:,:],cmap='twilight',interpolation='nearest')
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(mout[:,:,:,ii].detach().numpy()[0,:,:],cmap='twilight',interpolation='nearest')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('test3.pdf',bbox_inches='tight')
        plt.close(fig)
    
    # arctan_output_smooth = torch.arctan2(sin_output_smooth,cos_output_smooth)/2.0
    # arctan_output = torch.arctan2(torch.sin(2*mout[:,:,:,ii]),torch.cos(2*mout[:,:,:,ii]))/2.0
    # return torch.sum(torch.abs(arctan_output_smooth-arctan_output)**2.0) #+ \

    return torch.sum(torch.abs(sin_output_smooth-torch.sin(2*mout[:,:,:,ii]))**2.0) + \
         torch.sum(torch.abs(cos_output_smooth-torch.cos(2*mout[:,:,:,ii]))**2.0)

    # if sine is True:
    #     return torch.sum(torch.abs(sin_output_smooth-torch.sin(2*mout[:,:,:,ii]))**2.0) 
    # else:
    #     return torch.sum(torch.abs(cos_output_smooth-torch.cos(2*mout[:,:,:,ii]))**2.0) 

# ====================================================================
def regu_mean(out, ii,img):
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))
    return torch.sum(torch.abs(mout[:,:,:,ii]-mout[:,:,:,ii].mean()))


# ====================================================================
def regu_min(out, ii,img, value):
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))
    return torch.sum(F.relu(-(mout[:,:,:,ii]-value)))


# ====================================================================
def regu_value(out, ii,img, value):
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))
    return torch.sum(torch.abs(mout[:,:,:,ii]-value))


# ====================================================================
def regu_mean3(out, ii,img):
    nb_channels = 1
    m = nn.ReflectionPad2d(1)
    weights = torch.tensor([[0.5,   1.0, 0.5],
                            [1.0,   0.0, 1.0],
                            [0.5,   1.0, 0.5]])
    weights = weights/weights.sum()
    weights = weights.view(1, 1, 3, 3).repeat(1, nb_channels, 1, 1)
    mout = torch.reshape(out, (1,img.shape[1],img.shape[2],3))
    output_smooth = F.conv2d(m(mout[:,:,:,ii]), weights,padding='valid')
    return torch.sum(torch.abs(output_smooth-mout[:,:,:,ii].mean()))





import numpy as np
import torch
from torch._C import dtype
from typing import Dict


DTYPE_BIT_SIZE: Dict[dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float64: 64,
    torch.double: 64,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.complex32: 32,
    torch.complex64: 64,
    torch.complex128: 128,
    torch.cdouble: 128,
    torch.uint8: 8,
    torch.int8: 8,
    torch.int16: 16,
    torch.short: 16,
    torch.int32: 32,
    torch.int: 32,
    torch.int64: 64,
    torch.long: 64,
    torch.bool: 1
}


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.

    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-0.5, 0.5]
    coordinates[:,0] = coordinates[:,0] / (np.max(img.shape[1])) - 0.5
    coordinates[:,1] = coordinates[:,1] / (np.max(img.shape[2])) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features

def to_coordinates(img):
    """Transform any set of coordinates to [-1,1]"""
    coordinates = torch.ones(img.shape[:]).nonzero(as_tuple=False).float()
    for icoor in range(coordinates.shape[1]):
        coordinates[:,icoor] = coordinates[:,icoor] / (torch.max(coordinates[:,icoor])) - 0.5
    coordinates *= 2
    return coordinates

def model_size_in_bits(model):
    """Calculate total number of bits to store `model` parameters and buffers."""
    return sum(sum(t.nelement() * DTYPE_BIT_SIZE[t.dtype] for t in tensors)
               for tensors in (model.parameters(), model.buffers()))


def bpp(image, model):
    """Computes size in bits per pixel of model.

    Args:
        image (torch.Tensor): Image to be fitted by model.
        model (torch.nn.Module): Model used to fit image.
    """
    num_pixels = np.prod(image.shape) / 3  # Dividing by 3 because of RGB channels
    return model_size_in_bits(model=model) / num_pixels


def psnr(img1, img2):
    """Calculates PSNR between two images.

    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    return 20. * np.log10(1.) - 10. * (img1 - img2).detach().pow(2).mean().log10().to('cpu').item()


def clamp_image(img):
    """Clamp image values to like in [0, 1] and convert to unsigned int.

    Args:
        img (torch.Tensor):
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, 0., 1.)
    # Pixel values lie in {0, ..., 255}, so round float tensor
    return torch.round(img_ * 255) / 255.


def get_clamped_psnr(img, img_recon):
    """Get PSNR between true image and reconstructed image. As reconstructed
    image comes from output of neural net, ensure that values like in [0, 1] and
    are unsigned ints.

    Args:
        img (torch.Tensor): Ground truth image.
        img_recon (torch.Tensor): Image reconstructed by model.
    """
    return psnr(img, clamp_image(img_recon))


def mean(list_):
    return np.mean(list_)

