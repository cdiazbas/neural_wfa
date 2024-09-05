import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from . import utils


# ===================================================================
class temporal_mlp3(nn.Module):
    """
    DenseNet with Fourier Features
    """
    def __init__(self, dim_in=2, dim_out=1, num_resnet_blocks=3, 
                 num_layers_per_block=2, dim_hidden=50, activation=nn.GELU(), 
                 fourier_features=False, m_freqs=100, sigma=10, tune_beta=False):
        super(temporal_mlp3, self).__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.sigma = sigma
        self.m_freqs = m_freqs
        num_neurons = dim_hidden
        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block))
        else: 
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        self.first = nn.Linear(dim_in, num_neurons)
        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons) 
                for _ in range(num_layers_per_block)]) 
            for _ in range(num_resnet_blocks)])
        self.last = nn.Linear(num_neurons, dim_out)
        if fourier_features:
            self.first = nn.Linear(6*m_freqs+dim_in, num_neurons)
            # self.B = torch.from_numpy(np.random.normal(0.,sigma, size=(dim_in,m_freqs))).float() # Only valid for a single entry

            # sigma can be a float or a list of floats:
            n_param = len(sigma) if isinstance(sigma, list) else dim_in
            self.B = torch.from_numpy(np.random.normal(0.0, sigma, (m_freqs, n_param)).T).float()
            # print('sigma:',sigma,'self.B.shape',self.B.shape)

    def forward(self, x):
        xx = x.clone()
                
        # if x in device, move also B to device:
        if x.device != self.beta0.device:
            self.beta0 = self.beta0.to(x.device)
            # if self.B exist
            if hasattr(self, 'B'):
                self.B = self.B.to(x.device)

        if self.fourier_features:
            # print(x.shape,self.B.shape)
            # cosx = torch.cos(torch.matmul(x, self.B))
            # sinx = torch.sin(torch.matmul(x, self.B))
            
            # From [cos(2πBsx),sin(2πBsx),cos(2πBdx) cos(2πt),cos(2πBdx) sin(2πt),sin(2πBdx) cos(2πt), sin(2πBdx) sin(2πt)]
            xx_spatial = x[:,1:]
            xx_temporal = x[:,0:1]
            
            cosx_spatial = torch.cos(torch.matmul(xx_spatial, self.B[1:,:]))
            sinx_spatial = torch.sin(torch.matmul(xx_spatial, self.B[1:,:]))
            cosx_temporal = torch.cos(torch.matmul(xx_temporal, self.B[0:1,:]))
            sinx_temporal = torch.sin(torch.matmul(xx_temporal, self.B[0:1,:]))
            
            
            x = torch.cat((cosx_spatial, sinx_spatial, cosx_temporal*cosx_spatial, cosx_temporal*sinx_spatial, sinx_temporal*cosx_spatial, sinx_temporal*sinx_spatial, xx_spatial, xx_temporal), dim=1)
            # x = torch.cat((cosx_spatial, sinx_spatial, sinx_temporal, xx_spatial, xx_temporal), dim=1)
            # cosx[:,1:]*cosx[:,0:1], cosx[:,1:]*sinx[:,0:1], sinx[:,1:]*cosx[:,0:1], sinx[:,1:]*sinx[:,0:1],
            # print('x.shape',x.shape, self.m_freqs)
            
            
            # x = torch.cat((cosx, sinx, xx), dim=1)
            x = self.activation(self.beta0*self.first(x)) 

        else:
            x = self.activation(self.beta0*self.first(x))

        for i in range(self.num_resnet_blocks):
            # print(self.beta.shape)
            z = self.activation(self.beta[i][0]*self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j]*self.resblocks[i][j](z))
            x = z + x
        out = self.last(x)
        return out


# ===================================================================
class temporal_mlp2(nn.Module):
    """
    The temporal information is treated as a Film: Feature-wise Linear Modulation
    """
    def __init__(self, dim_in=3, dim_out=1, num_resnet_blocks=3, 
                 num_layers_per_block=2, dim_hidden=50, activation=nn.GELU(), 
                 fourier_features=False, m_freqs=100, sigma=10, tune_beta=False):
        super(temporal_mlp2, self).__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.num_neurons = dim_hidden
        self.sigma = sigma
        
        # Let's use a Film network:
        self.film = mlp(dim_in=1, dim_out=self.num_neurons,num_resnet_blocks=num_resnet_blocks, 
                 num_layers_per_block=num_layers_per_block, dim_hidden=self.num_neurons, activation=activation, 
                 fourier_features=True, m_freqs=m_freqs, sigma=sigma[0], tune_beta=False)
        
        sigma_spatial = sigma[1:] if isinstance(sigma, list) else sigma
        self.beta0 = torch.ones(1, 1)

        dim_in = 2
        self.first = nn.Linear(dim_in, self.num_neurons)
        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.num_neurons, self.num_neurons) 
                for _ in range(num_layers_per_block)]) 
            for _ in range(num_resnet_blocks)])
        self.last = nn.Linear(self.num_neurons, dim_out)
        if fourier_features:
            self.first = nn.Linear(2*m_freqs+dim_in, self.num_neurons)
            # self.B = torch.from_numpy(np.random.normal(0.,sigma, size=(dim_in,m_freqs))).float() # Only valid for a single entry

            # sigma can be a float or a list of floats:
            n_param = len(sigma_spatial) if isinstance(sigma_spatial, list) else 1
            self.B = torch.from_numpy(np.random.normal(0.0, sigma_spatial, (m_freqs, n_param)).T).float()
            # print('sigma:',sigma_spatial,'self.B.shape',self.B.shape)

    def forward(self, x):
        # if x in device, move also B to device:
        if x.device != self.beta0.device:
            self.beta0 = self.beta0.to(x.device)
            # if self.B exist
            if hasattr(self, 'B'):
                self.B = self.B.to(x.device)
               
        xx_spatial = x[:,1:]
        xx_temporal = x[:,0:1]

        beta = self.film(xx_temporal)
        #.reshape(xx_temporal.shape[0],self.num_neurons)

        if self.fourier_features:
            cosx = torch.cos(torch.matmul(xx_spatial, self.B))
            sinx = torch.sin(torch.matmul(xx_spatial, self.B))
            x = torch.cat((cosx, sinx, xx_spatial), dim=1)
            x = self.activation(self.beta0*self.first(x)) 

        else:
            x = self.activation(self.beta0*self.first(xx_spatial))

        # x = F.glu(torch.cat((x, beta+1.0), dim=1), dim=1)
        x = x*(beta+1.0)
        for i in range(self.num_resnet_blocks):
            # print(beta.shape,beta[:,i].shape,self.resblocks[i][0](x).shape)
            z = self.activation(self.resblocks[i][0](x))
            z = z*(beta+1.0)
            # z = F.glu(torch.cat((z, beta+1.0), dim=1), dim=1)

            for j in range(1, self.num_layers_per_block):
                # z = self.activation(beta[:,i]*self.resblocks[i][j](z))
                z = self.activation(self.resblocks[i][j](z))
                z = z*(beta+1.0)
                # z = F.glu(torch.cat((z, beta+1.0), dim=1), dim=1)

            x = z + x
        # x = z

        # print(beta.max(),beta.min(),beta.mean(),beta.std())
        # print(x.max(),x.min(),x.mean(),x.std())
        # x = F.glu(torch.cat((x, beta+1.0 ), dim=1), dim=1)
        x = x*(beta+1.0)
        out = self.last(x)
        return out



# ===================================================================
class temporal_mlp(nn.Module):
    """
    The temporal information is treated as a Film: Feature-wise Linear Modulation
    """
    def __init__(self, dim_in=3, dim_out=1, num_resnet_blocks=3, 
                 num_layers_per_block=2, dim_hidden=50, activation=nn.GELU(), 
                 fourier_features=False, m_freqs=100, sigma=10, tune_beta=False):
        super(temporal_mlp, self).__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.num_neurons = dim_hidden
        self.sigma = sigma
        
        # Let's use a Film network:
        self.film = mlp(dim_in=1, dim_out=self.num_resnet_blocks*1,num_resnet_blocks=num_resnet_blocks, 
                 num_layers_per_block=num_layers_per_block, dim_hidden=self.num_neurons, activation=activation, 
                 fourier_features=True, m_freqs=m_freqs, sigma=sigma[0], tune_beta=False)
        
        sigma_spatial = sigma[1:] if isinstance(sigma, list) else sigma
        self.beta0 = torch.ones(1, 1)

        dim_in = 2
        self.first = nn.Linear(dim_in, self.num_neurons)
        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(self.num_neurons, self.num_neurons) 
                for _ in range(num_layers_per_block)]) 
            for _ in range(num_resnet_blocks)])
        self.last = nn.Linear(self.num_neurons, dim_out)
        if fourier_features:
            self.first = nn.Linear(2*m_freqs+dim_in, self.num_neurons)
            # self.B = torch.from_numpy(np.random.normal(0.,sigma, size=(dim_in,m_freqs))).float() # Only valid for a single entry

            # sigma can be a float or a list of floats:
            n_param = len(sigma_spatial) if isinstance(sigma_spatial, list) else 1
            self.B = torch.from_numpy(np.random.normal(0.0, sigma_spatial, (m_freqs, n_param)).T).float()
            # print('sigma:',sigma_spatial,'self.B.shape',self.B.shape)

    def forward(self, x):
        # if x in device, move also B to device:
        if x.device != self.beta0.device:
            self.beta0 = self.beta0.to(x.device)
            # if self.B exist
            if hasattr(self, 'B'):
                self.B = self.B.to(x.device)
               
        xx_spatial = x[:,1:]
        xx_temporal = x[:,0:1]

        beta = self.film(xx_temporal).reshape(xx_temporal.shape[0],self.num_resnet_blocks, 1)

        if self.fourier_features:
            cosx = torch.cos(torch.matmul(xx_spatial, self.B))
            sinx = torch.sin(torch.matmul(xx_spatial, self.B))
            x = torch.cat((cosx, sinx, xx_spatial), dim=1)
            x = self.activation(self.beta0*self.first(x)) 

        else:
            x = self.activation(self.beta0*self.first(xx_spatial))

        for i in range(self.num_resnet_blocks):
            # print(beta.shape,beta[:,i].shape,self.resblocks[i][0](x).shape)
            z = self.activation(beta[:,i]+self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(beta[:,i]+self.resblocks[i][j](z))
            x = z + x
        out = self.last(x)
        return out


# ===================================================================
class mlp(nn.Module):
    """
    DenseNet with Fourier Features
    """
    def __init__(self, dim_in=2, dim_out=1, num_resnet_blocks=3, 
                 num_layers_per_block=2, dim_hidden=50, activation=nn.GELU(), 
                 fourier_features=False, m_freqs=100, sigma=10, tune_beta=False):
        super(mlp, self).__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.sigma = sigma
        num_neurons = dim_hidden
        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_resnet_blocks, self.num_layers_per_block))
        else: 
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        self.first = nn.Linear(dim_in, num_neurons)
        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(num_neurons, num_neurons) 
                for _ in range(num_layers_per_block)]) 
            for _ in range(num_resnet_blocks)])
        self.last = nn.Linear(num_neurons, dim_out)
        if fourier_features:
            self.first = nn.Linear(2*m_freqs+dim_in, num_neurons)
            # self.B = torch.from_numpy(np.random.normal(0.,sigma, size=(dim_in,m_freqs))).float() # Only valid for a single entry

            # sigma can be a float or a list of floats:
            n_param = len(sigma) if isinstance(sigma, list) else dim_in
            self.B = torch.from_numpy(np.random.normal(0.0, sigma, (m_freqs, n_param)).T).float()
            # print('sigma:',sigma,'self.B.shape',self.B.shape)

    def forward(self, x):
        xx = x.clone()
                
        # if x in device, move also B to device:
        if x.device != self.beta0.device:
            self.beta0 = self.beta0.to(x.device)
            # if self.B exist
            if hasattr(self, 'B'):
                self.B = self.B.to(x.device)

        if self.fourier_features:
            cosx = torch.cos(torch.matmul(x, self.B))
            sinx = torch.sin(torch.matmul(x, self.B))
            x = torch.cat((cosx, sinx, xx), dim=1)
            x = self.activation(self.beta0*self.first(x)) 

        else:
            x = self.activation(self.beta0*self.first(x))

        for i in range(self.num_resnet_blocks):
            # print(self.beta.shape)
            z = self.activation(self.beta[i][0]*self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j]*self.resblocks[i][j](z))
            x = z + x
        out = self.last(x)
        return out





# ====================================================================
def Trainer(nfmodel, wfamodel, coordinates, niter=1000, lrinit=1e-3, batchcoord=2000, trainBV=True, guesshelp=None, guess_regu=0.0):
    """ 
    Train the neural field model using the WFA model as a loss function.
    """
    
    from tqdm import tqdm, trange
    optimizer = torch.optim.Adam(nfmodel.parameters(), lr=lrinit)

    wholedataset = np.arange(coordinates.shape[0])
    if batchcoord < 0: batchcoord = coordinates.shape[0]

    # Scheduler so learning rate decreases after no improvement in loss:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, verbose=True)

    loss_list = []
    lr_list = []
    t = trange(niter, leave=True)
    for loop in t:
        np.random.shuffle(wholedataset)
        optimizer.zero_grad()        #reset gradients

        # Forward pass:
        out = nfmodel(coordinates[wholedataset[:batchcoord],:])

        if trainBV:
            loss = wfamodel.optimizeBlos(params=out,index=wholedataset[:batchcoord])

            # print('loss = {:.2e}'.format(loss.item()))
            if guesshelp is not None:
                # print(guesshelp.shape, type(guesshelp))
                
                
                loss += guess_regu*(out[:,0] - guesshelp[wholedataset[:batchcoord],0]).pow(2).mean()
            # print('loss = {:.2e}'.format(loss.item()))

        else: # train the tranverse component
            loss = wfamodel.optimizeBQU(params=out,index=wholedataset[:batchcoord])
            
            # print('loss = {:.2e}'.format(loss.item()))
            if guesshelp is not None:
                # evaluate only every two iterations
                loss += guess_regu*(out[:,0] - guesshelp[wholedataset[:batchcoord],0]).pow(2).mean()
                loss += guess_regu*(out[:,1] - guesshelp[wholedataset[:batchcoord],1]).pow(2).mean()
            # print('loss = {:.2e}'.format(loss.item()))

        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward
        t.set_postfix_str('loss = {:.2e}'.format(loss.item()))

        # Save loss:
        loss_list.append(loss.item())
        lr_list.append(optimizer.param_groups[0]['lr'])

        scheduler.step(loss)

        # If learning rate is too small, stop:
        if optimizer.param_groups[0]['lr'] < 1e-5:
            print('Learning rate too small. Stopping.')
            break

    output_dict = {'loss': loss_list, 'lr': lr_list}
    return output_dict





def Trainer_gpu(nfmodel, wfamodel, coordinates, niter=1000, lrinit=1e-3, batchcoord=2000, trainBV=True, guesshelp=None, guess_regu=0.0, device='cuda',patience=50, normgrad = True, noise=0.0):
    """ 
    Train the neural field model using the WFA model as a loss function.
    """

    if device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
        if device_name != 'cpu':
            print('Using device:',device_name)
    else:
        device = "cpu"

    from tqdm import tqdm, trange

    # Move model and coordinates to GPU if available
    nfmodel = nfmodel.to(device)
    coordinates = torch.tensor(coordinates, dtype=torch.float32).to(device)
    if guesshelp is not None:
        guesshelp = torch.tensor(guesshelp, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.Adam(nfmodel.parameters(), lr=lrinit)
    wholedataset = np.arange(coordinates.shape[0])
    if batchcoord < 0:
        batchcoord = coordinates.shape[0]

    # Scheduler so learning rate decreases after no improvement in loss:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)

    loss_list = []
    lr_list = []
    t = trange(niter, leave=True)
    for loop in t:
        np.random.shuffle(wholedataset)
        optimizer.zero_grad()  # reset gradients

        # Forward pass:
        out = nfmodel(coordinates[wholedataset[:batchcoord], :])

        if trainBV:
            loss = wfamodel.optimizeBlos(params=out, index=wholedataset[:batchcoord],noise=noise)


        else:  # train the transverse component
            loss = wfamodel.optimizeBQU(params=out, index=wholedataset[:batchcoord],noise=noise)
    

        loss.backward()  # calculate gradients
        
        # Add gradient norm:
        if normgrad:
            for parameters in nfmodel.parameters():
                parameters.grad = parameters.grad / (torch.mean(torch.abs(parameters.grad),dim=0) + 1e-9)
                
        
        optimizer.step()  # step forward
        t.set_postfix_str('loss = {:.2e}'.format(loss.item()))

        # Save loss:
        loss_list.append(loss.item())
        lr_list.append(optimizer.param_groups[0]['lr'])

        scheduler.step(loss)

        # If learning rate is too small, stop:
        if optimizer.param_groups[0]['lr'] < 1e-5:
            print('Learning rate too small. Stopping.')
            break

    output_dict = {'loss': loss_list, 'lr': lr_list}
    
    # Move model back to CPU:
    nfmodel = nfmodel.to('cpu')
    
    return output_dict




# ====================================================================
def plot_loss(output_dict):
    import matplotlib.pyplot as plt

    # Loss plot:
    plt.figure()
    plt.plot(output_dict['loss'])
    if len(output_dict['loss'])>1:
        output_title_latex = r'${:.2e}'.format(output_dict['loss'][-1]).replace('e','\\times 10^{')+'}$'
        plt.title('Final loss: '+output_title_latex) 
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.minorticks_on()
    plt.yscale('log')

    # Another axis with the lr:
    ax2 = plt.gca().twinx()
    ax2.plot(output_dict['lr'], 'k--', alpha=0.5) 
    ax2.set_yscale('log')
    ax2.set_ylabel('Learning rate')
        
    return plt.gcf()





def Trainer_gpu_full(nfmodel, wfamodel, coordinates, niter=1000, lrinit=1e-3, batchcoord=2000, trainBV=True, guesshelp=None, guess_regu=0.0, device='cuda',patience=50, normgrad = True, noise=0.0):
    """ 
    Train the neural field model using the WFA model as a loss function.
    """

    if device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
        if device_name != 'cpu':
            print('Using device:',device_name)
    else:
        device = "cpu"

    from tqdm import tqdm, trange

    # Move model and coordinates to GPU if available
    nfmodel = nfmodel.to(device)
    coordinates = torch.tensor(coordinates, dtype=torch.float32).to(device)
    if guesshelp is not None:
        guesshelp = torch.tensor(guesshelp, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.Adam(nfmodel.parameters(), lr=lrinit)
    wholedataset = np.arange(coordinates.shape[0])
    if batchcoord < 0:
        batchcoord = coordinates.shape[0]

    # Scheduler so learning rate decreases after no improvement in loss:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, verbose=True)

    nbatch = int(np.ceil(coordinates.shape[0]/batchcoord))
    print('Number of batches per epoch:',nbatch)

    loss_list = []
    lr_list = []
    t = trange(niter, leave=True)
    for loop in t:
        np.random.shuffle(wholedataset)
        # Split the dataset in batches:
        for batch in range(nbatch):
            optimizer.zero_grad()
            # Forward pass:
            out = nfmodel(coordinates[wholedataset[batch*batchcoord:(batch+1)*batchcoord], :])
            
            if trainBV:
                loss = wfamodel.optimizeBlos(params=out, index=wholedataset[batch*batchcoord:(batch+1)*batchcoord],noise=noise)
            else:
                loss = wfamodel.optimizeBQU(params=out, index=wholedataset[batch*batchcoord:(batch+1)*batchcoord],noise=noise)
                
            loss.backward()
            
            # Add gradient norm:
            if normgrad:
                for parameters in nfmodel.parameters():
                    parameters.grad = parameters.grad / (torch.mean(torch.abs(parameters.grad),dim=0) + 1e-9)
                    
            optimizer.step()
            t.set_postfix_str('loss = {:.2e}'.format(loss.item()))
            
            
            # Save loss:
            loss_list.append(loss.item())
            lr_list.append(optimizer.param_groups[0]['lr'])

        scheduler.step(loss)

        # If learning rate is too small, stop:
        if optimizer.param_groups[0]['lr'] < 1e-5:
            print('Learning rate too small. Stopping.')
            break

    output_dict = {'loss': loss_list, 'lr': lr_list}
    
    # Move model back to CPU:
    nfmodel = nfmodel.to('cpu')
    
    return output_dict