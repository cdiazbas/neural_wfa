import torch
import torch.nn as nn
import numpy as np


# =================================================================
class temporal_mlp(nn.Module):
    """
    The temporal information is treated as a Film: Feature-wise Linear Modulation. It does not work as good as the MLP below.
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=1,
        num_resnet_blocks=3,
        num_layers_per_block=2,
        dim_hidden=50,
        activation=nn.GELU(),
        fourier_features=False,
        m_freqs=100,
        sigma=10,
        tune_beta=False,
    ):
        super(temporal_mlp, self).__init__()
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.fourier_features = fourier_features
        self.activation = activation
        self.tune_beta = tune_beta
        self.num_neurons = dim_hidden
        self.sigma = sigma

        # Let's use a Film network:
        self.film = mlp(
            dim_in=1,
            dim_out=self.num_resnet_blocks * 1,
            num_resnet_blocks=num_resnet_blocks,
            num_layers_per_block=num_layers_per_block,
            dim_hidden=self.num_neurons,
            activation=activation,
            fourier_features=True,
            m_freqs=m_freqs,
            sigma=sigma[0],
            tune_beta=False,
        )

        sigma_spatial = sigma[1:] if isinstance(sigma, list) else sigma
        self.beta0 = torch.ones(1, 1)

        dim_in = 2
        self.first = nn.Linear(dim_in, self.num_neurons)
        self.resblocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(self.num_neurons, self.num_neurons)
                        for _ in range(num_layers_per_block)
                    ]
                )
                for _ in range(num_resnet_blocks)
            ]
        )
        self.last = nn.Linear(self.num_neurons, dim_out)
        if fourier_features:
            self.first = nn.Linear(2 * m_freqs + dim_in, self.num_neurons)

            # sigma can be a float or a list of floats:
            n_param = len(sigma_spatial) if isinstance(sigma_spatial, list) else 1
            self.B = torch.from_numpy(
                np.random.normal(0.0, sigma_spatial, (m_freqs, n_param)).T
            ).float()

    def forward(self, x):
        # if x in device, move also B to device:
        if x.device != self.beta0.device:
            self.beta0 = self.beta0.to(x.device)
            # if self.B exist
            if hasattr(self, "B"):
                self.B = self.B.to(x.device)

        xx_spatial = x[:, 1:]
        xx_temporal = x[:, 0:1]

        beta = self.film(xx_temporal).reshape(
            xx_temporal.shape[0], self.num_resnet_blocks, 1
        )

        if self.fourier_features:
            cosx = torch.cos(torch.matmul(xx_spatial, self.B))
            sinx = torch.sin(torch.matmul(xx_spatial, self.B))
            x = torch.cat((cosx, sinx, xx_spatial), dim=1)
            x = self.activation(self.beta0 * self.first(x))

        else:
            x = self.activation(self.beta0 * self.first(xx_spatial))

        for i in range(self.num_resnet_blocks):
            z = self.activation(beta[:, i] + self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(beta[:, i] + self.resblocks[i][j](z))
            x = z + x
        out = self.last(x)
        return out


# ===================================================================
class mlp_temporal0(nn.Module):
    """
    DenseNet mlp where the output is f(x,y) + g(x,y,t)
    """

    def __init__(
        self,
        dim_in=3,
        dim_out=1,
        num_resnet_blocks=3,
        num_layers_per_block=2,
        dim_hidden=50,
        activation=nn.GELU(),
        fourier_features=False,
        m_freqs=100,
        sigma=10,
        tune_beta=False,
    ):
        super(mlp_temporal0, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_resnet_blocks = num_resnet_blocks
        self.num_layers_per_block = num_layers_per_block
        self.dim_hidden = dim_hidden
        self.activation = activation
        self.fourier_features = fourier_features
        self.m_freqs = m_freqs
        self.sigma = sigma
        self.tune_beta = tune_beta
        self.mlp = mlp(
            dim_in=dim_in,
            dim_out=dim_out,
            num_resnet_blocks=num_resnet_blocks,
            num_layers_per_block=num_layers_per_block,
            dim_hidden=dim_hidden,
            activation=activation,
            fourier_features=fourier_features,
            m_freqs=m_freqs,
            sigma=sigma,
            tune_beta=tune_beta,
        )

        self.fxyz = mlp(
            dim_in=2,
            dim_out=dim_out,
            num_resnet_blocks=num_resnet_blocks,
            num_layers_per_block=num_layers_per_block,
            dim_hidden=dim_hidden,
            activation=activation,
            fourier_features=fourier_features,
            m_freqs=m_freqs,
            sigma=sigma[1:],
            tune_beta=tune_beta,
        )

    def forward(self, x):
        x0 = x.clone()
        x1 = x.clone()[:, 1:]
        xt = x.clone()[:, 0:1]
        return xt * self.mlp(x0) + self.fxyz(x1)


# =================================================================
class mlp(nn.Module):
    """
    DenseNet with Fourier Features.
    """

    def __init__(
        self,
        dim_in=2,
        dim_out=1,
        num_resnet_blocks=3,
        num_layers_per_block=2,
        dim_hidden=50,
        activation=nn.GELU(),
        fourier_features=False,
        m_freqs=100,
        sigma=10,
        tune_beta=False,
    ):
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
            self.beta = nn.Parameter(
                torch.ones(self.num_resnet_blocks, self.num_layers_per_block)
            )
        else:
            self.beta0 = torch.ones(1, 1)
            self.beta = torch.ones(self.num_resnet_blocks, self.num_layers_per_block)

        self.first = nn.Linear(dim_in, num_neurons)
        self.resblocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(num_neurons, num_neurons)
                        for _ in range(num_layers_per_block)
                    ]
                )
                for _ in range(num_resnet_blocks)
            ]
        )
        self.last = nn.Linear(num_neurons, dim_out)
        if fourier_features:
            self.first = nn.Linear(2 * m_freqs + dim_in, num_neurons)
            # self.B = torch.from_numpy(np.random.normal(0.,sigma, size=(dim_in,m_freqs))).float() # Only valid for a single entry

            # sigma can be a float or a list of floats:
            n_param = len(sigma) if isinstance(sigma, list) else dim_in
            self.B = torch.from_numpy(
                np.random.normal(0.0, sigma, (m_freqs, n_param)).T
            ).float()
            # print('sigma:',sigma,'self.B.shape',self.B.shape)

    def forward(self, x):
        xx = x.clone()

        # if x in device, move also B to device:
        if x.device != self.beta0.device:
            self.beta0 = self.beta0.to(x.device)
            # if self.B exist
            if hasattr(self, "B"):
                self.B = self.B.to(x.device)

        if self.fourier_features:
            cosx = torch.cos(torch.matmul(x, self.B))
            sinx = torch.sin(torch.matmul(x, self.B))
            x = torch.cat((cosx, sinx, xx), dim=1)
            x = self.activation(self.beta0 * self.first(x))

        else:
            x = self.activation(self.beta0 * self.first(x))

        for i in range(self.num_resnet_blocks):
            # print(self.beta.shape)
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](x))

            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))
            x = z + x
        out = self.last(x)
        return out


# =================================================================
def optimal_batch(
    nfmodel,
    wfamodel,
    coordinates,
    minbatch=1000,
    maxbatch=1e9,
    lrinit=1e-3,
    trainBV=True,
    guesshelp=None,
    guess_regu=0.0,
    device="cuda",
    patience=50,
    normgrad=True,
    noise=0.0,
    reguB=None,
    reguB_weight=0.0,
    reguBazi=None,
    reguBazi_weight=0.0,
):
    """
    Run the the network with different batch sizes to find the optimal size for
    max speed.
    """

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        )
        if device_name != "cpu":
            print("Using device:", device_name)
    else:
        device = "cpu"

    import time
    import matplotlib.pyplot as plt

    # Move model and coordinates to GPU if available
    nfmodel = nfmodel.to(device)
    coordinates = coordinates.clone().detach().float().to(device)

    optimizer = torch.optim.Adam(nfmodel.parameters(), lr=lrinit)
    wholedataset = np.arange(coordinates.shape[0])

    output = {}
    output["factor_size"] = []
    output["timing_perbatch"] = []
    batch2test = np.array(
        10
        ** np.linspace(
            0.1 * np.log10(len(wholedataset)), 1 * np.log10(len(wholedataset)), 20
        ),
        dtype=np.int32,
    )
    batch2test = batch2test[batch2test > minbatch]
    batch2test = batch2test[batch2test < maxbatch]

    for ii in batch2test:
        time_start = time.time()
        batchcoord = ii
        np.random.shuffle(wholedataset)
        optimizer.zero_grad()  # reset gradients

        # Forward pass:
        out = nfmodel(coordinates[wholedataset[:batchcoord], :])

        if trainBV:
            loss = wfamodel.optimizeBlos(
                params=out, index=wholedataset[:batchcoord], noise=noise
            )

        else:  # train the transverse component
            loss = wfamodel.optimizeBQU(
                params=out, index=wholedataset[:batchcoord], noise=noise
            )

        loss.backward()  # calculate gradients

        # Add gradient norm:
        if normgrad:
            for parameters in nfmodel.parameters():
                parameters.grad = parameters.grad / (
                    torch.mean(torch.abs(parameters.grad), dim=0) + 1e-9
                )

        time_stop = time.time()

        newtime = time_stop - time_start
        output["factor_size"].append(ii)
        output["timing_perbatch"].append(newtime / (ii))

    optimal_batchsize = output["factor_size"][
        np.argmin(output["timing_perbatch"] / output["timing_perbatch"][0])
    ]
    plt.plot(
        output["factor_size"],
        output["timing_perbatch"] / output["timing_perbatch"][0],
        ".-",
    )
    plt.ylabel("Time per batch [s]")
    plt.xlabel("Batch size")
    plt.axvline(optimal_batchsize, ls="--", color="gray")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("Optimal batch size: " + str(optimal_batchsize))

    # Move model back to CPU:
    nfmodel = nfmodel.to("cpu")
    return


# =================================================================
def Trainer_gpu(
    nfmodel,
    wfamodel,
    coordinates,
    niter=1000,
    lrinit=1e-3,
    batchcoord=2000,
    trainBV=True,
    guesshelp=None,
    guess_regu=0.0,
    device="cuda",
    patience=50,
    normgrad=True,
    noise=0.0,
    reguB=None,
    reguB_weight=0.0,
    reguBazi=None,
    reguBazi_weight=0.0,
):
    """
    Train the neural field model using the WFA model as a loss function,
    including GPU support.

    Adds the option of guide the inference with regularizations.
    """

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        )
        if device_name != "cpu":
            print("Using device:", device_name)
    elif device.startswith("cuda:"):
        device_index = int(device.split(":")[1])
        device = torch.device(
            f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
        )
        device_name = (
            torch.cuda.get_device_name(device_index)
            if torch.cuda.is_available()
            else "cpu"
        )
        if device_name != "cpu":
            print("Using device:", device_name)
    else:
        device = "cpu"

    from tqdm import trange

    # Move model and coordinates to GPU if available
    nfmodel = nfmodel.to(device)
    coordinates = coordinates.clone().detach().float().to(device)
    if guesshelp is not None:
        guesshelp = guesshelp.clone().detach().float().to(device)

    optimizer = torch.optim.Adam(nfmodel.parameters(), lr=lrinit)
    wholedataset = np.arange(coordinates.shape[0])
    if batchcoord < 0:
        batchcoord = coordinates.shape[0]

    if reguB is not None:
        # Now we regularize with a field extrapolation:
        Bq_pot = (
            torch.from_numpy(reguB[0]).float().reshape(wfamodel.nx * wfamodel.ny)
            / wfamodel.QUnorm
        )
        Bu_pot = (
            torch.from_numpy(reguB[1]).float().reshape(wfamodel.nx * wfamodel.ny)
            / wfamodel.QUnorm
        )
        reguB = [
            torch.tensor(Bq_pot, dtype=torch.float32).to(device),
            torch.tensor(Bu_pot, dtype=torch.float32).to(device),
        ]
        reguB_weight = torch.tensor(reguB_weight, dtype=torch.float32).to(device)

    if reguBazi is not None:
        # Now we regularize with the fibril azimuthal field:
        external_azimuth_torch = torch.from_numpy(
            np.array(reguBazi.astype(np.float32))
        ).reshape(wfamodel.ny * wfamodel.nx)
        external_azimuth_torch = external_azimuth_torch.to(device)
        sin2phi_torch, cos2phi_torch = (
            torch.sin(2 * external_azimuth_torch),
            torch.cos(2 * external_azimuth_torch),
        )
        sin2phi_torch = sin2phi_torch.to(device)
        cos2phi_torch = cos2phi_torch.to(device)

    # Scheduler so learning rate decreases after no improvement in loss:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=patience, verbose=True
    )

    loss_list = []
    lr_list = []
    t = trange(niter, leave=True)
    for loop in t:
        np.random.shuffle(wholedataset)
        optimizer.zero_grad()  # reset gradients

        # Forward pass:
        out = nfmodel(coordinates[wholedataset[:batchcoord], :])

        if trainBV:
            loss = wfamodel.optimizeBlos(
                params=out, index=wholedataset[:batchcoord], noise=noise
            )

        else:  # train the transverse component
            loss = wfamodel.optimizeBQU(
                params=out, index=wholedataset[:batchcoord], noise=noise
            )

            if reguB is not None:
                # Regularize with the field extrapolation:
                loss += reguB_weight * torch.sum(
                    (out[:, 0] - reguB[0][wholedataset[:batchcoord]]) ** 2
                )
                loss += reguB_weight * torch.sum(
                    (out[:, 1] - reguB[1][wholedataset[:batchcoord]]) ** 2
                )

            if reguBazi is not None:
                # Calculate only the azimutal component of the Bfield:
                Bt = torch.sqrt(out[:, 0] * out[:, 0] + out[:, 1] * out[:, 1])
                sin2phiB = out[:, 1] / Bt
                cos2phiB = out[:, 0] / Bt

                # Regularize with the fibril azimuthal field:
                loss += reguBazi_weight * torch.mean(
                    (sin2phiB - sin2phi_torch[wholedataset[:batchcoord]]) ** 2
                )
                loss += reguBazi_weight * torch.mean(
                    (cos2phiB - cos2phi_torch[wholedataset[:batchcoord]]) ** 2
                )

        loss.backward()  # calculate gradients

        # Add gradient norm:
        if normgrad:
            for parameters in nfmodel.parameters():
                parameters.grad = parameters.grad / (
                    torch.mean(torch.abs(parameters.grad), dim=0) + 1e-9
                )

        optimizer.step()  # step forward
        t.set_postfix_str("loss = {:.2e}".format(loss.item()))

        # Save loss:
        loss_list.append(loss.item())
        lr_list.append(optimizer.param_groups[0]["lr"])

        if patience > 0:
            scheduler.step(loss)

        # If learning rate is too small, stop:
        if optimizer.param_groups[0]["lr"] < 1e-5:
            print("Learning rate too small. Stopping.")
            break

    output_dict = {"loss": loss_list, "lr": lr_list}

    # Move model back to CPU:
    nfmodel = nfmodel.to("cpu")
    torch.cuda.empty_cache()

    return output_dict


# =================================================================
def nume2string(num):
    """
    Convert number to scientific latex mode.
    """
    mantissa, exp = f"{num:.2e}".split("e")
    return mantissa + " \\times 10^{" + str(int(exp)) + "}"


# =================================================================
def plot_loss(output_dict):
    """
    Plot the loss and learning rate during training.
    """
    import matplotlib.pyplot as plt

    # Loss plot:
    plt.figure()
    plt.plot(output_dict["loss"], alpha=0.5)
    # Smoothing windows of the 10% of the total number of iterations:
    savgol_loss = output_dict["loss"]
    if len(output_dict["loss"]) > 10:
        window = int(len(output_dict["loss"]) / 10)
        from scipy.signal import savgol_filter

        savgol_loss = savgol_filter(output_dict["loss"], window, 3 if window > 3 else 1)
        plt.plot(savgol_loss, "C0-", alpha=0.8)
        plt.plot(savgol_loss, "k-", alpha=0.2)

    if len(output_dict["loss"]) > 1:
        output_title_latex = (
            r"${:.2e}".format(output_dict["loss"][-1]).replace("e", "\\times 10^{")
            + "}$"
        )
        plt.title("Final loss: " + output_title_latex)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.minorticks_on()
    plt.yscale("log")

    # Another axis with the lr:
    ax2 = plt.gca().twinx()
    ax2.plot(output_dict["lr"], "k--", alpha=0.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Learning rate")

    return plt.gcf()


# =================================================================
def Trainer_gpu_full(
    nfmodel,
    wfamodel,
    coordinates,
    niter=1000,
    lrinit=1e-3,
    batchcoord=2000,
    trainBV=True,
    guesshelp=None,
    guess_regu=0.0,
    device="cuda",
    patience=50,
    normgrad=True,
    noise=0.0,
    reguB=None,
    reguB_weight=0.0,
    reguBazi=None,
    reguBazi_weight=0.0,
):
    """
    Train the neural field model using the WFA model as a loss function.
    """

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        )
        if device_name != "cpu":
            print("Using device:", device_name)
    elif device.startswith("cuda:"):
        device_index = int(device.split(":")[1])
        device = torch.device(
            f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
        )
        device_name = (
            torch.cuda.get_device_name(device_index)
            if torch.cuda.is_available()
            else "cpu"
        )
        if device_name != "cpu":
            print("Using device:", device_name)
    else:
        device = "cpu"

    from tqdm import trange

    # Move model and coordinates to GPU if available
    nfmodel = nfmodel.to(device)
    coordinates = coordinates.clone().detach().float().to(device)
    if guesshelp is not None:
        guesshelp = guesshelp.clone().detach().float().to(device)

    optimizer = torch.optim.Adam(nfmodel.parameters(), lr=lrinit)
    wholedataset = np.arange(coordinates.shape[0])
    if batchcoord < 0:
        batchcoord = coordinates.shape[0]

    if reguB is not None:
        # Now we regularize with a field extrapolation:
        Bq_pot = (
            torch.from_numpy(reguB[0]).float().reshape(wfamodel.nx * wfamodel.ny)
            / wfamodel.QUnorm
        )
        Bu_pot = (
            torch.from_numpy(reguB[1]).float().reshape(wfamodel.nx * wfamodel.ny)
            / wfamodel.QUnorm
        )
        reguB = [
            torch.tensor(Bq_pot, dtype=torch.float32).to(device),
            torch.tensor(Bu_pot, dtype=torch.float32).to(device),
        ]
        reguB_weight = torch.tensor(reguB_weight, dtype=torch.float32).to(device)

    if reguBazi is not None:
        # Now we regularize with the fibril azimuthal field:
        external_azimuth_torch = torch.from_numpy(
            np.array(reguBazi.astype(np.float32))
        ).reshape(wfamodel.ny * wfamodel.nx)
        external_azimuth_torch = external_azimuth_torch.to(device)
        sin2phi_torch, cos2phi_torch = (
            torch.sin(2 * external_azimuth_torch),
            torch.cos(2 * external_azimuth_torch),
        )
        sin2phi_torch = sin2phi_torch.to(device)
        cos2phi_torch = cos2phi_torch.to(device)

    # Scheduler so learning rate decreases after no improvement in loss:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=patience, verbose=True
    )

    nbatch = int(np.ceil(coordinates.shape[0] / batchcoord))
    print("Number of batches per epoch:", nbatch)

    loss_list = []
    lr_list = []
    t = trange(niter, leave=True)
    for loop in t:
        np.random.shuffle(wholedataset)
        # Split the dataset in batches:
        for batch in range(nbatch):
            optimizer.zero_grad()
            # Forward pass:
            out = nfmodel(
                coordinates[
                    wholedataset[batch * batchcoord : (batch + 1) * batchcoord], :
                ]
            )

            if trainBV:
                loss = wfamodel.optimizeBlos(
                    params=out,
                    index=wholedataset[batch * batchcoord : (batch + 1) * batchcoord],
                    noise=noise,
                )
            else:
                loss = wfamodel.optimizeBQU(
                    params=out,
                    index=wholedataset[batch * batchcoord : (batch + 1) * batchcoord],
                    noise=noise,
                )

                if reguB is not None:
                    # Regularize with the field extrapolation:
                    loss += reguB_weight * torch.sum(
                        (out[:, 0] - reguB[0][wholedataset[:batchcoord]]) ** 2
                    )
                    loss += reguB_weight * torch.sum(
                        (out[:, 1] - reguB[1][wholedataset[:batchcoord]]) ** 2
                    )

                if reguBazi is not None:
                    # Calculate only the azimutal component of the Bfield:
                    Bt = torch.sqrt(out[:, 0] * out[:, 0] + out[:, 1] * out[:, 1])
                    sin2phiB = out[:, 1] / Bt
                    cos2phiB = out[:, 0] / Bt

                    # Regularize with the fibril azimuthal field:
                    loss += reguBazi_weight * torch.mean(
                        (sin2phiB - sin2phi_torch[wholedataset[:batchcoord]]) ** 2
                    )
                    loss += reguBazi_weight * torch.mean(
                        (cos2phiB - cos2phi_torch[wholedataset[:batchcoord]]) ** 2
                    )

            loss.backward()

            # Add gradient norm:
            if normgrad:
                for parameters in nfmodel.parameters():
                    parameters.grad = parameters.grad / (
                        torch.mean(torch.abs(parameters.grad), dim=0) + 1e-9
                    )

            optimizer.step()
            t.set_postfix_str("loss = {:.2e}".format(loss.item()))

            # Save loss:
            loss_list.append(loss.item())
            lr_list.append(optimizer.param_groups[0]["lr"])

        if patience > 0:
            scheduler.step(loss)

        # If learning rate is too small, stop:
        if optimizer.param_groups[0]["lr"] < 1e-5:
            print("Learning rate too small. Stopping.")
            break

    output_dict = {"loss": loss_list, "lr": lr_list}

    # Move model back to CPU:
    nfmodel = nfmodel.to("cpu")
    torch.cuda.empty_cache()

    return output_dict
