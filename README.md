# Spectropolarimetric inversions using neural fields under the weak-field approximation

This repository contains the code used to generate the results of the paper "Exploring spectropolarimetric inversions using neural fields: solar chromospheric magnetic field under the weak-field approximation" ([arXiv:2106.16007](https://arxiv.org/abs/2106.16007)).

![example](docs/sketch_wINRWFA_large.png?raw=true "")
**Figure 1** — Sketch of the neural field inversion of the magnetic field vector under the weak-field approximation. The neural field is a continuous representation of the magnetic field vector over the spatial and temporal domain. 

![example](docs/transverse_comparison.png?raw=true "")
**Figure 2** — Comparison of the transverse component of the magnetic field vector inferred from a simulated observation at Ca II 8542A using the pixel-wise weak-field approximation (WFA) inversion (middle panel) and the neural field inversion (right panel).


## Abstract
Full-Stokes polarimetric datasets, originating from slit-spectrograph or narrow-band filtergrams, are routinely acquired nowadays. The data rate is increasing with the advent of bi-dimensional spectropolarimeters and observing techniques that allow long-time sequences of high-quality observations. There is a clear need to go beyond the traditional pixel-by-pixel strategy in spectropolarimetric inversions by exploiting the spatiotemporal coherence of the inferred physical quantities that contain valuable information about the conditions of the solar atmosphere. We explore the potential of neural networks as a continuous representation of the physical quantities over time and space (also known as neural fields), for spectropolarimetric inversions. We have implemented and tested a neural field to perform one of the simplest forms of spectropolarimetric inversions, the inference of the magnetic field vector under the weak-field approximation (WFA). By using a neural field to describe the magnetic field vector, we can regularize the solution in the spatial and temporal domain by assuming that the physical quantities are continuous functions of the coordinates. This technique can be trivially generalized to account for more complex inversion methods. We have tested the performance of the neural field to describe the magnetic field of a realistic 3D magnetohydrodynamic (MHD) simulation. We have also tested the neural field as a magnetic field inference tool (approach also known as physics-informed neural networks) using the WFA as our radiative transfer model. We investigated the results in synthetic and real observations of the Ca II 8542 A line. We also explored the impact of other explicit regularizations, such as using the information of an extrapolated magnetic field, or the orientation of the chromospheric fibrils. Compared to the traditional pixel-by-pixel inversion, the neural field approach improves the fidelity of the reconstruction of the magnetic field vector, especially the transverse component. This implicit regularization is a way of increasing the effective signal-to-noise of the observations. Although it is slower than the pixel-wise WFA estimation, this approach shows a promising potential for depth-stratified inversions, by reducing the number of free parameters and inducing spatio-temporal constraints in the solution.