# Exact analytical diffusion MRI signal for diffusion confined to a cylindrical surface using a spectral Laplacian formalism

<img src="Graphical_abstract.png" width="1024">

---

This repository contains the models, synthetic data, and scripts described in our manuscript (under revision, 2026):  
**"Exact analytical PGSE signal for diffusion confined to a cylindrical surface using a spectral Laplacian formalism"**  
*Erick J. Canales-Rodríguez, Chantal M.W. Tax, Juan Manuel Górriz, Derek K. Jones, Jean-Philippe Thiran, Jonathan Rafael-Patiño*

## Summary

The proposed framework provides an exact analytical expression for the diffusion MRI signal arising from diffusion confined to a cylindrical surface under finite rectangular pulsed-gradient spin-echo (PGSE) gradients. It develops computationally efficient strategies for repeated model evaluations. 
We solved the Bloch–Torrey equation using a spectral matrix formalism of the Laplace operator in the eigenbasis of the cylindrical surface. The resulting dMRI signal is expressed as a product of non-commuting matrix exponentials and is valid for arbitrary rectangular gradient durations and separations, without approximations to the diffusion propagator or spin phase distribution. We reduced a real spectral basis to reduce the dimensionality of the problem. We developed accelerated implementations based on Strang splitting and Gauss–Legendre quadrature for repeated signal evaluations and computation of the spherical mean. The analytical signal was validated against Monte Carlo diffusion simulations.

## Repository Structure 📖

- **`Data/`**  
  Includes synthetic datasets used for validation. To access the datasets, unzip the provided `Data.zip` file:  
  ```bash
  unzip Data.zip -d Data/
  
## Main Directory and Usage 🚀
It contains Python scripts to reproduce the figures and analyses presented in the manuscript. For example:

    python3 run_Fig2_Spherical_mean_signal_vs_Radius_Monte_Carlo_Signal_vs_Spectral_Approach_G500_D0.8_6bvalues.py
    
## Getting Started - dependencies 🔧
Before running the scripts, ensure the following requirements are met:
```
-Python 3.8+
-numpy
-matplotlib
-scipy

## Installation 🎁
Clone the repository and navigate to its directory:

    git clone https://github.com/ejcanalesr/exact-PGSE-dMRI-signal-cylindrical-surface.git
    cd exact-PGSE-dMRI-signal-cylindrical-surface
