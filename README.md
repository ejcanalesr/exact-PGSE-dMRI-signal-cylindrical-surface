# Exact analytical diffusion MRI signal for diffusion confined to a cylindrical surface using a spectral Laplacian formalism

<img src="Graphical_abstract.png" width="1024">

---

This repository contains the models, synthetic data, and scripts described in our manuscript (under revision, 2026):  
**"Exact analytical PGSE signal for diffusion confined to a cylindrical surface using a spectral Laplacian formalism"**  
*Erick J. Canales-Rodríguez, Chantal M.W. Tax, Juan Manuel Górriz, Derek K. Jones, Jean-Philippe Thiran, Jonathan Rafael-Patiño*

## Summary

The proposed framework provides an exact analytical expression for the diffusion MRI signal arising from diffusion confined to a cylindrical surface under finite rectangular pulsed-gradient spin-echo (PGSE) gradients. It develops computationally efficient strategies for repeated model evaluations. 
We solved the Bloch–Torrey equation using a spectral matrix formalism of the Laplace operator in the eigenbasis of the cylindrical surface. The resulting dMRI signal is expressed as a product of non-commuting matrix exponentials and is valid for arbitrary rectangular gradient durations and separations, without approximations to the diffusion propagator or spin phase distribution. We reduced a real spectral basis to reduce the dimensionality of the problem. We developed accelerated implementations based on Strang splitting and Gauss–Legendre quadrature for repeated signal evaluations and computation of the spherical mean. The analytical signal was validated against Monte Carlo diffusion simulations.
