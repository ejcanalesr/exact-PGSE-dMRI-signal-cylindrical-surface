#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Erick Jorge Canales-Rodriguez, 2026

import numpy as np

from threadpoolctl import threadpool_limits
threadpool_limits(1, user_api="blas")

from myelin_water_diffusionMRI_models import signal_radial_pgse_spectral_approach, build_K_reduced_from_full, SMT_signal_Gaussian_WidePulse
from myelin_water_diffusionMRI_models import spherical_voronoi_weights

import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.rc('text', usetex = True)

# Load bvecs
bvecs  = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/directions.txt')
bvecs = bvecs[1:]

# original b-vecs are on one hemisphere, projecting to the other hemisphere
bvecs_antipodal = bvecs / np.linalg.norm(bvecs, axis=1)[:, None]
bvecs_antipodal = np.vstack([bvecs_antipodal, -bvecs_antipodal])
voronoi_weights = spherical_voronoi_weights(bvecs_antipodal, normalize="1")
print("sum(w)  =", voronoi_weights.sum())                         # should be 1
print("min/max =", voronoi_weights.min(), voronoi_weights.max())  # should be positive

# ---------------------- Define experimental parameters ------------------------
D          = 0.8
Nterms     = 50 # For the series
K          = build_K_reduced_from_full(Nterms)

BigDelta   = 11 #ms
smalldelta = 3 #ms
# -------------------------Generate Signals -----------------------------------#
radius   = np.array([1.0, 2.0, 3.0])
Nb       = 201
b        = np.linspace(0, 40, Nb)

Signal_anat = np.zeros((Nb, radius.shape[0]))

start      = time.perf_counter()

for ib in range(Nb):
    #print(ib)
    bi = b[ib]
    Signal_vec = np.zeros((bvecs.shape[0], radius.shape[0]))

    for j in range(0, radius.shape[0]):
        for i in range(0, bvecs.shape[0]):
            cos_alpha2  = bvecs[i,2]**2
            sin_alpha   = np.sqrt(1 -cos_alpha2)
            Signal_vec[i,j] = signal_radial_pgse_spectral_approach(bi, D, radius[j], BigDelta, smalldelta, sin_alpha, Nterms, K) * np.exp(-bi * D * cos_alpha2)
        #end for
        Smean_radius_j = np.hstack([ Signal_vec[:,j], Signal_vec[:,j] ]) # duplicate the signal on the other hemisphere
        Signal_anat[ib,j] = np.sum(voronoi_weights * Smean_radius_j) # Compute the Spherical Means
    #end for
#end

end = time.perf_counter()
print(f"Elapsed time (analytical model (not optimized for speed)): {end - start:.6f} seconds")

# Gaussian Phase Approximation (GPA) model
Signal_anat_GPA = np.zeros((Nb, radius.shape[0]))
start      = time.perf_counter()
for j in range(0, radius.shape[0]):
    Signal_anat_GPA[:,j] = SMT_signal_Gaussian_WidePulse(D, b, radius[j], BigDelta, smalldelta)
#end
end = time.perf_counter()
print(f"Elapsed time (GPA model): {end - start:.6f} seconds")

# ------------------------ Plot results ----------------------------------------
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ax1 = plt.figure(figsize=(7.5,5))

plt.plot(b, Signal_anat[:,0], label=r"$radius$ = " + str(round(radius[0],1)) +  " $\mu m$: Spectral Approach", color=colors[0])
plt.plot(b, Signal_anat[:,1], label=r"$radius$ = " + str(round(radius[1],1)) +  " $\mu m$: Spectral Approach", color=colors[1])
plt.plot(b, Signal_anat[:,2], label=r"$radius$ = " + str(round(radius[2],1)) +  " $\mu m$: Spectral Approach", color=colors[2])

plt.plot(b, Signal_anat_GPA[:,0], label=r"GPA", color=colors[0], linestyle='-.')
plt.plot(b, Signal_anat_GPA[:,1], label=r"GPA", color=colors[1], linestyle='-.')
plt.plot(b, Signal_anat_GPA[:,2], label=r"GPA", color=colors[2], linestyle='-.')

ax = plt.gca()
ax.legend(fontsize=12, ncol=2, handleheight=0, labelspacing=0.75, frameon=False, title=r"$\Delta$ = " + str(round(BigDelta,3)) + " ms, $\delta$ = " + str(round(smalldelta,3)) + " ms, " + "$D$ = " + str(D) + " $\mu m^2/ms$", title_fontsize=12)

ax.set_yscale('log')
ax.set_ylim([1e-2, 1.1])

plt.xticks(fontsize=11.5)
plt.yticks(fontsize=11.5)
plt.ylabel(r"Log of Normalized Signal Amplitude", fontsize=14)
plt.xlabel(r"$b$" + " $(ms/ \mu m^2)$", fontsize=14)
plt.title(r"Spherical Mean dMRI Signal", fontsize=14)

plt.savefig('Fig1_Diffraction_peaks_Delta_' + str(round(BigDelta,2)) + '_delta_' + str(round(smalldelta,2)) + '_D_'  + str(D) + '.png', bbox_inches='tight', dpi=600)
plt.show()
