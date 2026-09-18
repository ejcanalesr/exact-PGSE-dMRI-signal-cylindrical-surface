#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Erick Jorge Canales-Rodriguez, 2026

import numpy as np
import time
from threadpoolctl import threadpool_limits

from myelin_water_diffusionMRI_models import signal_radial_pgse_spectral_approach, build_K_reduced_from_full, SMT_signal_Gaussian_WidePulse
from myelin_water_diffusionMRI_models import spherical_voronoi_weights

threadpool_limits(1, user_api="blas")

# ------------------------- Experimental parameters ---------------------------#
D            = 0.8
Bvalues      = np.array([3.0, 6.0, 20.0])
Nterms       = np.array([3, 5, 10, 20, 50])  # The last order is the reference.
Nrepetitions = 1  # Repetitions for timing; use 1 for a single timed sweep.

radius_discrete = np.round(np.linspace(0.1, 5.0, 50), 1)

# ------------------------- Load acquisition data ------------------------------#
bvecs = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/directions.txt')
bvecs = bvecs[1:]  # Remove b0; the remaining rows are the 92 directions.

bvecs_antipodal = bvecs / np.linalg.norm(bvecs, axis=1)[:, None]
bvecs_antipodal = np.vstack([bvecs_antipodal, -bvecs_antipodal])
voronoi_weights = spherical_voronoi_weights(bvecs_antipodal, normalize="1")

bvalues    = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/b_val_vector.txt')
BigDelta   = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/Delta_vector.txt')
smalldelta = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/delta_vector.txt')

# PGSE diffusion times
BigDelta6   = 9.4455558 #ms
smalldelta6 = 4.6122224 #ms

cos_alpha2 = bvecs[:, 2]**2
sin_alpha  = np.sqrt(1 - cos_alpha2)

# ------------------------- Table for all b-values ----------------------------#
# Columns: b, method, M, MRAE, time per radius, speed-up against the reference.
# One row per spectral order (including the reference), plus one row for GPA.
Table_results = np.empty((Bvalues.shape[0] * (Nterms.shape[0] + 1), 6), dtype=object)

# ------------------------- Loop over b-values --------------------------------#
for ib in range(Bvalues.shape[0]):
    bval = Bvalues[ib]
    print(' ')
    print('b =', bval, 'ms/um^2')

    # --------------------- Spectral approach ---------------------------------#
    Signal_anat   = np.zeros((radius_discrete.shape[0], Nterms.shape[0]))
    time_spectral = np.zeros(Nterms.shape[0])

    for k in range(Nterms.shape[0]):
        Nterms_k = Nterms[k]
        K_k      = build_K_reduced_from_full(Nterms_k)
        times    = np.zeros(Nrepetitions)

        # The first sweep is a warm-up and is not included in the reported time.
        for repetition in range(Nrepetitions + 1):
            start = time.perf_counter()
            Signal_vec = np.zeros((bvecs.shape[0], radius_discrete.shape[0]))
            for j in range(radius_discrete.shape[0]):
                for i in range(bvecs.shape[0]):
                    Signal_vec[i, j] = signal_radial_pgse_spectral_approach(bval, D, radius_discrete[j], BigDelta6, smalldelta6, sin_alpha[i], Nterms_k, K_k) * np.exp(-bval * D * cos_alpha2[i])
                #end
                Smean_radius_j    = np.hstack([Signal_vec[:, j], Signal_vec[:, j]])
                Signal_anat[j, k] = np.sum(voronoi_weights * Smean_radius_j)
            #end
            end = time.perf_counter()
            if repetition > 0:
                times[repetition - 1] = (end - start) / radius_discrete.shape[0]
        #end
        time_spectral[k] = np.mean(times)
        print('M =', Nterms_k, ': mean time per radius (s) =', time_spectral[k])
    #end

    # --------------------- Gaussian Phase Approximation ----------------------#
    Signal_anat_GPA = np.zeros(radius_discrete.shape[0])
    times          = np.zeros(Nrepetitions)

    for repetition in range(Nrepetitions + 1):
        start = time.perf_counter()
        for j in range(radius_discrete.shape[0]):
            Signal_anat_GPA[j] = SMT_signal_Gaussian_WidePulse(D, bval, radius_discrete[j], BigDelta6, smalldelta6)
        #end
        end = time.perf_counter()
        if repetition > 0:
            times[repetition - 1] = (end - start) / radius_discrete.shape[0]
    #end
    time_GPA = np.mean(times)

    # --------------------- Compute errors and fill table ---------------------#
    # Same reference as the original script: Voronoi mean with M=50.
    Sref    = Signal_anat[:, -1]
    rel_err = np.zeros(Nterms.shape[0])

    for k in range(Nterms.shape[0]):
        S_k        = Signal_anat[:, k]
        rel_err[k] = 100 * np.mean(np.abs(S_k - Sref) / Sref)
        row        = ib * (Nterms.shape[0] + 1) + k
        Table_results[row, :] = [bval, 'Spectral', Nterms[k], rel_err[k],
                                 time_spectral[k], time_spectral[-1] / time_spectral[k]]
    #end

    # Keep the original GPA comparison: analytical spherical mean versus Sref.
    rel_err_GPA = 100 * np.mean(np.abs(Signal_anat_GPA - Sref) / Sref)
    row         = ib * (Nterms.shape[0] + 1) + Nterms.shape[0]
    Table_results[row, :] = [bval, 'GPA', '-', rel_err_GPA, time_GPA, time_spectral[-1] / time_GPA]

    print('Spectral orders:', Nterms)
    print('MRAE spectral (%):', rel_err)
    print('MRAE GPA (%):', rel_err_GPA)
    print('GPA: mean time per radius (s) =', time_GPA)
#end

# ------------------------- Save table ----------------------------------------#
# All MRAEs are percentages averaged over the 50 radii.
# Times are means of the repeated sweeps, divided by the number of radii.
# The M=50 reference row has MRAE=0 and speed-up=1 by definition.
# M='-' for GPA, since GPA has no spectral truncation order.
# Re-running the script replaces these files.
header = 'b_ms_per_um2,method,M,MRAE_pct,time_s_per_radius,speedup_vs_reference'
formats = ['%.1f', '%s', '%s', '%.12g', '%.12g', '%.12g']

np.savetxt('Table_II_truncation.csv', Table_results, delimiter=',', header=header, comments='', fmt=formats)
np.savetxt('Table_II_truncation.txt', Table_results, delimiter='\t', header=header.replace(',', '\t'), comments='', fmt=formats)
