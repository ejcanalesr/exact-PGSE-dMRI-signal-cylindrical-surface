#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Erick Jorge Canales-Rodriguez, 2026

import numpy as np
import time
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from myelin_water_diffusionMRI_models import signal_radial_pgse_spectral_approach, build_K_reduced_from_full, signal_strang_diagK_fast_numba
from myelin_water_diffusionMRI_models import spherical_voronoi_weights

threadpool_limits(1, user_api="blas")

# ------------------------- Experimental parameters ---------------------------#
D            = 0.8
Bvalues      = np.array([3.0, 6.0, 20.0])
Nterms       = 10
Nsteps       = np.array([5, 10, 20, 40, 80])  # Strang steps per pulse.
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

K      = build_K_reduced_from_full(Nterms)
lam, Q = eigh(K)
Q      = np.ascontiguousarray(Q.real)

# ------------------------- Table for all b-values ----------------------------#
# Columns: b, method, M, steps, MRAE, time per radius, speed-up.
# One spectral reference row, followed by one row for each number of Strang steps.
Table_results = np.empty((Bvalues.shape[0] * (Nsteps.shape[0] + 1), 7), dtype=object)

# ------------------------- Loop over b-values --------------------------------#
for ib in range(Bvalues.shape[0]):
    bval = Bvalues[ib]
    print(' ')
    print('b =', bval, 'ms/um^2')

    # --------------------- Spectral reference (M=10) -------------------------#
    Signal_vec      = np.zeros((bvecs.shape[0], radius_discrete.shape[0]))
    Signal_anat_ref = np.zeros(radius_discrete.shape[0])
    times_reference = np.zeros(Nrepetitions)

    # The first sweep is a warm-up and is not included in the reported time.
    for repetition in range(Nrepetitions + 1):
        start = time.perf_counter()
        for j in range(radius_discrete.shape[0]):
            for i in range(bvecs.shape[0]):
                Signal_vec[i, j] = signal_radial_pgse_spectral_approach(bval, D, radius_discrete[j], BigDelta6, smalldelta6, sin_alpha[i], Nterms, K) * np.exp(-bval * D * cos_alpha2[i])
            #end
            Smean_radius_j     = np.hstack([Signal_vec[:, j], Signal_vec[:, j]])
            Signal_anat_ref[j] = np.sum(voronoi_weights * Smean_radius_j)
        #end
        end = time.perf_counter()
        if repetition > 0:
            times_reference[repetition - 1] = (end - start) / radius_discrete.shape[0]
    #end
    time_reference = np.mean(times_reference)

    row = ib * (Nsteps.shape[0] + 1)
    Table_results[row, :] = [bval, 'Spectral', Nterms, '-', 0.0, time_reference, 1.0]

    # --------------------- Strang splitting ----------------------------------#
    Signal_anat = np.zeros((radius_discrete.shape[0], Nsteps.shape[0]))
    time_strang = np.zeros(Nsteps.shape[0])
    rel_err     = np.zeros(Nsteps.shape[0])

    for k in range(Nsteps.shape[0]):
        Nsteps_k   = Nsteps[k]
        Signal_vec = np.zeros((bvecs.shape[0], radius_discrete.shape[0]))
        times      = np.zeros(Nrepetitions)

        # Discard the first sweep, including any Numba compilation.
        for repetition in range(Nrepetitions + 1):
            start = time.perf_counter()
            for j in range(radius_discrete.shape[0]):
                for i in range(bvecs.shape[0]):
                    Signal_vec[i, j] = signal_strang_diagK_fast_numba(bval, D, radius_discrete[j], BigDelta6, smalldelta6, sin_alpha[i], lam, Q, Nsteps_k) * np.exp(-bval * D * cos_alpha2[i])
                #end
                Smean_radius_j    = np.hstack([Signal_vec[:, j], Signal_vec[:, j]])
                Signal_anat[j, k] = np.sum(voronoi_weights * Smean_radius_j)
            #end
            end = time.perf_counter()
            if repetition > 0:
                times[repetition - 1] = (end - start) / radius_discrete.shape[0]
        #end
        time_strang[k] = np.mean(times)

        # Same error definition and reference as in the original script.
        Sref       = Signal_anat_ref
        S_k        = Signal_anat[:, k]
        rel_err[k] = 100 * np.mean(np.abs(S_k - Sref) / Sref)

        row = ib * (Nsteps.shape[0] + 1) + k + 1
        Table_results[row, :] = [bval, 'Strang', Nterms, Nsteps_k, rel_err[k],
                                 time_strang[k], time_reference / time_strang[k]]
    #end

    print('Strang steps per pulse:', Nsteps)
    print('MRAE Strang (%):', rel_err)
    print('Mean time per radius, spectral reference (s):', time_reference)
    print('Mean time per radius, Strang (s):', time_strang)
    print('Speed-up against the spectral reference:', time_reference / time_strang)
#end

# ------------------------- Save table ----------------------------------------#
# Both methods use the Voronoi-weighted mean over the same directions and M=10.
# Times are means of the repeated sweeps, divided by the number of radii.
# Warm-up/Numba compilation, matrix setup and Voronoi weights are not included.
# MRAE is a percentage averaged over the 50 radii; the spectral reference has MRAE=0.
# Re-running the script replaces these files.
header = 'b_ms_per_um2,method,M,steps_per_pulse,MRAE_pct,time_s_per_radius,speedup_vs_spectral'
formats = ['%.1f', '%s', '%s', '%s', '%.12g', '%.12g', '%.12g']

np.savetxt('Table_III_Strang_splitting.csv', Table_results, delimiter=',', header=header, comments='', fmt=formats)
np.savetxt('Table_III_Strang_splitting.txt', Table_results, delimiter='\t', header=header.replace(',', '\t'), comments='', fmt=formats)
