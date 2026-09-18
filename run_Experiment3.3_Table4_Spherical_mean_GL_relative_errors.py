#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Erick Jorge Canales-Rodriguez, 2026

import numpy as np
import time
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from myelin_water_diffusionMRI_models import signal_radial_pgse_spectral_approach, build_K_reduced_from_full
from myelin_water_diffusionMRI_models import spherical_mean_cylinder_surface, spherical_mean_cylinder_surface_strang
from myelin_water_diffusionMRI_models import spherical_voronoi_weights

threadpool_limits(1, user_api="blas")

# ------------------------- Experimental parameters ---------------------------#
D            = 0.8
Bvalues      = np.array([3.0, 6.0, 20.0])
Nterms       = 10
Nsteps       = 10
Nodes        = np.array([5, 6, 7, 8, 20])
Nodes_ref    = 100  # Can be changed to 200 to compare the high-order GL results.
Nrepetitions = 1   # Repetitions for timing; use 1 for a single timed sweep.

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

# GL reference nodes and weights on [0, 1].
nodes_ref, weights_ref = np.polynomial.legendre.leggauss(Nodes_ref)
nodes_ref   = 0.5 * (nodes_ref + 1.0)
weights_ref = 0.5 * weights_ref

# ------------------------- Tables for all b-values ---------------------------#
# GL table: b, nodes, three MRAEs, two times per radius, two speed-ups.
Table_GL = np.zeros((Bvalues.shape[0] * Nodes.shape[0], 9))

# Separate comparison: b, number of directions, MRAE, Voronoi time per radius.
Table_Voronoi = np.zeros((Bvalues.shape[0], 4))

# Original comparison: b, nodes, spectral MRAE, Strang MRAE (Voronoi denominator).
Table_legacy = np.zeros((Bvalues.shape[0] * Nodes.shape[0], 4))

# ------------------------- Loop over b-values --------------------------------#
for ib in range(Bvalues.shape[0]):
    bval = Bvalues[ib]
    print(' ')
    print('b =', bval, 'ms/um^2')

    # --------------------- High-order GL references --------------------------#
    Signal_GL_ref        = np.zeros(radius_discrete.shape[0])
    Signal_GL_strang_ref = np.zeros(radius_discrete.shape[0])

    for j in range(radius_discrete.shape[0]):
        Signal_GL_ref[j] = spherical_mean_cylinder_surface(bval, D, radius_discrete[j], BigDelta6, smalldelta6, Nterms, K, nodes_ref, weights_ref)
        Signal_GL_strang_ref[j] = spherical_mean_cylinder_surface_strang(bval, D, radius_discrete[j], BigDelta6, smalldelta6, Q, lam, nodes_ref, weights_ref, Nsteps)
    #end

    # --------------------- Voronoi spherical mean ----------------------------#
    Signal_vec    = np.zeros((bvecs.shape[0], radius_discrete.shape[0]))
    Signal_Voronoi = np.zeros(radius_discrete.shape[0])
    times_voronoi  = np.zeros(Nrepetitions)

    # The first sweep is a warm-up and is not included in the reported time.
    for repetition in range(Nrepetitions + 1):
        start = time.perf_counter()
        for j in range(radius_discrete.shape[0]):
            for i in range(bvecs.shape[0]):
                Signal_vec[i, j] = signal_radial_pgse_spectral_approach(bval, D, radius_discrete[j], BigDelta6, smalldelta6, sin_alpha[i], Nterms, K) * np.exp(-bval * D * cos_alpha2[i])
            #end
            Smean_radius_j    = np.hstack([Signal_vec[:, j], Signal_vec[:, j]])
            Signal_Voronoi[j] = np.sum(voronoi_weights * Smean_radius_j)
        #end
        end = time.perf_counter()
        if repetition > 0:
            times_voronoi[repetition - 1] = (end - start) / radius_discrete.shape[0]
    #end
    time_voronoi = np.mean(times_voronoi)

    # --------------------- Spectral approach with GL -------------------------#
    Signal_anat = np.zeros((radius_discrete.shape[0], Nodes.shape[0]))
    time_GL     = np.zeros(Nodes.shape[0])

    for k in range(Nodes.shape[0]):
        nodes, weights = np.polynomial.legendre.leggauss(Nodes[k])
        nodes   = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        times   = np.zeros(Nrepetitions)

        for repetition in range(Nrepetitions + 1):
            start = time.perf_counter()
            for j in range(radius_discrete.shape[0]):
                Signal_anat[j, k] = spherical_mean_cylinder_surface(bval, D, radius_discrete[j], BigDelta6, smalldelta6, Nterms, K, nodes, weights)
            #end
            end = time.perf_counter()
            if repetition > 0:
                times[repetition - 1] = (end - start) / radius_discrete.shape[0]
        #end
        time_GL[k] = np.mean(times)
    #end

    # --------------------- Strang splitting with GL --------------------------#
    Signal_anat_strang = np.zeros((radius_discrete.shape[0], Nodes.shape[0]))
    time_strang        = np.zeros(Nodes.shape[0])

    for k in range(Nodes.shape[0]):
        nodes, weights = np.polynomial.legendre.leggauss(Nodes[k])
        nodes   = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        times   = np.zeros(Nrepetitions)

        for repetition in range(Nrepetitions + 1):
            start = time.perf_counter()
            for j in range(radius_discrete.shape[0]):
                Signal_anat_strang[j, k] = spherical_mean_cylinder_surface_strang(bval, D, radius_discrete[j], BigDelta6, smalldelta6, Q, lam, nodes, weights, Nsteps)
            #end
            end = time.perf_counter()
            if repetition > 0:
                times[repetition - 1] = (end - start) / radius_discrete.shape[0]
        #end
        time_strang[k] = np.mean(times)
    #end

    # --------------------- Compute errors and fill tables --------------------#
    rel_err           = np.zeros(Nodes.shape[0])
    rel_err_strang    = np.zeros(Nodes.shape[0])
    rel_err_strang_GL = np.zeros(Nodes.shape[0])

    for k in range(Nodes.shape[0]):
        # Spectral GL convergence against the high-order spectral reference.
        rel_err[k] = 100 * np.mean(np.abs(Signal_anat[:, k] - Signal_GL_ref) / Signal_GL_ref)

        # Total Strang + GL discrepancy against that same spectral reference.
        rel_err_strang[k] = 100 * np.mean(np.abs(Signal_anat_strang[:, k] - Signal_GL_ref) / Signal_GL_ref)

        # GL convergence alone for Strang: use its own high-order reference.
        rel_err_strang_GL[k] = 100 * np.mean(np.abs(Signal_anat_strang[:, k] - Signal_GL_strang_ref) / Signal_GL_strang_ref)

        row = ib * Nodes.shape[0] + k
        Table_GL[row, :] = [bval, Nodes[k], rel_err[k], rel_err_strang[k], rel_err_strang_GL[k],
                            time_GL[k], time_strang[k], time_voronoi / time_GL[k], time_voronoi / time_strang[k]]

        # Retain the original comparisons, with Voronoi in the denominator.
        rel_err_voronoi = 100 * np.mean(np.abs(Signal_anat[:, k] - Signal_Voronoi) / Signal_Voronoi)
        rel_err_strang_voronoi = 100 * np.mean(np.abs(Signal_anat_strang[:, k] - Signal_Voronoi) / Signal_Voronoi)
        Table_legacy[row, :] = [bval, Nodes[k], rel_err_voronoi, rel_err_strang_voronoi]
    #end

    # Finite-direction discrepancy, with the continuous GL reference in the denominator.
    rel_err_sampling = 100 * np.mean(np.abs(Signal_Voronoi - Signal_GL_ref) / Signal_GL_ref)
    Table_Voronoi[ib, :] = [bval, bvecs.shape[0], rel_err_sampling, time_voronoi]

    print('MRAE GL vs high-order GL (%):', rel_err)
    print('MRAE Strang + GL vs high-order spectral GL (%):', rel_err_strang)
    print('MRAE GL within Strang (%):', rel_err_strang_GL)
    print('MRAE Voronoi vs high-order spectral GL (%):', rel_err_sampling)
    print('Mean time per radius, Voronoi (s):', time_voronoi)
    print('Mean time per radius, GL (s):', time_GL)
    print('Mean time per radius, Strang + GL (s):', time_strang)
#end

# ------------------------- Save tables ---------------------------------------#
# All MRAEs are percentages averaged over the 50 radii.
# Times are means of the repeated sweeps, divided by the number of radii.
# Speed-ups use the spectral Voronoi time as the baseline.
# Re-running the script replaces these files.
header_GL = 'b_ms_per_um2,nodes,MRAE_GL_pct,MRAE_Strang_total_pct,MRAE_Strang_GL_pct,time_GL_s_per_radius,time_Strang_s_per_radius,speedup_GL,speedup_Strang'
header_Voronoi = 'b_ms_per_um2,directions,MRAE_Voronoi_vs_GLref_pct,time_Voronoi_s_per_radius'
header_legacy = 'b_ms_per_um2,nodes,MRAE_GL_vs_Voronoi_pct,MRAE_Strang_vs_Voronoi_pct'

np.savetxt('Table_IV_GL_convergence.csv', Table_GL, delimiter=',', header=header_GL, comments='', fmt='%.12g')
np.savetxt('Table_IV_GL_convergence.txt', Table_GL, delimiter='\t', header=header_GL.replace(',', '\t'), comments='', fmt='%.12g')

np.savetxt('Table_IV_GL_vs_Voronoi.csv', Table_Voronoi, delimiter=',', header=header_Voronoi, comments='', fmt='%.12g')
np.savetxt('Table_IV_GL_vs_Voronoi.txt', Table_Voronoi, delimiter='\t', header=header_Voronoi.replace(',', '\t'), comments='', fmt='%.12g')

np.savetxt('Table_IV_legacy_GL_vs_Voronoi.csv', Table_legacy, delimiter=',', header=header_legacy, comments='', fmt='%.12g')
np.savetxt('Table_IV_legacy_GL_vs_Voronoi.txt', Table_legacy, delimiter='\t', header=header_legacy.replace(',', '\t'), comments='', fmt='%.12g')
