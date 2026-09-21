#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Erick Jorge Canales-Rodriguez, 2026

import numpy as np
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

bvecs_antipodal = bvecs / np.linalg.norm(bvecs, axis=1)[:, None]
bvecs_antipodal = np.vstack([bvecs_antipodal, -bvecs_antipodal])

voronoi_weights = spherical_voronoi_weights(bvecs_antipodal, normalize="1")
print("sum(w) =", voronoi_weights.sum())               # should be 1
print("min/max =", voronoi_weights.min(), voronoi_weights.max())     # should be positive

# ---------------------- Define experimental parameters ------------------------
D               = 0.8 # um^2/ms
Nterms_spectral = 50 # For the series
K = build_K_reduced_from_full(Nterms_spectral)

# ---------------------------- Load MC simulations -----------------------------
radius  = np.linspace(0.1, 5.0, 50)
#radius  = np.array([0.0001, 0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.003, 0.0035 , 0.004, 0.0045, 0.005])*1e3

radius_discrete = np.round(radius,1)

bvalues     = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/b_val_vector.txt')
BigDelta    = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/Delta_vector.txt')
smalldelta  = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/delta_vector.txt')
ramptime    = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/ramp_vector.txt')

ind_b1  = (bvalues > 0.95) & (bvalues < 1.05)
ind_b2  = (bvalues > 1.95) & (bvalues < 2.05)
ind_b3  = (bvalues > 2.95) & (bvalues < 3.05)
ind_b4 =  (bvalues > 3.95) & (bvalues < 4.05)
ind_b5 =  (bvalues > 4.95) & (bvalues < 5.05)
ind_b6 =  (bvalues > 5.95) & (bvalues < 6.05)

grad_b1 = bvalues[ind_b1]
grad_b2 = bvalues[ind_b2]
grad_b3 = bvalues[ind_b3]
grad_b4 = bvalues[ind_b4]
grad_b5 = bvalues[ind_b5]
grad_b6 = bvalues[ind_b6]

BigDelta1 = BigDelta[ind_b1][0]# ms
BigDelta2 = BigDelta[ind_b2][0]# ms
BigDelta3 = BigDelta[ind_b3][0]# ms
BigDelta4 = BigDelta[ind_b4][0]# ms
BigDelta5 = BigDelta[ind_b5][0]# ms
BigDelta6 = BigDelta[ind_b6][0]# ms

smalldelta1 = smalldelta[ind_b1][0]# ms
smalldelta2 = smalldelta[ind_b2][0]# ms
smalldelta3 = smalldelta[ind_b3][0]# ms
smalldelta4 = smalldelta[ind_b4][0]# ms
smalldelta5 = smalldelta[ind_b5][0]# ms
smalldelta6 = smalldelta[ind_b6][0]# ms

E1 = ramptime[ind_b1][0]# ms
E2 = ramptime[ind_b2][0]# ms
E3 = ramptime[ind_b3][0]# ms
E4 = ramptime[ind_b4][0]# ms
E5 = ramptime[ind_b5][0]# ms
E6 = ramptime[ind_b6][0]# ms

b0      = bvalues == 0

MC = np.zeros((6, radius_discrete.shape[0]))
for i in range(radius_discrete.shape[0]):
    radius_i  = radius_discrete[i]
    Signal_MC_protocol  = np.loadtxt('Data/myelin_simulation_ouput_6Shells_bvals_1000_6000/myelin_' + str(radius_i) + '0_um_D_0.8_2D_1_75k_15k_remaked_DWI.txt')
    S0_protocol         = np.mean(Signal_MC_protocol[b0])

    # Robust mean: using the area/weight of voronoi cells on the sphere
    Signal_MC_protocol_b1 = np.hstack([Signal_MC_protocol[ind_b1], Signal_MC_protocol[ind_b1]])/S0_protocol
    SMT_protocol_b1 = np.sum(voronoi_weights * Signal_MC_protocol_b1)

    Signal_MC_protocol_b2 = np.hstack([Signal_MC_protocol[ind_b2], Signal_MC_protocol[ind_b2]])/S0_protocol
    SMT_protocol_b2 = np.sum(voronoi_weights * Signal_MC_protocol_b2)

    Signal_MC_protocol_b3 = np.hstack([Signal_MC_protocol[ind_b3], Signal_MC_protocol[ind_b3]])/S0_protocol
    SMT_protocol_b3 = np.sum(voronoi_weights * Signal_MC_protocol_b3)

    Signal_MC_protocol_b4 = np.hstack([Signal_MC_protocol[ind_b4], Signal_MC_protocol[ind_b4]])/S0_protocol
    SMT_protocol_b4 = np.sum(voronoi_weights * Signal_MC_protocol_b4)

    Signal_MC_protocol_b5 = np.hstack([Signal_MC_protocol[ind_b5], Signal_MC_protocol[ind_b5]])/S0_protocol
    SMT_protocol_b5 = np.sum(voronoi_weights * Signal_MC_protocol_b5)

    Signal_MC_protocol_b6 = np.hstack([Signal_MC_protocol[ind_b6], Signal_MC_protocol[ind_b6]])/S0_protocol
    SMT_protocol_b6 = np.sum(voronoi_weights * Signal_MC_protocol_b6)

    Signal_MC = np.array([SMT_protocol_b1, SMT_protocol_b2, SMT_protocol_b3, SMT_protocol_b4, SMT_protocol_b5, SMT_protocol_b6])
    MC[:,i] = Signal_MC
    # --------------------------------------------------------------------------
#end

S_MC_b1 = MC[0,:]
S_MC_b2 = MC[1,:]
S_MC_b3 = MC[2,:]
S_MC_b4 = MC[3,:]
S_MC_b5 = MC[4,:]
S_MC_b6 = MC[5,:]

# -------------------------Generate Signals -----------------------------------#
radius_cont = np.linspace(0.001, 5, 20)

b1     = 1.0 # ms/um2
start  = time.perf_counter()
Signal1_vec   = np.zeros((bvecs.shape[0], radius_cont.shape[0]))
Signal1_anat  = np.zeros((radius_cont.shape[0]))
Signal1_anat_GPA = np.zeros((radius_cont.shape[0]))
for j in range(0, radius_cont.shape[0]):
    for i in range(0, bvecs.shape[0]):
        cos_alpha2  = bvecs[i,2]**2
        sin_alpha   = np.sqrt(1 - cos_alpha2)
        Signal1_vec[i,j] = signal_radial_pgse_spectral_approach(b1, D, radius_cont[j], BigDelta1, smalldelta1, sin_alpha, Nterms_spectral, K) * np.exp(-b1 * D * cos_alpha2)
    #end for
    Smean_radius_j = np.hstack([ Signal1_vec[:,j], Signal1_vec[:,j] ]) # duplicate the signal on the other hemisphere
    Signal1_anat[j] = np.sum(voronoi_weights * Smean_radius_j) # Compute the Spherical Means
    # GPA model
    Signal1_anat_GPA[j] = SMT_signal_Gaussian_WidePulse(D, b1, radius_cont[j], BigDelta1, smalldelta1)
#end for
#end

end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")
#------------------------------------------------------------------------------#

b2 =  2.0 # ms/um2
start  = time.perf_counter()
Signal2_vec   = np.zeros((bvecs.shape[0], radius_cont.shape[0]))
Signal2_anat  = np.zeros((radius_cont.shape[0]))
Signal2_anat_GPA = np.zeros((radius_cont.shape[0]))
for j in range(0, radius_cont.shape[0]):
    for i in range(0, bvecs.shape[0]):
        cos_alpha2  = bvecs[i,2]**2
        sin_alpha   = np.sqrt(1 - cos_alpha2)
        Signal2_vec[i,j] = signal_radial_pgse_spectral_approach(b2, D, radius_cont[j], BigDelta2, smalldelta2, sin_alpha, Nterms_spectral, K) * np.exp(-b2 * D * cos_alpha2)
    #end for
    Smean_radius_j = np.hstack([ Signal2_vec[:,j], Signal2_vec[:,j] ]) # duplicate the signal on the other hemisphere
    Signal2_anat[j] = np.sum(voronoi_weights * Smean_radius_j) # Compute the Spherical Means
    # GPA model
    Signal2_anat_GPA[j] = SMT_signal_Gaussian_WidePulse(D, b2, radius_cont[j], BigDelta2, smalldelta2)
#end for
#end

end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")
#------------------------------------------------------------------------------#

b3 = 3.0 # ms/um2
start  = time.perf_counter()
Signal3_vec   = np.zeros((bvecs.shape[0], radius_cont.shape[0]))
Signal3_anat  = np.zeros((radius_cont.shape[0]))
Signal3_anat_GPA = np.zeros((radius_cont.shape[0]))
for j in range(0, radius_cont.shape[0]):
    for i in range(0, bvecs.shape[0]):
        cos_alpha2  = bvecs[i,2]**2
        sin_alpha   = np.sqrt(1 - cos_alpha2)
        Signal3_vec[i,j] = signal_radial_pgse_spectral_approach(b3, D, radius_cont[j], BigDelta3, smalldelta3, sin_alpha, Nterms_spectral, K) * np.exp(-b3 * D * cos_alpha2)
    #end for
    Smean_radius_j = np.hstack([ Signal3_vec[:,j], Signal3_vec[:,j] ]) # duplicate the signal on the other hemisphere
    Signal3_anat[j] = np.sum(voronoi_weights * Smean_radius_j) # Compute the Spherical Means
    # GPA model
    Signal3_anat_GPA[j] = SMT_signal_Gaussian_WidePulse(D, b3, radius_cont[j], BigDelta3, smalldelta3)
#end for
#end

end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")
#------------------------------------------------------------------------------#

b4 = 4.0 # ms/um2
start  = time.perf_counter()
Signal4_vec   = np.zeros((bvecs.shape[0], radius_cont.shape[0]))
Signal4_anat  = np.zeros((radius_cont.shape[0]))
Signal4_anat_GPA = np.zeros((radius_cont.shape[0]))
for j in range(0, radius_cont.shape[0]):
    for i in range(0, bvecs.shape[0]):
        cos_alpha2  = bvecs[i,2]**2
        sin_alpha   = np.sqrt(1 - cos_alpha2)
        Signal4_vec[i,j] = signal_radial_pgse_spectral_approach(b4, D, radius_cont[j], BigDelta4, smalldelta4, sin_alpha, Nterms_spectral, K) * np.exp(-b4 * D * cos_alpha2)
    #end for
    Smean_radius_j = np.hstack([ Signal4_vec[:,j], Signal4_vec[:,j] ]) # duplicate the signal on the other hemisphere
    Signal4_anat[j] = np.sum(voronoi_weights * Smean_radius_j) # Compute the Spherical Means
    # GPA model
    Signal4_anat_GPA[j] = SMT_signal_Gaussian_WidePulse(D, b4, radius_cont[j], BigDelta4, smalldelta4)
#end for
#end

end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")
#------------------------------------------------------------------------------#

b5 = 5.0 # ms/um2
start  = time.perf_counter()
Signal5_vec   = np.zeros((bvecs.shape[0], radius_cont.shape[0]))
Signal5_anat  = np.zeros((radius_cont.shape[0]))
Signal5_anat_GPA = np.zeros((radius_cont.shape[0]))
for j in range(0, radius_cont.shape[0]):
    for i in range(0, bvecs.shape[0]):
        cos_alpha2  = bvecs[i,2]**2
        sin_alpha   = np.sqrt(1 - cos_alpha2)
        Signal5_vec[i,j] = signal_radial_pgse_spectral_approach(b5, D, radius_cont[j], BigDelta5, smalldelta5, sin_alpha, Nterms_spectral, K) * np.exp(-b5 * D * cos_alpha2)
    #end for
    Smean_radius_j = np.hstack([ Signal5_vec[:,j], Signal5_vec[:,j] ]) # duplicate the signal on the other hemisphere
    Signal5_anat[j] = np.sum(voronoi_weights * Smean_radius_j) # Compute the Spherical Means
    # GPA model
    Signal5_anat_GPA[j] = SMT_signal_Gaussian_WidePulse(D, b5, radius_cont[j], BigDelta5, smalldelta5)
#end for
#end

end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")
#------------------------------------------------------------------------------#

b6 = 6.0 # ms/um2
start  = time.perf_counter()
Signal6_vec   = np.zeros((bvecs.shape[0], radius_cont.shape[0]))
Signal6_anat  = np.zeros((radius_cont.shape[0]))
Signal6_anat_GPA = np.zeros((radius_cont.shape[0]))
for j in range(0, radius_cont.shape[0]):
    for i in range(0, bvecs.shape[0]):
        cos_alpha2  = bvecs[i,2]**2
        sin_alpha   = np.sqrt(1 - cos_alpha2)
        Signal6_vec[i,j] = signal_radial_pgse_spectral_approach(b6, D, radius_cont[j], BigDelta6, smalldelta6, sin_alpha, Nterms_spectral, K) * np.exp(-b6 * D * cos_alpha2)
    #end for
    Smean_radius_j = np.hstack([ Signal6_vec[:,j], Signal6_vec[:,j] ]) # duplicate the signal on the other hemisphere
    Signal6_anat[j] = np.sum(voronoi_weights * Smean_radius_j) # Compute the Spherical Means
    # GPA model
    Signal6_anat_GPA[j] = SMT_signal_Gaussian_WidePulse(D, b6, radius_cont[j], BigDelta6, smalldelta6)
#end for
#end

end = time.perf_counter()
print(f"Elapsed time: {end - start:.6f} seconds")

# ------------------------ Plot results ----------------------------------------
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

plt.figure(figsize=(7.5,4.2))

legend = True
if legend == False:
    plt.plot(radius_cont, Signal1_anat, color=colors[0])
    plt.plot(radius_cont, Signal2_anat, color=colors[1])
    plt.plot(radius_cont, Signal3_anat, color=colors[2])
    plt.plot(radius_cont, Signal4_anat, color=colors[3])
    plt.plot(radius_cont, Signal5_anat, color=colors[4])
    plt.plot(radius_cont, Signal6_anat, color=colors[5])

    plt.plot(radius_cont, Signal1_anat_GPA, color=colors[0],linestyle='-.')
    plt.plot(radius_cont, Signal2_anat_GPA, color=colors[1],linestyle='-.')
    plt.plot(radius_cont, Signal3_anat_GPA, color=colors[2],linestyle='-.')
    plt.plot(radius_cont, Signal4_anat_GPA, color=colors[3],linestyle='-.')
    plt.plot(radius_cont, Signal5_anat_GPA, color=colors[4],linestyle='-.')
    plt.plot(radius_cont, Signal6_anat_GPA, color=colors[5],linestyle='-.')

    plt.plot(radius_discrete, S_MC_b1, marker='o', fillstyle='none', color=colors[0], linewidth=0, markersize=4)
    plt.plot(radius_discrete, S_MC_b2, marker='o', fillstyle='none', color=colors[1], linewidth=0, markersize=4)
    plt.plot(radius_discrete, S_MC_b3, marker='o', fillstyle='none', color=colors[2], linewidth=0, markersize=4)
    plt.plot(radius_discrete, S_MC_b4, marker='o', fillstyle='none', color=colors[3], linewidth=0, markersize=4)
    plt.plot(radius_discrete, S_MC_b5, marker='o', fillstyle='none', color=colors[4], linewidth=0, markersize=4)
    plt.plot(radius_discrete, S_MC_b6, marker='o', fillstyle='none', color=colors[5], linewidth=0, markersize=4)
elif legend == True:
    plt.plot(radius_cont, Signal1_anat, label=r"$b$ = " + str(round(b1,1)) +  " $ms/ \mu m^2$: Spectral Approach", color=colors[0],linewidth=1)
    plt.plot(radius_cont, Signal2_anat, label=r"$b$ = " + str(round(b2,1)) +  " $ms/ \mu m^2$: Spectral Approach", color=colors[1],linewidth=1)
    plt.plot(radius_cont, Signal3_anat, label=r"$b$ = " + str(round(b3,1)) +  " $ms/ \mu m^2$: Spectral Approach", color=colors[2],linewidth=1)
    plt.plot(radius_cont, Signal4_anat, label=r"$b$ = " + str(round(b4,1)) +  " $ms/ \mu m^2$: Spectral Approach", color=colors[3],linewidth=1)
    plt.plot(radius_cont, Signal5_anat, label=r"$b$ = " + str(round(b5,1)) +  " $ms/ \mu m^2$: Spectral Approach", color=colors[4],linewidth=1)
    plt.plot(radius_cont, Signal6_anat, label=r"$b$ = " + str(round(b6,1)) +  " $ms/ \mu m^2$: Spectral Approach", color=colors[5],linewidth=1)

    plt.plot(radius_cont, Signal1_anat_GPA, label=r"GPA", color=colors[0],linestyle='-.', linewidth=1)
    plt.plot(radius_cont, Signal2_anat_GPA, label=r"GPA", color=colors[1],linestyle='-.', linewidth=1)
    plt.plot(radius_cont, Signal3_anat_GPA, label=r"GPA", color=colors[2],linestyle='-.', linewidth=1)
    plt.plot(radius_cont, Signal4_anat_GPA, label=r"GPA", color=colors[3],linestyle='-.', linewidth=1)
    plt.plot(radius_cont, Signal5_anat_GPA, label=r"GPA", color=colors[4],linestyle='-.', linewidth=1)
    plt.plot(radius_cont, Signal6_anat_GPA, label=r"GPA", color=colors[5],linestyle='-.', linewidth=1)

    plt.plot(radius_discrete, S_MC_b1, label="MC signal", marker='o', fillstyle='none', color=colors[0], linewidth=0, markersize=3)
    plt.plot(radius_discrete, S_MC_b2, label="MC signal", marker='o', fillstyle='none', color=colors[1], linewidth=0, markersize=3)
    plt.plot(radius_discrete, S_MC_b3, label="MC signal", marker='o', fillstyle='none', color=colors[2], linewidth=0, markersize=3)
    plt.plot(radius_discrete, S_MC_b4, label="MC signal", marker='o', fillstyle='none', color=colors[3], linewidth=0, markersize=3)
    plt.plot(radius_discrete, S_MC_b5, label="MC signal", marker='o', fillstyle='none', color=colors[4], linewidth=0, markersize=3)
    plt.plot(radius_discrete, S_MC_b6, label="MC signal", marker='o', fillstyle='none', color=colors[5], linewidth=0, markersize=3)
#end

ax = plt.gca()
ax.set_ylim([0.0, 1])
ax.legend(fontsize=10, ncol=3, handleheight=1.5, labelspacing=0.5, bbox_to_anchor=(0.5, 1.3), loc="upper center", fancybox=True, shadow=True)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.yscale('log')
plt.ylabel(r"Spherical Mean Signal Amplitude", fontsize=14)
plt.xlabel(r"$radius$" + " $(\mu m)$", fontsize=14)

plt.xticks(np.linspace(0,5,11))
plt.savefig('Fig2_SMT_vs_Radius_D_' + str(D) +  '_G500_Spectral_Approach_vs_MonteCarlo_highb_all.png', bbox_inches='tight', dpi=600)
plt.show()
