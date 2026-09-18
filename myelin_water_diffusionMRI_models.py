#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Erick Jorge Canales-Rodriguez, 2024-2026


from __future__ import division

################################################################################
# For the matrix sizes used in this toolbox, it is better to use a single cpu/core/threads.
# This also allows for a fair comparisson across methods
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
################################################################################

# Myelin water diffusion models
import numpy as np

#from numpy.math import factorial
#from scipy.sparse.linalg import expm_multiply, expm
# Faster alternative:
from expm.expm_fast import expm

import scipy.sparse as sp

from scipy.spatial import SphericalVoronoi
from scipy.special import erf

from numba import njit

################################################################################
#  ***  Gaussian Phase Approximation (Analytical for rectangular pulses)   *** #
################################################################################

def SMT_signal_Gaussian_WidePulse(D, b, radius, BigDelta, smalldelta):
    # b can be a vector of b-values
    b     = np.asarray(b, dtype=float)
    t     =  BigDelta - smalldelta/3
    # ------------------------------------------------------------
    tc    = (radius**2)/D
    Dr    = (radius**2/(2*t)) * (tc/smalldelta)**2 * (  2*smalldelta/tc - 2 + 2*np.exp(-BigDelta/tc) +  2*np.exp(-smalldelta/tc) - np.exp(-(BigDelta + smalldelta)/tc) - np.exp(-(BigDelta - smalldelta)/tc))
    value = np.ones_like(b, dtype=float)
    mask  = b > 1e-6
    value[mask]  = np.sqrt(np.pi/4) * np.exp(-b[mask]*Dr) * erf(np.sqrt(b[mask]*(D-Dr))) / np.sqrt(b[mask]*(D-Dr)) # only evaluate for non-zeros b-values
    return value
#end

################################################################################
# *****  Radial Signal: exact analytical from Laplacian Spectral theory  ***** #
#        This is the function using a matrix with dimension (2M+1)x(2M+1)      #
#                         It is the slowest version                            #
################################################################################

def precompute_spectral_objects(M):
    m = np.arange(-M, M + 1, dtype=int)
    N = m.size
    off = np.ones(N - 1, dtype=float)
    #K = sp.diags([off, off], offsets=[-1, 1], shape=(N, N), format='csc')
    K = sp.diags([off, off], offsets=[-1, 1], shape=(N, N))
    idx0 = np.where(m == 0)[0][0]
    return m, K, idx0
#end

def build_generators(M, Ds, r, q, delta):
    """
    Generators during the pulse:
        H_q   = Ds * m^2 / r^2  -  i * (gamma*g * r / 2) * K
        H_-q  = Ds * m^2 / r^2  +  i * (gamma*g * r / 2) * K
    """
    m, K, idx0 = precompute_spectral_objects(M)
    m2 = m.astype(float)**2
    diag0 = Ds * m2 / r**2
    H0_diag = diag0.copy()

    beta = -1j * 0.5 * (q/delta) * r
    Hq  = sp.diags(diag0, offsets=0, format='csc') + beta * K
    Hmq = sp.diags(diag0, offsets=0, format='csc') - beta * K
    return Hq, Hmq, H0_diag, idx0
#end

def signal_radial_pgse_finite_delta_old_slow(b, Ds, r, Delta, delta, sin_alpha, M):
    t =  Delta - delta/3
    q =  np.sqrt(b/t) * sin_alpha

    Hq, Hmq, H0_diag, idx0 = build_generators(M, Ds, r, q, delta)
    N = Hq.shape[0]
    u0 = np.zeros(N, dtype=complex)
    u0[idx0] = 1.0

    # First pulse
    #psi1 = expm_multiply(-Hq * delta, u0)
    psi1 = expm(-Hq.todense() * delta) @ u0

    # Inter-pulse
    psi2 = psi1 * np.exp(-H0_diag * (Delta - delta))

    # Second pulse
    #psif = expm_multiply(-Hmq * delta, psi2)
    psif = expm(-Hmq.todense() * delta) @ psi2

    return psif[idx0].real
#end

################################################################################
# *****  Radial Signal: exact analytical from Laplacian Spectral theory  ***** #
#        This is the function using a matrix with dimension (M+1)x(M+1)        #
#   It is the reference function in the paper: using one matrix exponential    #
################################################################################

def build_T(M):
    # Construye la matriz de cambio de base (2M+1 x M+1)
    N_full = 2*M + 1
    N_red  = M + 1
    T = np.zeros((N_full, N_red), dtype=float)
    # m=0
    T[M, 0] = 1.0
    for m in range(1, M+1):
        T[M+m, m] = 1/np.sqrt(2)
        T[M-m, m] = 1/np.sqrt(2)
    #return sp.csr_matrix(T)
    return T

def build_K_reduced_from_full(M):
    m, K_full, idx0 = precompute_spectral_objects(M)
    T = build_T(M)
    #K_red = (T.T @ K_full @ T).tocsc()
    K_red = (T.T @ K_full @ T)
    return K_red

def build_generators_reduced(M, Ds, r, q, delta, K_red):
    m = np.arange(0, M + 1)
    diag0 = Ds * (m**2) / r**2
    #K = build_K_reduced_from_full(M)
    beta = -1j * 0.5 * (q/delta) * r
    Hq  = np.diag(diag0, 0) + beta * K_red
    Hmq = np.diag(diag0, 0) - beta * K_red
    return Hq, Hmq, diag0

def signal_radial_pgse_spectral_approach(b, Ds, r, Delta, delta, sin_alpha, M, K_red):
    t =  Delta - delta/3
    q =  np.sqrt(b/t) * sin_alpha

    Hq, Hmq, H0_diag = build_generators_reduced(M, Ds, r, q, delta, K_red)
    u0 = np.zeros(M + 1, dtype=complex)
    u0[0] = 1

    exp1 = expm(-Hq * delta)
    exp2 = np.conjugate(exp1)
    psi1 = exp1@u0
    psi2 = psi1 * np.exp(-H0_diag * (Delta - delta))
    #psif = expm(-Hmq * delta)@ psi2 # we avoid computing the second matrix exponential
    psif = exp2 @ psi2
    return psif[0].real

################################################################################
# *****  Radial Signal: exact analytical from Laplacian Spectral theory  ***** #
#        This is the function using a matrix with dimension (M+1)x(M+1)        #
#                 Second version using one matrix exponential                  #
#         Similar to the previous version, same computation time               #
################################################################################

def signal_radial_pgse_spectral_approach_v2(b, Ds, r, Delta, delta, sin_alpha, M, K_red):
    # Precompute these outside if we can call this many times
    m = np.arange(M + 1)
    m2 = (m * m).astype(float)

    # Effective diffusion time
    t = Delta - delta / 3.0
    q = np.sqrt(b / t) * sin_alpha

    # Build diagonal part: diag0 = Ds * m^2 / r^2
    diag0 = (Ds / (r * r)) * m2  # real vector length M+1

    # beta = -1j * 0.5 * (q/delta) * r  (purely imaginary)
    beta = (-1j) * 0.5 * (q / delta) * r

    # Hq = diag(diag0) + beta*K
    # We build it as a dense matrix; for 11x11 this is tiny.
    Hq = np.diag(diag0) + beta * K_red

    # Single matrix exponential
    E = expm(-Hq * delta)  # complex (M+1)x(M+1)

    # psi1 = E @ e0  is just first column
    psi1 = E[:, 0]

    # Free evolution under H0 (diagonal)
    psi2 = psi1 * np.exp(-diag0 * (Delta - delta))

    # expm(-Hmq*delta) = conj(E), so psif0 = first row of conj(E) dot psi2
    psif0 = np.vdot(E[0, :], psi2)  # vdot conjugates first arg -> conj(E[0,:])·psi2

    return psif0.real


################################################################################
#     ***    First implementation of the Strang splitting approach    ***      #
#            Not used in this work, we kept it for code clarity                #
################################################################################

def signal_radial_pgse_spectral_approach_strang(b, Ds, r, Delta, delta, sin_alpha, M, K_red, n = 4):

    t = Delta - delta/3.0
    q = np.sqrt(b/t) * sin_alpha

    m = np.arange(M + 1)
    diag0 = Ds * (m**2) / (r**2)             # real
    D_half = np.exp(-(delta/(2.0*n)) * diag0) # vector, exp of diagonal half-step

    beta = (-1j) * 0.5 * (q/delta) * r       # purely imaginary

    # One-step K exponentials for the pulse
    EK_minus = expm(-(delta/n) * (beta * K_red))  # for +q pulse
    #EK_plus  = expm(+(delta/n) * (beta * K_red))  # for -q pulse (beta flips sign)
    EK_plus = EK_minus.conj()

    u = np.zeros(M + 1, dtype=complex)
    u[0] = 1.0

    # First pulse: u <- (D_half * EK_minus * D_half)^n u
    for _ in range(n):
        u *= D_half
        u = EK_minus @ u
        u *= D_half

    # Free evolution under D for (Delta - delta)
    u *= np.exp(-diag0 * (Delta - delta))

    # Second pulse: u <- (D_half * EK_plus * D_half)^n u
    for _ in range(n):
        u *= D_half
        u = EK_plus @ u
        u *= D_half

    return u[0].real
# end

################################################################################
#     ***    Second version of the Strang splitting approach    ***            #
#            It is much faster than the first version, we avoid using expm     #
#            Not used in this work, we kept it for code clarity                #
################################################################################

def signal_radial_pgse_spectral_approach_strang_diagK(b, Ds, r, Delta, delta, sin_alpha, lam, Q, M, n=4):
    t = Delta - delta/3.0
    q = np.sqrt(b/t) * sin_alpha

    m = np.arange(M + 1)
    diag0 = Ds * (m**2) / (r**2)

    beta = (-1j) * 0.5 * (q/delta) * r

    # diagonal half-step of D
    D_half = np.exp(-(delta/(2.0*n)) * diag0)

    # exponent for K step
    h = delta / n

    # eigenvalue exponentials
    exp_lam_minus = np.exp(-h * beta * lam)
    exp_lam_plus  = np.conj(exp_lam_minus)   # exact in our case

    u = np.zeros(M + 1, dtype=complex)
    u[0] = 1.0

    # ----- first pulse ( +q ) -----
    for _ in range(n):
        u *= D_half
        u = Q @ (exp_lam_minus * (Q.T @ u))
        u *= D_half

    # ----- free evolution -----
    u *= np.exp(-diag0 * (Delta - delta))

    # ----- second pulse ( -q ) -----
    for _ in range(n):
        u *= D_half
        u = Q @ (exp_lam_plus * (Q.T @ u))
        u *= D_half

    return u[0].real

################################################################################
#     ***    Third version of the Strang splitting approach    ***             #
#            It is much faster than the previous version, we avoid using expm  #
#            Not used in this work, we kept it for code clarity                #
################################################################################


def _apply_Keig(u, Q, QT, exp_lam):
    # u -> Q diag(exp_lam) QT u
    tmp = QT @ u
    tmp *= exp_lam
    return Q @ tmp

def signal_strang_diagK_fast(b, Ds, r, Delta, delta, sin_alpha, lam, Q, M, n=4):
    QT = Q.T

    t = Delta - delta/3.0
    q = np.sqrt(b/t) * sin_alpha

    m = np.arange(M + 1)
    diag0 = Ds * (m*m) / (r*r)  # real

    # beta = -1j * c  (purely imaginary)
    c = 0.5 * (q/delta) * r
    h = delta / n

    # Diagonal factors: use fused form
    D_full = np.exp(-(h) * diag0)     # e^{-h D}
    D_half = np.sqrt(D_full)          # e^{-h D/2}

    # exp_lam_minus = exp(-h*beta*lam) = exp(i * (h*c*lam))
    phi = (h * c) * lam               # real
    exp_lam_minus = np.cos(phi) + 1j*np.sin(phi)
    exp_lam_plus  = np.conj(exp_lam_minus)

    u = np.zeros(M + 1, dtype=np.complex64)
    u[0] = 1.0

    # ---- first pulse (+q): u <- (D_half K D_half)^n u ----
    u *= D_half
    for _ in range(n-1):
        u = _apply_Keig(u, Q, QT, exp_lam_minus)
        u *= D_full
    u = _apply_Keig(u, Q, QT, exp_lam_minus)
    u *= D_half

    # ---- free evolution ----
    u *= np.exp(-diag0 * (Delta - delta))

    # ---- second pulse (-q) ----
    u *= D_half
    for _ in range(n-1):
        u = _apply_Keig(u, Q, QT, exp_lam_plus)
        u *= D_full
    u = _apply_Keig(u, Q, QT, exp_lam_plus)
    u *= D_half

    return u[0].real

################################################################################
#     ***    Final version of the Strang splitting approach    ***             #
#            It is much faster than the previous version, we avoid using expm  #
#            We modified the third version to accelerate the code using numba! #
#   *****             THIS IS THE VERSION used in this work              ****  #
################################################################################

@njit(cache=True, fastmath=True)
def _apply_Keig_inplace(u, Q, exp_lam, tmp, out):
    """
    out = Q * diag(exp_lam) * Q^T * u
    Q: (N,N) real (float64) or complex128 if you prefer
    u: (N,) complex128
    exp_lam: (N,) complex128
    tmp, out: (N,) complex128 buffers provided by caller
    """
    N = u.shape[0]

    # tmp = Q^T u
    for i in range(N):
        s = 0.0 + 0.0j
        for j in range(N):
            s += Q[j, i] * u[j]   # Q^T[i,j] = Q[j,i]
        tmp[i] = s

    # tmp *= exp_lam  (diag)
    for i in range(N):
        tmp[i] *= exp_lam[i]

    # out = Q tmp
    for i in range(N):
        s = 0.0 + 0.0j
        for j in range(N):
            s += Q[i, j] * tmp[j]
        out[i] = s


@njit(cache=True, fastmath=True)
def signal_strang_diagK_fast_numba(b, Ds, r, Delta, delta, sin_alpha, lam, Q, n):
    # Dimensions
    N = lam.shape[0]  # should be M+1

    t = Delta - delta / 3.0
    q = np.sqrt(b / t) * sin_alpha

    # diag0 = Ds * m^2 / r^2
    diag0 = np.empty(N, dtype=np.float64)
    rr = r * r
    for m in range(N):
        diag0[m] = Ds * (m * m) / rr

    # beta = -1j * c (pure imaginary)
    c = 0.5 * (q / delta) * r
    h = delta / n

    # Diagonal factors
    D_full = np.empty(N, dtype=np.float64)
    D_half = np.empty(N, dtype=np.float64)
    for i in range(N):
        D_full[i] = np.exp(-h * diag0[i])
        D_half[i] = np.sqrt(D_full[i])

    # exp_lam_minus = exp(i * h*c*lam)
    exp_lam_minus = np.empty(N, dtype=np.complex128)
    exp_lam_plus  = np.empty(N, dtype=np.complex128)
    for i in range(N):
        phi = (h * c) * lam[i]   # lam[i] real
        exp_lam_minus[i] = np.cos(phi) + 1j * np.sin(phi)
        exp_lam_plus[i]  = np.cos(phi) - 1j * np.sin(phi)  # conj

    # Buffers
    u   = np.zeros(N, dtype=np.complex128)
    u[0] = 1.0 + 0.0j
    tmp = np.empty(N, dtype=np.complex128)
    out = np.empty(N, dtype=np.complex128)

    # ---- first pulse (+q): u <- (D_half K D_half)^n u ----
    for i in range(N):
        u[i] *= D_half[i]

    for _ in range(n - 1):
        _apply_Keig_inplace(u, Q, exp_lam_minus, tmp, out)
        # swap u <- out
        for i in range(N):
            u[i] = out[i] * D_full[i]

    _apply_Keig_inplace(u, Q, exp_lam_minus, tmp, out)
    for i in range(N):
        u[i] = out[i] * D_half[i]

    # ---- free evolution ----
    free_t = Delta - delta
    for i in range(N):
        u[i] *= np.exp(-diag0[i] * free_t)

    # ---- second pulse (-q) ----
    for i in range(N):
        u[i] *= D_half[i]

    for _ in range(n - 1):
        _apply_Keig_inplace(u, Q, exp_lam_plus, tmp, out)
        for i in range(N):
            u[i] = out[i] * D_full[i]

    _apply_Keig_inplace(u, Q, exp_lam_plus, tmp, out)
    for i in range(N):
        u[i] = out[i] * D_half[i]

    return u[0].real


################################################################################
#       **************         Spherical Means         **************
#                      Strang splitting Approximation                          #
#                           Accelerated using numba                            #
################################################################################

@njit(cache=True, fastmath=True)
def spherical_mean_cylinder_surface_strang(b, Dpar, r, Delta, delta, Q, lam, mu, w, n_strang):
    """
    Spherical mean of the total signal:
        S(α) = S_perp(b sin^2 α) * exp(-b Dpar cos^2 α)
    averaged over gradient directions (axial symmetry).

    Uses Gauss–Legendre quadrature in μ = cos α on [0, 1]:
        S̄(b) = ∫_0^1 exp(-b Dpar μ^2) S_perp(b(1-μ^2)) dμ
    """

    # Accumulate quadrature
    Sm = 0.0
    for k in range(mu.shape[0]):
        mk = mu[k]
        wk = w[k]

        cos2 = mk * mk
        sin_alpha = np.sqrt(max(0.0, 1.0 - cos2))

        S_perp = signal_strang_diagK_fast_numba(b, Dpar, r, Delta, delta, sin_alpha, lam, Q, n_strang)

        S_par = np.exp(-b * Dpar * cos2)
        Sm += wk * (S_perp * S_par)

    return Sm
#end

################################################################################
#       **************         Spherical Means         **************
#                              Exact Signal                                    #
################################################################################

def spherical_mean_cylinder_surface(b, Dpar, r, Delta, delta, M, K_red, mu, w):
    """
    Spherical mean of the total signal:
        S(α) = S_perp(b sin^2 α) * exp(-b Dpar cos^2 α)
    averaged over gradient directions (axial symmetry).

    Uses Gauss–Legendre quadrature in μ = cos α on [0, 1]:
        S̄(b) = ∫_0^1 exp(-b Dpar μ^2) S_perp(b(1-μ^2)) dμ
    """

    # Accumulate quadrature
    Sm = 0.0
    for k in range(mu.shape[0]):
        mk = mu[k]
        wk = w[k]

        cos2 = mk * mk
        sin_alpha = np.sqrt(max(0.0, 1.0 - cos2))

        S_perp = signal_radial_pgse_spectral_approach(b, Dpar, r, Delta, delta, sin_alpha, M, K_red)

        S_par = np.exp(-b * Dpar * cos2)
        Sm += wk * (S_perp * S_par)

    return Sm
#end

################################################################################
#********* Compute spherical Voronoi weights for unit vectors on S^2. **********
################################################################################

def spherical_voronoi_weights(bvecs, normalize="1", eps=1e-12):
    """
    Compute spherical Voronoi weights for unit vectors on S^2.

    Parameters
    ----------
    bvecs : (N,3) array
        Unit vectors on the sphere.
    normalize : {"4pi","1"}
        If "4pi", sum(weights)=4*pi.
        If "1",   sum(weights)=1.
    eps : float
        Tolerance for normalization / assertions.

    Returns
    -------
    w : (N,) array
        Voronoi cell areas (weights).
    """
    bvecs = np.asarray(bvecs, dtype=np.float64)
    N = bvecs.shape[0]
    assert bvecs.shape[1] == 3

    # Ensure unit norm (important for SphericalVoronoi)
    norms = np.linalg.norm(bvecs, axis=1)
    bvecs = bvecs / norms[:, None]

    # Build spherical Voronoi on the unit sphere centered at origin
    sv = SphericalVoronoi(bvecs, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()  # ensures vertices of each region are ordered

    # Area of a spherical polygon on unit sphere from its vertices (ordered)
    def _spherical_polygon_area(verts):
        """
        Compute area of spherical polygon (unit sphere) given ordered vertices on S^2.
        Uses triangulation around the first vertex and spherical triangle areas.
        """
        # Robust clamp for dot products
        def _clamp(x):
            return max(-1.0, min(1.0, x))

        # Spherical triangle area via L'Huilier's formula.
        # Given triangle with vertices a,b,c on unit sphere.
        def _spherical_triangle_area(a, b, c):
            # Side lengths (central angles)
            ab = np.arccos(_clamp(np.dot(a, b)))
            bc = np.arccos(_clamp(np.dot(b, c)))
            ca = np.arccos(_clamp(np.dot(c, a)))
            s = 0.5 * (ab + bc + ca)
            # L'Huilier
            tan_term = np.tan(s/2) * np.tan((s-ab)/2) * np.tan((s-bc)/2) * np.tan((s-ca)/2)
            tan_term = max(tan_term, 0.0)
            E = 4.0 * np.arctan(np.sqrt(tan_term))  # spherical excess = area on unit sphere
            return E

        if verts.shape[0] < 3:
            return 0.0

        v0 = verts[0]
        area = 0.0
        for k in range(1, verts.shape[0] - 1):
            area += _spherical_triangle_area(v0, verts[k], verts[k+1])
        return area

    # Compute each cell area
    w = np.zeros(N, dtype=np.float64)
    for i in range(N):
        region_idx = sv.regions[i]
        verts = sv.vertices[region_idx]
        # Normalize vertices to unit sphere (numerical safety)
        verts = verts / np.linalg.norm(verts, axis=1)[:, None]
        w[i] = _spherical_polygon_area(verts)

    # Sanity: sum of areas should be 4*pi (unit sphere)
    total = w.sum()
    if not np.isfinite(total) or total < eps:
        raise RuntimeError("Voronoi areas invalid. Check bvecs (duplicates / degeneracy).")

    # Renormalize (SphericalVoronoi may have tiny numerical drift)
    w *= (4.0 * np.pi) / total

    if normalize == "1":
        w /= (4.0 * np.pi)
    elif normalize != "4pi":
        raise ValueError("normalize must be '4pi' or '1'")

    return w
#end
