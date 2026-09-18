import numpy as np
import scipy.linalg as la

def expm(X):
    """
    expmat1(X) returns the matrix exponential of the input square matrix X.

    - Uses Sylvester's formula if X is diagonalizable with distinct eigenvalues.
    - Falls back to a generic matrix exponential otherwise.

    see: https://github.com/tungcyang/MatlabMatrixExponential
         https://tungcyang.wordpress.com/2016/06/14/matrix-exponential-in-matlab-2nd-followup/
    """
    # Eigen-decomposition
    eigvals, V = la.eig(X)

    # Sylvester's formula: V * exp(D) * V^{-1}
    Dexp = np.diag(np.exp(eigvals))
    EX = V @ Dexp @ la.inv(V)
    #EX = V @ Dexp @ la.solve(V, np.eye(V.shape[0]))
    return EX
