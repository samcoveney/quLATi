
import numpy as np
from scipy import linalg

#{{{ negative loglikelihood
#@partial(jit, static_argnums=(0, 2))
def LLH(self, guess, fixed_nugget):
    """Return the negative loglikelihood.
    
       Arguments:
       guess -- log(hyperparameter values) for training
       fixed_nugget -- value for fixed nugget 
    
    """

    # set the hyperparameters
    if guess.shape[0] > 2: # training on nugget
        HP = np.exp(guess[0:-1])
        nugget = np.exp(guess[-1])
    else: # fixed nugget
        HP = np.exp(guess)
        nugget = fixed_nugget

    # spectral density
    SD = self.spectralDensity( HP[0] ) # NOTE: multiply SD by signal variance
    SD = HP[1] * SD

    # set outputs and inputs
    y = self.y
    V = np.sqrt(SD)*self.V[self.vertex] # NOTE: absorb SD(eigenvalues) into the eigenvectors

    # define Q := phi phi^T + D
    # D is diagonal matrix of observation variances + nugget, but can represent as a vector only for ease

    # form noise matrix (diagonal, store as vector)
    D = self.yerr**2 + nugget
    invD = 1.0/D

    # form Z (different from Solin paper)
    ones = np.eye(SD.shape[0])
    Z = ones + (V.T).dot(invD[:,None]*V) # invD is diagonal, faster than diag(invD).dot(V)

    try:

        # attempt cholesky factorization
        # ------------------------------
        L = np.linalg.cholesky(Z)
        cho_success = True

        # log |Q| = log |Z| + log |D|
        # ---------------------------
        logDetZ = 2.0*np.sum(np.log(np.diag(L))) 
        log_Q = logDetZ + np.sum(np.log(D))  # SD absorbed into eigenvectors

        # y^T Q^-1 y
        # ----------
        tmp = V.T.dot(invD*y)
        tmp = linalg.solve_triangular(L, tmp, lower = True) # NOTE: non-JAX
        tmp = tmp.T.dot(tmp)
        yQy = np.dot(y, invD*y) - tmp

        # LLH = 1/2 log|Q| + 1/2 y^T Q^-1 y + n/2 log(2 pi)
        # -------------------------------------------------
        n_log_2pi = y.shape[0]*np.log(2*np.pi)
        llh = 0.5 * (log_Q + yQy + n_log_2pi)

        return llh

    except:

        return np.nan

#}}}


