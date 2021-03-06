"""
   gpmi_mo.py

   Module implementing Gaussian Process Manifold Interpolation (GPMI) for MULTIPLE OUTPUTS, making use of the reduced-rank of the covariance matrix.

   Contains:
      class AbstractModel
      class Matern(AbstractModel)

   This model implements the Intrinsic Model of Coregionaliztion (Linear Model of Coregionalization with one latent GP).
   It makes use of the reduced-rank form of the covariance matrix to gain computational efficiency.

   Created: 23-Apr-2020
   Author:  Sam Coveney

"""

#{{{ module imports
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np

from scipy import linalg
from scipy.optimize import minimize, check_grad, approx_fprime
from scipy.special import gamma
from scipy.spatial.distance import pdist, cdist, squareform

import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#}}}


#{{{ utilities

#{{{ plot matrix
def imShow(A):
    plt.imshow(A, cmap = "jet"); plt.colorbar(); plt.show()
#}}}

#{{{ timeit
## use '@timeit' to decorate a function for timing
import time
def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        R = 10
        for r in range(R): # calls function 100 times
            result = f(*args, **kw)
        te = time.time()
        print('func: %r took: %2.4f sec for %d runs' % (f.__name__, te-ts, R) )
        return result
    return timed
#}}}

#{{{ scatter with histograms on axes
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.0025
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')
#}}}

#}}}

# faster kron function
kron = lambda a, b:  (a[:, None, :, None]*b[None, :, None, :]).reshape(a.shape[0]*b.shape[0],a.shape[1]*b.shape[1])

# abstract class utlizing the reduced-rank speedups
# ================================================

class AbstractModel(ABC):
    """ Gaussian Process Manifold Interpolation for multiple outputs, making use of reduced-rank expressions."""

    #{{{ decorators for checking methods arguments
    class Decorators(object):
        """Class of methods to use as decorators."""

        @classmethod
        def check_posterior_exists(self, method):
            """Checks that posterior function has been called so that sampling is possible."""
            @wraps(method)
            def _wrapper(self, *args, **kwargs):

                try:
                    self.post_mean; self.post_var;
                except AttributeError:
                    print("\n[ERROR] posteriorSamples: must call <posterior> before sampling.\n[return] None")
                    return None

                return method(self, *args, **kwargs)
            return _wrapper
    #}}}

    #{{{ everything in common: init, del, reset, and abstract methods
    def __init__(self, X, Tri, Q, V):
        """Initialize class with manifold data."""

        self.X = X
        self.Tri = Tri
        self.Q = Q
        self.V = V

        self.reset()

        super().__init__()


    def __del__(self):
        """Method to cleanup when we delete the instance of this class."""
        pass


    def reset(self):
        """Resets parameters and predictions."""

        #self.HP, self.nugget = None, None
        self.post_mean, self.post_var = None, None

        # set these values so that user does not have to call setMeanFunction
        self.meanFunction = 0


    @abstractmethod
    def spectralDensity(self, rho, w):
        """Spectral Density function."""
        pass


    @abstractmethod
    def setMeanFunction(self, w, rho):
        """Mean function m(x) so that y = m(x) + GP."""
        pass

    #}}}

    # {{{ data handling including scalings
    def set_data(self, y, vertex, yerr = None):
        """Set data for interpolation, scaled and centred."""

        # set mean and stdev of y
        self.y_mean, self.y_std = np.mean(y, axis = 0), np.std(y, axis = 0)
        self.y = self.scale(y, std = False)

        if yerr is None:
            self.yerr = np.zeros(y.shape)
        else:
            self.yerr = self.scale(yerr, std = True)

        self.vertex = vertex

        self.reset()


    def scale(self, y, std = False):
        """Scales and centres y data."""
        if std == False:
            return (y - self.y_mean) / self.y_std
        else:
            return (y / self.y_std)


    def unscale(self, y, std = False):
        """Unscales and uncenteres y data into original scale."""
        if std == False:
            return (y * self.y_std) + self.y_mean
        else:
            return (y * self.y_std)
    #}}}

    #{{{ untransform optimization guess into model parameters
    def HP_from_guess(self, guess):

        # set hyperparameter values
        self.HP = np.exp(guess[0])

        if True:
            # no transforms on L at all
            for i in range(0, self.L_idx[0].shape[0]):
                self.L[self.L_idx[0][i], self.L_idx[1][i]] = guess[i + 1] 
        else:
            # transform L carefully so that diagonals are positive
            for i in range(0, self.L_idx[0].shape[0]):
                if self.L_idx[0][i] == self.L_idx[1][i]: # exp transform on diagonals
                    self.L[self.L_idx[0][i], self.L_idx[1][i]] = np.exp(guess[i + 1]) 
                else: # no transform on off-diagonals
                    self.L[self.L_idx[0][i], self.L_idx[1][i]] = guess[i + 1] 

        #self.L[self.L_idx] = guess[1:-self.y.shape[1]]

        #print("L:", self.L)
        #print("A:", self.L.dot(self.L.T))

        if self.nugget_train:
            self.D = np.exp(guess[-self.y.shape[1]:])
        #else:
        #    self.D = self.nugget

        return
    #}}}

    #{{{ negative loglikelihood
    def LLH(self, guess):
        """Return the negative loglikelihood.
        
           Arguments:
           guess -- log(hyperparameter values) for training
        
        """

        #print("guess:", guess)
        self.HP_from_guess(guess)
        #print(self.L); input()

        # calculate spectral density
        self.SD = self.spectralDensity( self.HP ) # self.HP is just the lengthscale hyperparameter

        # set outputs and inputs
        y = self.y.T.flatten()
        V = np.sqrt(self.SD)*self.V[self.vertex] # NOTE: absorb SD(eigenvalues) into the eigenvectors

        # create inverse of A \kron K_spatial; L_TMP is the cholesky of matrix Z needed for an efficient inverse
        L_TMP = self.make_invSigma(V_precalc = V) # NOTE: L_TMP is the Cholesky of the small inverse needed for efficient invSigma calculation
        invK = self.invSigma
        invK_y = invK.dot(y)

        # calculating log | K |
        # ---------------------

        # calculate logDetK utilizing L_TMP
        delta = np.repeat(self.D, self.vertex.shape[0]) + self.yerr.T.flatten()**2
        logDetK = 2*np.sum(np.log(np.diag(L_TMP[0]))) + np.sum(np.log(delta))

        #{{{ test logDetK accuracy
        #print("logDetK:", logDetK)
        #K = self.kernelMatrix(V, V, nugget = True) # this is the full kernel Sigma := A \kron K(x,x'), although it's called 'K'
        #L_test = linalg.cho_factor(K)        
        #logDetK = 2.0*np.sum(np.log(np.diag(L_test[0])))        
        #print("logDetK check:", logDetK)
        #}}}

        # calculate loglikelihood
        # -----------------------
        llh = 0.5 * ( logDetK + ( y.T ).dot( invK_y ) + y.shape[0]*np.log(2*np.pi) )  # NOTE: I'm not entirely sure about the constant when we have derivatives

        return llh
        
    #}}}

    #{{{ optimize the hyperparameters
    def optimize(self, restarts, nugget = None, zero_off_diagonals = False):
        """Optimize the hyperparameters.
        
           Arguments:
           nugget -- value of the nugget, if None then train nugget, if array then fix nugget
           restart -- how many times to restart the optimizer.
           zero_off_diagonals -- initial guesses for off diagonals of L (where A = LL^T) are zero (off diagonals are not fixed to zero though)

           NOTES:
           * initial guesses are picked at random from predefined ranges set in this function
        
        """

        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str(x)[:5] % x

        # initial guesses for hyperparameters
        # -----------------------------------

        hdr = "| Restart | "

        # lengthscale guess
        dguess = np.random.uniform(np.log(5), np.log(50), size = restarts).reshape([1,-1]).T
        hdr = hdr + " len " + " | "

        # between-outputs-correlation guess
        # ---------------------------------
        for i in range(0, self.L_idx[0].shape[0]):
            if zero_off_diagonals == True:
                if self.L_idx[0][i] == self.L_idx[1][i]: # diagonals
                    sg = np.random.uniform(1.0, 10.0, size = restarts).reshape([1,-1]).T
                else: # off diagonals
                    sg = np.zeros(shape = restarts).reshape([1,-1]).T
            else:
                sg = np.random.uniform(1.0, 10.0, size = restarts).reshape([1,-1]).T
            sguess = sg if i == 0 else np.hstack([sguess, sg])
            hdr = hdr + " L{:d}{:d} ".format(self.L_idx[0][i], self.L_idx[1][i]) + " | "
        print("L guesses:\n", sguess)

        # nugget options
        # --------------

        if nugget is None:  # nugget not supplied; we must train on nugget
            self.nugget_train = True

            # nugget guesses
            for i in range(0,self.y.shape[1]):
                ng = np.random.uniform(np.log(0.1), np.log(1.0), size = restarts).reshape([1,-1]).T
                nguess = ng if i == 0 else np.hstack([nguess, ng])
                hdr = hdr + " nu{:d} ".format(i) + " | "

            guess = np.hstack([dguess, sguess, nguess])

        else:
            self.nugget_train = False

            #self.nugget = np.abs(nugget)
            self.D = np.abs(nugget)

            hdr = hdr + " (fixed nuggets) "

            guess = np.hstack([dguess, sguess])


        # minimimize LLH
        # --------------
        print("Optimizing Hyperparameters...\n" + hdr)

        for gn, g in enumerate(guess):
            optFail = False
            try:
                bestStr = "   "
                res = minimize(self.LLH, g, method = 'Nelder-Mead') 
                #res = minimize(self.LLH, g, method = 'L-BFGS-B') 

                if np.isfinite(res.fun):
                    try:
                        if res.fun < bestRes.fun:
                            bestRes = res
                            bestStr = " * "
                    except:
                        bestRes = res
                        bestStr = " * "
                else:
                    bestStr = " ! "

                
                self.HP_from_guess(res.x)
                untransform_res = np.hstack([ self.HP, self.L[self.L_idx].flatten(), self.D ])


                print("|  {:02d}/{:02d} ".format(gn + 1, restarts),
                      "| %s" % ' | '.join(map(str, [fmt(i) for i in untransform_res])),
                      "| {:s} llh: {:.3f}".format(bestStr, -1.0*np.around(res.fun, decimals=4)))


            except TypeError as e:
                optFail = True

            except ValueError as e:
                optFail = True

            except np.linalg.LinAlgError as e:
                optFail = True

            if optFail: print("|  {:02d}/{:02d} ".format(gn + 1, restarts),
                              "| [ERROR]: Something went numerically wrong, optimization failed.")

        self.HP_from_guess(bestRes.x)
        print("Optimization complete.")


    #}}}

    #{{{ posterior distribution
    def posterior(self, pointwise = True):
        """Calculates GP posterior mean and variance and returns *total* posterior mean and square root of pointwise variance

           NOTES:
           * the returned *total* posterior mean is the sum of the GP posterior mean and the mean function

        """

        # FIXME: Need to implement a way to calculate the posterior for a subset of vertices (usefully, for first self.X.shape[0] positions i.e. vertices)
        #        This is because we will want to take posterior samples, and so need the full posterior covariance, but for all vertices this is too big
        # FIXME: also need to internally save the resulting matrices for use by the posteriorSamples() function

        # set outputs and inputs
        y = self.y.T.flatten()  # stack them output by outputs i.e. y = y_1(x_1) ... y_1(x_N), y_2(x_1) ... y_2(x_N) 

        # calculate spectral density
        self.SD = self.spectralDensity( self.HP )

        # recalculate self.invSigma
        self.make_invSigma()

        # for K** and K*
        V = np.sqrt(self.SD)*self.V[self.vertex]
        Vpred = np.sqrt(self.SD)*self.V[0:self.X.shape[0]] # NOTE: I am predicting at vertices only, for time being
        cross_cov = self.kernelMatrix(Vpred, V, noise = False)

        # posterior mean
        postmean = cross_cov.dot( self.invSigma.dot(y) )
        
        # posterior variance
        if pointwise == False: # full covariance, very memory intensive
            pred_cov = self.kernelMatrix(Vpred, Vpred, noise = False)
            postvar = pred_cov - cross_cov.dot( self.invSigma.dot(cross_cov.T) )
        else: # pointwise i.e. diagonal only
            postvar = self.kernelMatrix(Vpred, Vpred, noise = False, pointwise = True) - np.einsum("ij, ji -> i", cross_cov, self.invSigma.dot(cross_cov.T))

        # return results
        if pointwise == True:
            # reshape so that columns are different outputs...
            postmean = postmean.reshape((-1, Vpred.shape[0])).T
            postvar = postvar.reshape((-1, Vpred.shape[0])).T
            return self.meanFunction + self.unscale(postmean), self.unscale(np.sqrt(postvar), std = True)
        else:
            return self.meanFunction + self.unscale(postmean), self.unscale(np.sqrt(np.diag(postvar)), std = True)

    #}}}

    #{{{ posterior samples
    def posteriorSamples(self, num):
        """Returns posterior samples from the whole model (samples from the posterior + mean function)
        
        Arguments:
        num -- how many posterior samples to calculate
        
        """

        if self.post_var == None or self.post_mean == None:
            print("[ERROR]: need to call posterior first, and succeed in saving full posterior covariance.")

        print("Calculating {:d} samples from posterior distribution...".format(num))

        nugget = 1e-10
        while True:
            try:
                print("  (adding nugget {:1.0e} to posterior var diagonal for stability)".format(nugget))
                np.fill_diagonal(self.post_var, np.diag(self.post_var) + nugget)
                #print("  Cholesky decomp")
                L = np.linalg.cholesky(self.post_var)
                break
            except:
                nugget = nugget * 10
        
        N = self.post_mean.shape[0]

        #print("Calculate samples")
        u = np.random.randn(N, num)
        samples = self.post_mean[:,None] + L.dot(u)

        return self.meanFunction[:,None] + self.unscale(samples)
        
    #}}}


# subclasses implementing different models for the reduced-rank gpmi
# ==================================================================

#{{{ subclasses for different models

#{{{ Matern for GP, no mean function
# FIXME: currently implemented a hack whereby SpectralDensity is calculated and stored outside of kernel functions
class Matern(AbstractModel):
    """Model y = GP(x) + e where GP(x) is a manifold GP with Matern kernel."""       

    def setup(self, smoothness):

        # kernel smoothness parameter
        self.smoothness = smoothness

        # output correlation matrix A = L.L^T; hard-wired full-rank; L is lower diagonal
        #M = int(self.y.shape[1]*(self.y.shape[1] + 1)/2)
        self.L_idx = np.tril_indices(self.y.shape[1], 0) # allows for indexing into L to set the non-zero values
        self.L = np.zeros((self.y.shape[1],self.y.shape[1]))
        self.L[self.L_idx] = 1
        print("L:\n", self.L)

        # noise parameters; stored in flat array
        self.D = 1e-2 * np.ones(self.y.shape[1])
        print("D:\n", self.D)

        # kernel parameters
        self.HP = 1.0

        # precalculate Spectral Density terms that do not change
        # ------------------------------------------------------
        D = 2  # dimension
        v = self.smoothness  # smoothness
        w = np.sqrt(self.Q)  # sqrt of eigenfunctions 
        self.alpha = (2**D * np.pi**(D/2) * gamma(v + (D/2)) * (2*v)**v) / gamma(v)
        self.gamma = 4*(np.pi**2)*(w**2) 
        self.expo = -(v + (D/2))

        # will be zero, so hiding this call in here...
        self.setMeanFunction()


    def kernelMatrix(self, a, b, noise = False, pointwise = False):
        """Sigma = A \kron K(X,X') + Delta"""

        # create A from L
        A = self.L.dot(self.L.T)

        if pointwise == False:

            # create K(X,X) = PHI * PHI.T (as sqrt(SD) absorbed into Phi)
            K = a.dot(b.T)

            # create Sigma matrix (without adding noise)
            Sigma = kron(A, K)

            # add the noise matrix
            if noise == True:
                delta = np.repeat(self.D, self.vertex.shape[0]) + self.yerr.T.flatten()**2
                np.fill_diagonal(Sigma, Sigma.diagonal() + delta)

        else: # diagonal only (purpose is for posterior, no option to add noise)

            Sigma = np.kron(np.diag(A), np.einsum("ij, ij -> i", a, b))

        return Sigma


    def make_invSigma(self, V_precalc = None):
        """ Inverse of Sigma, taking advantage of reduced-rank formulation of K.
        
            Sigma := A \kron K + Delta, with K = V.V^T (sqrt(SD) absorbed into V in advance)
            
            Delta = D \kron I_N + yerr, where
                D is (noise_var_1, noise_var_2, ..., noise_var_M)
                yerr**2 is observation specific noise variance (for all output 1 obs, then output 2 obs, ..., etc.)
        """
        
        # absorb SD into V
        V = np.sqrt(self.SD)*self.V[self.vertex] if V_precalc is None else V_precalc

        # get Delta^-1 = inv(D \kron I_N + y_err^2)
        # -----------------------------------------
        #inv_delta = np.repeat(1.0/self.D, N) # no heteroscedastic niose
        delta = np.repeat(self.D, self.vertex.shape[0]) + self.yerr.T.flatten()**2
        inv_delta = 1.0/delta
            
        # calculate inverse of Sigma = A \kron K + D
        # ------------------------------------------

        # define B := L \kron Phi
        B = kron(self.L, V)

        # parts of inverse expression
        inv_delta_B = inv_delta[:,None]*B
        B_inv_delta_B = B.T.dot(inv_delta_B)

        # now calculate the inverse - use Cholesky because I'll need it later
        np.fill_diagonal(B_inv_delta_B, B_inv_delta_B.diagonal() + 1.0) # B_inv_delta_B --> B_inv_delta_B + Identity
        L_TMP = linalg.cho_factor(B_inv_delta_B)

        # calculate inverse of Sigma = A \kron K + D
        self.invSigma = - ( inv_delta_B ).dot( linalg.cho_solve(L_TMP, inv_delta_B.T) )
        np.fill_diagonal(self.invSigma, self.invSigma.diagonal() + inv_delta)

        #{{{ check result against explicit inverse
        #Sigma = self.kernelMatrix(V, V, noise = True)
        #explicit_invSigma = np.linalg.inv(Sigma)
        #print("max difference:\n", (self.invSigma - explicit_invSigma).max())
        #print("Same?", np.allclose(self.invSigma, explicit_invSigma))
        #}}}

        return L_TMP


    def spectralDensity(self, rho):
        """Spectral Density for Matern kernel.
        
           w: sqrt(eigenvalues)
           rho: lengthscale parameter

        """
        
        v = self.smoothness  # smoothness

        delta = 1.0 / rho**(2*v)

        beta = 2*v / (rho**2)  + self.gamma 

        result = (self.alpha*delta) * beta**self.expo
        
        return result


    def setMeanFunction(self):
        """Set mean function."""
        self.meanFunction = np.array([0])
#}}}

#}}}

