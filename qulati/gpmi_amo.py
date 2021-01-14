"""
   gpmi_amo.py

   Module implementing Gaussian Process Manifold Interpolation (GPMI) for multiple outputs.

   This model is an 'alternative multiple output' model. The between-outputs correlation comes *only* from the noise matrix.

   Contains:
      class AbstractModel
      class Matern(AbstractModel)

   Created: 25-Aug-2020
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
from scipy.linalg import block_diag

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

# abstract class
# ==============

class AbstractModel(ABC):
    """ Gaussian Process Manifold Interpolation for multiple outputs, with between-outpus correlations coming only from correlated observation specific noise."""

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
        """ Set data for interpolation, scaled and centred.

            yerr: array of variances of shape N x (m X m)

            Notes:
            * yerr represents, for each vertex, multi-output observation noise. This is processed here into a noise matrix that can added to the main kernel matrix.
            * yerr is allowed to be None (becomes zeros), in which case the model should be identical to single-output GPs. This is for validation only.
            * yerr is a variance in this model, but is a standard deviation in gpmi_rr (and gpmi_mo when noise is univariate).

        """

        # set mean and stdev of y
        self.y_mean, self.y_std = np.mean(y, axis = 0), np.std(y, axis = 0)
        self.y = self.scale(y, std = False)
        print(self.y.min(axis = 0), self.y.max(axis = 0))

        if yerr is None:
            self.yerr = np.zeros((y.size, y.size))
        else: # construct the multivariate noise matrix
            
            self.yerr = np.zeros((y.size, y.size))

            for v in np.arange(y.shape[0]):

                indices = []
                for r in range(0,y.shape[1]):
                    pad = r * y.shape[1] * y.shape[0] * y.shape[0] + v * y.shape[1] * y.shape[0] 
                    indices = indices + [pad + v, pad + y.shape[0] + v , pad + 2*y.shape[0] + v ]

                np.put(self.yerr, indices, yerr[v])

            # scale yerr
            U = np.repeat(1.0/self.y_std, y.shape[0])
            self.yerr = U * self.yerr * U[:,None]
            
        #plt.imshow(self.yerr, cmap = "seismic")
        #plt.show()

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
        self.HP = np.exp(guess)

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
        if self.nugget_train: self.nugget = self.HP[-self.y.shape[1]:] # setting nugget value here too
        #print("HP in LLH:", self.HP)
        #print("nugget:", self.nugget)

        y = self.y.T.flatten()

        a = self.V[self.vertex]
        Sigma = self.kernelMatrix(a, a, noise = True) # NOTE: instead of defining V as self.V[vertex], this happens in kernel function... maybe not best for generality

        L = linalg.cho_factor(Sigma)
        logDetK = 2.0*np.sum(np.log(np.diag(L[0])))        
        invK_y = linalg.cho_solve(L, y)
    
        llh = 0.5 * ( logDetK + ( y.T ).dot( invK_y ) + y.shape[0]*np.log(2*np.pi) )

        return llh
        
    #}}}

    #{{{ optimize the hyperparameters
    def optimize(self, restarts, nugget = None, guess_seed = None):
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

        # if a guess_seed has not been provided, then create one
        if guess_seed is None:
            guess_seed = {}
            for i in range(self.y.shape[1]): 
                guess_seed["len_{:d}".format(i)] = [10,100]
                guess_seed["var_{:d}".format(i)] = [10,100]
                guess_seed["nug_{:d}".format(i)] = [1e-3,1e-2]


        # lengthscale guesses
        dguess = np.empty((restarts, self.y.shape[1]))
        for i in range(self.y.shape[1]):
            hdr = hdr + " len " + " | "
            dguess[:,i] = np.random.uniform(np.log(guess_seed["len_{:d}".format(i)][0]), np.log(guess_seed["len_{:d}".format(i)][1]), size = (restarts) )

        # variance guesses
        sguess = np.empty((restarts, self.y.shape[1]))
        for i in range(self.y.shape[1]):
            hdr = hdr + " var " + " | "
            sguess[:,i] = np.random.uniform(np.log(guess_seed["var_{:d}".format(i)][0]), np.log(guess_seed["var_{:d}".format(i)][1]), size = (restarts) )

        # combine lengthscale and variance guesses
        guess = np.hstack([dguess, sguess])

        # nugget options
        if nugget is None:  # nugget not supplied; we must train on nugget
            self.nugget_train = True

            # nugget guesses
            nguess = np.empty((restarts, self.y.shape[1]))
            for i in range(self.y.shape[1]):
                hdr = hdr + " nug " + " | "
                nguess[:,i] = np.random.uniform(np.log(guess_seed["nug_{:d}".format(i)][0]), np.log(guess_seed["nug_{:d}".format(i)][1]), size = (restarts) )

            guess = np.hstack([guess, nguess])

        else: # fix the nugget
            self.nugget_train = False

            self.nugget = np.abs(nugget)

            hdr = hdr + " (fixed nuggets) "

        #print("guess:\n", guess)

        # minimimize LLH
        # --------------
        print("Optimizing Hyperparameters...\n" + hdr)

        for gn, g in enumerate(guess):
            optFail = False
            try:
            #if True:
                bestStr = "   "
                res = minimize(self.LLH, g, method = 'Nelder-Mead') 

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
                self.nugget = self.HP[-self.y.shape[1]:]

                print("|  {:02d}/{:02d} ".format(gn + 1, restarts),
                      "| %s" % ' | '.join(map(str, [fmt(i) for i in self.HP])),
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

        # covariance between data
        a = self.V[self.vertex]
        Sigma = self.kernelMatrix(a, a, noise = True) 
        L = linalg.cho_factor(Sigma)

        # for K** and K*
        b = self.V[0:self.X.shape[0]] # NOTE: predicting at vertices only
        cross_cov = self.kernelMatrix(b, a, noise = False)

        # posterior mean
        postmean = cross_cov.dot( linalg.cho_solve(L, y) )

        
        # posterior variance choice (pointwise or not), then return result
        # ----------------------------------------------------------------
    
        if pointwise == False: # full covariance, very memory intensive

            # FIXME: not working yet
            pred_cov = self.kernelMatrix(Vpred, Vpred, noise = False)
            postvar = pred_cov - cross_cov.dot( self.invSigma.dot(cross_cov.T) )

            return self.meanFunction + self.unscale(postmean), self.unscale(np.sqrt(np.diag(postvar)), std = True)

        else: # pointwise i.e. diagonal only

            postmean = postmean.reshape((-1, b.shape[0])).T
            return self.meanFunction + self.unscale(postmean), None # FIXME: currently just returning the mean, for ease of testing

            postvar = self.kernelMatrix(b, b, noise = False, pointwise = True) - np.einsum("ij, ji -> i", cross_cov, linalg.cho_solve(L, cross_cov.T))
            postvar = postvar.reshape((-1, b.shape[0])).T
            return self.meanFunction + self.unscale(postmean), self.unscale(np.sqrt(postvar), std = True)

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
class Matern(AbstractModel):
    """Model y = GP(x) + e where GP(x) is a manifold GP with Matern kernel."""       

    def kernelSetup(self, smoothness):

        # kernel smoothness parameter
        self.smoothness = smoothness

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
        """
            Sigma = block_diagonal_K + Noise, where each block is a kernel matrix for each output feature

            Hyperparameters in self.HP stored as len, len, ..., var, var, ..., nug, nug, ...

        """

        blocks = []         
        for i in range(self.y.shape[1]):

            # spectral density
            SD = self.spectralDensity( self.HP[i] )
            #print("len[{:d}]:".format(i), self.HP[i])
        
            A = np.sqrt(SD)*a # NOTE: absorb SD(eigenvalues) into the eigenvectors
            B = np.sqrt(SD)*b # NOTE: absorb SD(eigenvalues) into the eigenvectors
            sub_K = self.HP[self.y.shape[1] + i] * A.dot(B.T)
            #print("var[{:d}]:".format(i), self.HP[self.y.shape[1] + i])

            if noise: np.fill_diagonal(sub_K, np.diag(sub_K) + self.nugget[i])
            #print("nug[{:d}]:".format(i), self.nugget[i])

            blocks.append(sub_K)

        Sigma = block_diag(*blocks)
        
        if noise: Sigma = Sigma + self.yerr

        #plt.imshow(Sigma)
        #plt.show()

        return Sigma


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

