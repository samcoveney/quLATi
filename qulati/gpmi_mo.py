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

        self.D = np.exp(guess[-self.y.shape[1]:])

        return
    #}}}

    #{{{ negative loglikelihood
    def LLH(self, guess):
        """Return the negative loglikelihood.
        
           Arguments:
           guess -- log(hyperparameter values) for training

           NOTES:
           * this function receives guess = log(HP), therefore derivatives wrt guess are returned as dLLH/dg = dLLH/dHP * dHP/dguess
        
        """

        #print("guess:", guess)
        self.HP_from_guess(guess)
        #print(self.L); input()

        # calculate spectral density
        self.SD = self.spectralDensity( self.HP, np.sqrt(self.Q) ) # self.HP is just the lengthscale hyperparameter

        # set outputs and inputs
        y = self.y.T.flatten()
        V = np.sqrt(self.SD)*self.V[self.vertex] # NOTE: absorb SD(eigenvalues) into the eigenvectors

        # create inverse of A \kron K_spatial; L_TMP is the cholesky of matrix Z needed for an efficient inverse
        L_TMP = self.make_invSigma() # NOTE: L_TMP is the Cholesky of the small inverse needed for efficient invSigma calculation
        invK = self.invSigma
        invK_y = invK.dot(y)


        # calculating log | K |
        # ---------------------

        # NOTE: full matrix "K", actually Sigma := A \kron K(x,x'), not used in LLH routine, just used here for checking logDetK
        #K = self.kernelMatrix(V, V, nugget = True) # this is the full kernel Sigma := A \kron K(x,x'), although it's called 'K'
        #L_test = linalg.cho_factor(K)        
        #logDetK = 2.0*np.sum(np.log(np.diag(L_test[0])))        
        #print("logDetK:", logDetK)

        # calculate logDetK with the expression I derived
        logDetK = 2*np.sum(np.log(np.diag(L_TMP[0]))) + self.y.shape[0]*np.sum(np.log(self.D))
        #print("logDetK:", logDetK)

        # calculate loglikelihood
        # -----------------------
        llh = 0.5 * ( logDetK + ( y.T ).dot( invK_y ) + y.shape[0]*np.log(2*np.pi) )  # NOTE: I'm not entirely sure about the constant when we have derivatives
        #print("llh:", llh)

        return llh
        
    #}}}

    #{{{ optimize the hyperparameters
    def optimize(self, restarts):
        """Optimize the hyperparameters.
        
           Arguments:
           restart -- how many times to restart the optimizer.

           NOTES:
           * initial guesses are picked at random from predefined ranges set in this function
        
        """

        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str(x)[:5] % x

        # initial guesses for hyperparameters
        # -----------------------------------

        hdr = "| Restart | "

        # lengthscale guess
        dguess = np.random.uniform(np.log(10), np.log(100), size = restarts).reshape([1,-1]).T
        hdr = hdr + " len " + " | " 


        # new attempt, without any transformation of L at all... just in case!
        if True:
            for i in range(0, self.L_idx[0].shape[0]):
                sg = np.random.uniform(-5.0, 5.0, size = restarts).reshape([1,-1]).T
                sguess = sg if i == 0 else np.hstack([sguess, sg])
                hdr = hdr + " L{:d}{:d} ".format(self.L_idx[0][i], self.L_idx[1][i]) + " | "
            print("L guesses original:", sguess)

        else:
            if True: # old, random for L
                # multi-output covariance guesses; flattened values such that self.L[self.L_idx] filled by sguess
                # NOTE: IS THIS CORRECT? CERTAINLY FOR A = L.dot(L^T), WE WANT CONDITIONS ON A IDEALLY, BUT HOW DOES THAT TRANSLATE TO L PROPERLY?
                for i in range(0, self.L_idx[0].shape[0]):
                    if self.L_idx[0][i] == self.L_idx[1][i]: # exp transform on diagonals
                        sg = np.random.uniform(np.log(1), np.log(10), size = restarts).reshape([1,-1]).T
                    else: # no transform on off-diagonals
                        sg = np.random.uniform(-5.0, 5.0, size = restarts).reshape([1,-1]).T
                    sguess = sg if i == 0 else np.hstack([sguess, sg])
                    hdr = hdr + " L{:d}{:d} ".format(self.L_idx[0][i], self.L_idx[1][i]) + " | "
                print("L guesses original:", sguess)

            else: # try to create an initial sensible estimate for these values; this does not really give good results
                # multi-output covariance guesses; flattened values such that self.L[self.L_idx] filled by sguess
        
                YCORR = np.cov(self.y.T)
                print(YCORR)
                L = linalg.cholesky(YCORR, lower = True)
                print(L)
                L_lower = L[np.tril_indices(L.shape[0])]
                print(L_lower)

                for i in range(0, self.L_idx[0].shape[0]):
                    if self.L_idx[0][i] == self.L_idx[1][i]: # exp transform will be applied on diagonals in LLH
                        sg = np.log(L_lower[i])
                    else: # no transform on off-diagonals
                        sg = L_lower[i]
                    sguess = sg if i == 0 else np.hstack([sguess, sg])
                    hdr = hdr + " L{:d}{:d} ".format(self.L_idx[0][i], self.L_idx[1][i]) + " | "
                sguess = sguess[None,:] 
                sguess = np.tile(sguess, restarts).reshape(restarts, -1)
                #print("sguess:", sguess)


        # nugget guesses
        for i in range(0,self.y.shape[1]):
            ng = np.random.uniform(np.log(1.0), np.log(10.0), size = restarts).reshape([1,-1]).T
            nguess = ng if i == 0 else np.hstack([nguess, ng])
            hdr = hdr + " nu{:d} ".format(i) + " | "

        guess = np.hstack([dguess, sguess, nguess])

        # minimimize LLH
        # --------------
        print("Optimizing Hyperparameters...\n" + hdr)

        for gn, g in enumerate(guess):
            optFail = False
            try:
                bestStr = "   "
                #res = minimize(self.LLH, g, method = 'Nelder-Mead') 
                res = minimize(self.LLH, g, method = 'L-BFGS-B') 

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


        self.HP_from_guess(bestRes.x)
        print("Optimization complete.")


    #}}}

    #{{{ posterior distribution
    def posterior(self, pointwise = False):
        """Calculates GP posterior mean and variance and returns *total* posterior mean and square root of pointwise variance

           NOTES:
           * the returned *total* posterior mean is the sum of the GP posterior mean and the mean function

        """

        # FIXME: Need to implement a way to calculate the posterior for a subset of vertices (usefully, for first self.X.shape[0] positions i.e. vertices)
        #        This is because we will want to take posterior samples, and so need the full posterior covariance

        # set outputs and inputs
        y = self.y.T.flatten()  # stack them output by outputs i.e. y = y_1(x_1) ... y_1(x_N), y_2(x_1) ... y_2(x_N) 

        # calculate spectral density
        self.SD = self.spectralDensity( self.HP, np.sqrt(self.Q) )

        # recalculate self.invSigma
        self.make_invSigma()

        # for K** and K*
        V = np.sqrt(self.SD)*self.V[self.vertex]
        Vpred = np.sqrt(self.SD)*self.V[0:self.X.shape[0]] # NOTE: I am predicting at vertices only, for time being
        cross_cov = self.kernelMatrix(Vpred, V)

        # mean function
        postmean = cross_cov.dot( self.invSigma.dot(y) )
        postmean = postmean.reshape((-1, Vpred.shape[0])).T # reshape so that columns are different outputs...

        return self.meanFunction + self.unscale(postmean)

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

        # will be zero, so hiding this call in here...
        self.setMeanFunction()


    def kernelMatrix(self, a, b, nugget = False):
        """Sigma = A \kron K(X,X') + D \kron I_N"""

        # create A from L
        A = self.L.dot(self.L.T)

        # create K(X,X) = PHI * PHI.T (as sqrt(SD) absorbed into Phi)
        K = a.dot(b.T)

        # create Sigma matrix
        Sigma = kron(A, K)
        
        # FIXME: NOISE
        if nugget == True: 
            N = K.shape[0]
            Delta = np.diag(np.repeat(self.D, N))

            return (Sigma + Delta)

        else:

            return Sigma


    def make_invSigma(self):
        """ Inverse of Sigma, taking advantage of reduced-rank formulation of K.
        
            Sigma := A \kron K + Delta, with K = V.V^T (sqrt(SD) absorbed into V in advance)
        """

        # NOTE: may be more efficient to avoid formation of invD

        # FIXME: NOISE
        # get inv(D \kron I_N)
        N = self.vertex.shape[0]
        #inv_delta = np.diag(np.repeat(1.0/self.D, N))
        inv_delta = np.repeat(1.0/self.D, N) # NOTE: changed
        
        # define B: L \kron PHI
        V = np.sqrt(self.SD)*self.V[self.vertex]
        B = kron(self.L, V)

        # parts of inverse expression
        #tmp_1 = np.diag(np.tile(1.0/SD, self.y.shape[1]))
        #tmp_2 = np.diag(inv_delta).dot(B) # old way, but since inv_delta flat above, i put the np.diag in here
        tmp_2 = inv_delta[:,None]*B # NOTE: changed

        tmp_3 = B.T.dot(tmp_2)

        # now calculate the inverse - use Cholesky because I'll need it later
        #ONES = np.eye(V.shape[1] * self.y.shape[1])
        #L_TMP = linalg.cho_factor(ONES + tmp_3) # NOTE: formerly tmp_1 + tmp_3 before SD absorption
        np.fill_diagonal(tmp_3, tmp_3.diagonal() + 1.0)
        L_TMP = linalg.cho_factor(tmp_3) # NOTE: formerly tmp_1 + tmp_3 before SD absorption


        # FIXME: this will be wrong until I change inv_delta again
        #self.invSigma_old = np.diag(inv_delta) - ( tmp_2 ).dot( linalg.cho_solve(L_TMP, tmp_2.T) ) # old way, but since inv_delta flat above, i put the np.diag in here
        self.invSigma = - ( tmp_2 ).dot( linalg.cho_solve(L_TMP, tmp_2.T) )
        np.fill_diagonal(self.invSigma, self.invSigma.diagonal() + inv_delta)

        #print(self.invSigma_old, "\n", self.invSigma)
        #print("close?", np.allclose(self.invSigma_old, self.invSigma))

        # check against explicit invere
        #explicit_invSigma = np.linalg.inv(self.Sigma)
        #print("Same?", np.allclose(invSigma, explicit_invSigma))

        return L_TMP


    def spectralDensity(self, rho, w, grad = False):
        """Spectral Density for Matern kernel.
        
           w: sqrt(eigenvalues)
           rho: lengthscale parameter

        """
        
        D = 2  # dimension
        v = self.smoothness  # smoothness

        alpha = (2**D * np.pi**(D/2) * gamma(v + (D/2)) * (2*v)**v) / gamma(v)
        delta = 1.0 / rho**(2*v)

        beta = 2*v / (rho**2)  + 4*(np.pi**2)*(w**2) 

        expo = -(v + (D/2))
        result = (alpha*delta) * beta**expo
        
        return result


    def setMeanFunction(self):
        """Set mean function."""
        self.meanFunction = np.array([0])
#}}}

#}}}

