"""
   gpmi_rr.py

   Module implementing Gaussian Process Manifold Interpolation (GPMI), making use of the reduced-rank of the covariance matrix.

   Contains:
      class AbstractModel
      class Matern(AbstractModel)


   Specifically, this module makes use of the reduced-rank form of the covariance matrix.
   The models are fast and can handle very large amounts of data.

   These models are less flexible; auxillary inputs cannot be supplied.

   At present heteroscedastic noise and gradient observations cannot be supplied.


   Created: 07-Apr-2020
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


# abstract class utlizing the reduced-rank speedups
# ================================================

class AbstractModel(ABC):
    """ Gaussian Process Manifold Interpolation, making use of reduced-rank expressions.
    
        Much less flexible. Will not be able to accept gradient observations or auxillary observations.

        Heteroscedastic noise can now be used.

        May or may not be able to prediction the posterior gradients with the reduced rank framework...
    """

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
    def __init__(self, X, Tri, Q, V, gradV):
        """Initialize class with manifold data."""

        self.X = X
        self.Tri = Tri
        self.Q = Q
        self.V = V
        self.gradV = gradV

        self.reset()

        super().__init__()


    def __del__(self):
        """Method to cleanup when we delete the instance of this class."""
        pass


    def reset(self):
        """Resets parameters and predictions."""

        self.HP, self.nugget = None, None
        self.post_mean, self.post_var = None, None
        self.grad_post_mean, self.grad_post_var = None, None

        # set these values so that user does not have to call setMeanFunction
        self.meanFunction, self.grad_meanFunction = 0, np.array([[0,0,0]])


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
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y = self.scale(y, std = False)

        if yerr is None:
            self.yerr = np.zeros(y.shape[0])
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

    #{{{ negative loglikelihood
    def LLH(self, guess, grad = False):
        """Return the negative loglikelihood.
        
           Arguments:
           guess -- log(hyperparameter values) for training

           NOTES:
           * this function receives guess = log(HP), therefore derivatives wrt guess are returned as dLLH/dg = dLLH/dHP * dHP/dguess
        
        """

        # FIXME: for now, fix grad = False, so we can convert llh without worrying about grad_llh; can add grad_llh later
        grad = False
       
        # set hyperparameter values
        self.HP = np.exp(guess)
        if self.nugget_train: self.nugget = self.HP[self.nugget_index]

        # spectral density
        SD, _ = self.spectralDensity( self.HP[0], np.sqrt(self.Q) ) # NOTE: multiply SD by signal variance
        SD = self.HP[1] * SD

        # set outputs and inputs
        y = self.y
        V = np.sqrt(SD)*self.V[self.vertex] # NOTE: absorb SD(eigenvalues) into the eigenvectors

        # define Q := phi phi^T + D
        # D is diagonal matrix of observation variances + nugget, but can represent as a vector only for ease

        # form noise matrix (diagonal, store as vector)
        D = self.yerr**2 + self.nugget
        invD = 1.0/D

        # form Z (different from Solin paper)
        ones = np.eye(SD.shape[0])
        Z = ones + (V.T).dot(invD[:,None]*V) # invD is diagonal, faster than diag(invD).dot(V)

        #{{{ ###### Woodbury inversion tests #######
        if False:
            E = np.diag(D)
            F = V
            H = F.T
            G = np.diag(np.ones(SD.shape[0]))

            # standard invserse
            Q = E + F.dot(G).dot(H)
            invQ = np.linalg.inv(Q)

            # Woodbury inverse with SD absorbed
            invE = np.linalg.inv(E)
            Z = np.linalg.inv(G) + H.dot(invE).dot(F)
            invZ = np.linalg.inv(Z)
            #invQ_wb = invE - invE.dot(F).dot(invZ).dot(H).dot(invE)
            invQ_wb = invE - (invE.dot(F)).dot(invZ).dot(H.dot(invE))
            print("close?", np.allclose(invQ, invQ_wb))
            print("difference:", invQ - invQ_wb)
            input("waiting...")

            # Woodbury inverse with SD not absorbed...
            E = np.diag(D)
            invE = np.linalg.inv(E)
            F = self.V[self.vertex]
            H = F.T
            G = np.diag(SD)
            invG = np.diag(1.0/SD)
            Z = invG + H.dot(invE).dot(F)
            invZ = np.linalg.inv(Z)
            #invQ_wb = invE - invE.dot(F).dot(invZ).dot(H).dot(invE)
            invQ_wb = invE - (invE.dot(F)).dot(invZ).dot(H.dot(invE))
            print("close?", np.allclose(invQ, invQ_wb))
            print("difference:", invQ - invQ_wb)
            input("waiting...")

        #}}} ######

        try:

            # attempt cholesky factorization
            # ------------------------------
            #L = linalg.cho_factor(Z)
            L = linalg.cholesky(Z, lower = True) # NOTE: trying cholesky instead of cho_solve
            cho_success = True
            #print("cholesky success")

            # log |Q| = log |Z| + log |D|
            # ---------------------------
            #logDetZ = 2.0*np.sum(np.log(np.diag(L[0])))
            logDetZ = 2.0*np.sum(np.log(np.diag(L))) # NOTE: trying cholesky instead of cho_solve
            log_Q = logDetZ + np.sum(np.log(D))  # SD absorbed into eigenvectors

            # y^T Q^-1 y
            # ----------
            tmp = V.T.dot(invD*y)
            #yQy = y.dot(invD*y) - (tmp.T).dot( linalg.cho_solve(L, tmp) )
            # NOTE: trying cholesky instead of cho_solve
            tmp = (tmp.T).dot( linalg.solve_triangular(L.T, linalg.solve_triangular(L, tmp, lower = True)) )
            yQy = y.dot(invD*y) - tmp

            # LLH = 1/2 log|Q| + 1/2 y^T Q^-1 y + n/2 log(2 pi)
            n_log_2pi = y.shape[0]*np.log(2*np.pi)
            llh = 0.5 * (log_Q + yQy + n_log_2pi)


            #{{{ ###### more Woodbury inversion tests #######
            if False:
                print("--------------")
                print("log|Q|:", log_Q)
                print("yKy:", yQy)
                print("llh_rr:", llh)
                K = V.dot(V.T) + np.diag(D)
                L = linalg.cho_factor(K)        
                invK_y = linalg.cho_solve(L, y)
                logDetK = 2.0*np.sum(np.log(np.diag(L[0])))        
                yKy = ( y.T ).dot( invK_y )
                llh_old = 0.5 * ( logDetK + yKy + y.shape[0]*np.log(2*np.pi) )
                print("log|Q| old:", logDetK)
                print("yKy old:", yKy)
                print("llh_old:", llh_old)
                print("--------------")
            #}}} ###### 

        # NOTE: version which avoids Cholesky not implemented yet

        except np.linalg.linalg.LinAlgError as e:
            cho_success = False
            print("\n[WARNING]: Matrix not PSD for", self.HP, ", not fit.\n")
            return None

        except ValueError as e:
            cho_success = False
            print("\n[WARNING]: Ill-conditioned matrix for", self.HP, ", not fit.\n")
            return None

        return llh

        # NOTE: LLH gradients for reduced-rank formulation not implemented yet
        
    #}}}

    #{{{ optimize the hyperparameters
    def optimize(self, nugget, restarts, llh_gradients = False):
        """Optimize the hyperparameters.
        
           Arguments:
           nugget -- value of the nugget, if None then train nugget, if number then fix nugget
           restart -- how many times to restart the optimizer.

           NOTES:
           * values for hyperparameters for training are supplied as log(HP) to the loglikelihood function
           * currently using Newton-CG method needing LLH gradients, hopefully these LLH gradients wrt log(HP) are correct...
           * initial guesses are picked at random from predefined ranges set in this function
        
        """

        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str(x)[:5] % x

        # initial guesses for hyperparameters
        dguess = np.random.uniform(np.log(10), np.log(100), size = restarts).reshape([1,-1]).T
        sguess = np.random.uniform(np.log(10), np.log(100), size = restarts).reshape([1,-1]).T
        nguess = np.random.uniform(np.log(1e-3), np.log(1e-2), size = restarts).reshape([1,-1]).T

        hdr = "Restart | " + " len " + " | " + " var " + " | "
        guess = np.hstack([dguess, sguess])

        # nugget
        if nugget is None:  # nugget not supplied; we must train on nugget
            self.nugget_train = True
            guess = np.hstack([guess, nguess])
            self.nugget_index = guess.shape[1] - 1
            hdr = hdr + " nug " + " | " 
        else:
            self.nugget = np.abs(nugget)
            self.nugget_train = False

        # run optimization code
        grad_print = "with LLH gradients" if llh_gradients else ""
        print("Optimizing Hyperparameters" + grad_print + "...\n" + hdr)

        for ng, g in enumerate(guess):
            optFail = False
            try:
                bestStr = "   "
                if llh_gradients: res = minimize(self.LLH, g, args = (True), method = 'L-BFGS-B', jac=True) # make use of gradLLH
                else:             res = minimize(self.LLH, g, args = (False), method = 'Nelder-Mead') 

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

                print(" {:02d}/{:02d} ".format(ng + 1, restarts),
                      "| %s" % ' | '.join(map(str, [fmt(i) for i in np.exp(res.x)])),
                      "| {:s} llh: {:.3f}".format(bestStr, -1.0*np.around(res.fun, decimals=4)))


            except TypeError as e:
                optFail = True

        self.HP = np.exp(bestRes.x)
        self.nugget = self.HP[self.nugget_index] if nugget is None else nugget 

        print("\nHyperParams:", ", ".join(map(str, [fmt(i) for i in self.HP])), \
              "\n  (nugget: {:f})".format(self.nugget))

    #}}}

    #{{{ posterior distribution, old version with no specified heteroscedastic observation noise
    def posterior(self, indices = None, pointwise = True):
        """Calculates GP posterior mean and variance and returns *total* posterior mean and square root of pointwise variance

           NOTES:
           * the returned *total* posterior mean is the sum of the GP posterior mean and the mean function

        """

        # FIXME: Need to implement a way to calculate the posterior for a subset of vertices (usefully, for first self.X.shape[0] positions i.e. vertices)
        #        This is because we will want to take posterior samples, and so need the full posterior covariance

        # spectral density
        SD, _ = self.spectralDensity( self.HP[0], np.sqrt(self.Q) ) # NOTE: multiply SD by signal variance
        SD = self.HP[1] * SD

        # set outputs and inputs
        y = self.y
        V = np.sqrt(SD)*self.V[self.vertex] # NOTE: absorb SD(eigenvalues) into the eigenvectors

        # where to calculate the posterior
        if indices is None: indices = np.arange(self.V.shape[0])
        V_other = np.sqrt(SD)*self.V[indices]

        # form noise matrix (diagonal, store as vector)
        D = self.yerr**2 + self.nugget
        invD = 1.0/D

        # explicit formation of covariances 
        k_train = V.dot(V.T) + np.diag(D)
        k_cross = V_other.dot(V.T)

        try:

            L = linalg.cho_factor(k_train)
            Ef = k_cross.dot(linalg.cho_solve(L, y))
            self.post_mean = Ef

            if pointwise == False:
                k_pred = V_other.dot(V_other.T)
                Vf = k_pred - k_cross.dot(linalg.cho_solve(L, k_cross.T))
                self.post_var = Vf
            else:
                k_pred = np.einsum("ij, ij -> i", V_other, V_other)
                tmp = np.einsum("ij, ji -> i", k_cross, linalg.cho_solve(L, k_cross.T))
                Vf = k_pred - tmp 
                self.post_var = Vf

        except MemoryError as me:
            print("[MEMORY ERROR]: cannot store such a big posterior covariance matrix! Try 'indices' or 'pointwise' arguments to this function.")

            print("[WARNING]: setting posterior variance as None.")
            Vf = None
            self.post_var = Vf

        except np.linalg.linalg.LinAlgError as e:
            print("\n[WARNING]: Matrix not PSD.\n")
            return None

        except ValueError as e:
            print("\n[WARNING]: ValueError of some sort.\n")
            return None


        # return the posterior mean and posterier (pointwise) standard deviation
        if Ef.shape == Vf.shape:
            return self.meanFunction + self.unscale(Ef), self.unscale(np.sqrt(Vf), std = True)
        else: 
            return self.meanFunction + self.unscale(Ef), self.unscale(np.sqrt(np.diag(Vf)), std = True)

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

    #{{{ posterior distribution of gradient
    def posteriorGradient(self):
        """Calculates GP posterior mean and variance *of gradient* and returns posterior mean and square root of pointwise variance
        
           NOTES:
           * the returned *total* posterior mean *of gradient* is the sum of the GP posterior mean of gradient and the mean function gradient
           * the GP posterior variance *of gradient* is only calculated pointwise, giving a 3x3 matrix at every centroid

        """

        print("Calculating posterior distribution of gradient...")

        # spectral density
        SD, _ = self.spectralDensity( self.HP[0], np.sqrt(self.Q) ) # NOTE: multiply SD by signal variance
        SD = self.HP[1] * SD

        # set outputs and inputs
        y = self.y
        D = self.yerr**2 + self.nugget

        # absorb SD into the eigenvectors
        # -------------------------------
        V = np.sqrt(SD)*self.V[self.vertex] # NOTE: absorb SD(eigenvalues) into the eigenvectors
        k_train = V.dot(V.T) + np.diag(D)
        invK = np.linalg.inv(k_train) # NOTE: direct inverse used in einsum calculations below

        # gradient kernel
        gradV = np.sqrt(SD)*self.gradV 
        a = gradV.reshape(-1,self.V.shape[1])
        gradKern = a.dot( (V).T ) # faster like this
        grad2Kern = np.einsum('ijk, ilk -> ijl', gradV, gradV)

        # posterior mean of gradient
        # --------------------------
        self.grad_post_mean = gradKern.dot(invK.dot(y)).reshape(-1,3)

        # posterior variance of gradient
        # ------------------------------
        zeta = np.inner(gradKern, invK.T) # much faster than doing einsum above
        zeta = np.einsum('klp , kjp -> klj', zeta.reshape(-1, 3, y.shape[0]), gradKern.reshape(-1, 3, y.shape[0])) # reshape for cross terms on different directions
        self.grad_post_var = grad2Kern - zeta

        return self.grad_meanFunction + self.unscale(self.grad_post_mean, std = True) # NOTE: not a standard deviation, but should not be shifted by data centering

    #}}}

    #{{{ gradient magnitude statistics
    def gradientStatistics(self, numSamples = 2000, centIdx = None):
        """Calculate statistics for magnitude of posterior gradients, returns mesh idx and these statistics."""
    
        if centIdx is not None:
            idx = centIdx
        else:
            # all centroids 
            idx = list(range(self.gradV.shape[0]))

        print("Calculating posterior distribution gradient magnitudes... (a bit slow...)")
        print("  (statistics: mean, stdev, 9th, 25th, 50th, 75th, 91st percentiles)")
        
        mag_stats = np.empty([len(idx), 7])

        if self.grad_post_mean is not None and self.grad_post_var is not None:

            for ii, i in enumerate(idx):

                meanF = 0 if self.grad_meanFunction.shape[0] == 1 else self.grad_meanFunction[i]

                mean, var = meanF + self.grad_post_mean[i], self.grad_post_var[i]

                # slower
                samples = np.random.multivariate_normal(mean, var, size = numSamples)
                mag_grad = np.linalg.norm(self.unscale(samples, std = True), axis = 1)
 
                #{{{ scatter of two compononets with histograms on axes
                if False:

                    left, width = 0.1, 0.65
                    bottom, height = 0.1, 0.65
                    spacing = 0.005

                    rect_scatter = [left, bottom, width, height]
                    rect_histx = [left, bottom + height + spacing, width, 0.2]
                    rect_histy = [left + width + spacing, bottom, 0.2, height]

                    # start with a square Figure
                    fig = plt.figure(figsize=(8, 8))

                    ax = fig.add_axes(rect_scatter)
                    ax_histx = fig.add_axes(rect_histx, sharex=ax)
                    ax_histy = fig.add_axes(rect_histy, sharey=ax)

                    # use the previously defined function
                    scatter_hist(samples[:,0] - mean[0], samples[:,1] - mean[1], ax, ax_histx, ax_histy)

                    plt.show()

                #}}}

                #{{{ 3d scatter of sample components
                if False:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(samples[:,0], samples[:,1], samples[:,2], c = mag_grad, cmap = "jet")
                    ax.scatter(mean[0], mean[1], mean[2], color = "black", s = 1000)
                    plt.show()
                #}}}
                
                #{{{ 3d vector field plot
                if False:
                    c = mag_grad
                    c = (c.ravel() - c.min()) / c.ptp()
                    c = np.concatenate((c, np.repeat(c, 2))) # Repeat for each body line and two head lines
                    c = plt.cm.jet(c) # Colormap

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.quiver(0,0,0,samples[:,0], samples[:,1], samples[:,2], colors = c)
                    ax.quiver(0,0,0,mean[0], mean[1], mean[2], color = "black")
                    plt.show()
                #}}}

                # statistics
                p = np.percentile(mag_grad, [9, 25, 50, 75, 91])
                m, s = np.mean(mag_grad), np.std(mag_grad)
                mag_stats[ii, 0] = m
                mag_stats[ii, 1] = s
                mag_stats[ii, 2:] = p

                #{{{ histogram of magnitudes
                if False:

                    plt.hist(mag_grad, bins = 30);
                    #plt.hist(mag_grad_chol, bins = 30, color = "red");
                    #plt.hist(mag_grad_eigh, bins = 30, color = "green");
                    for pi in p:
                        plt.axvline(pi, color = "pink")
                    plt.axvline(m, color = "red")
                    plt.axvline(m + 2*s, color = "red", linestyle = "--")
                    plt.axvline(m - 2*s, color = "red", linestyle = "--")
                    plt.show()
                #}}}

        return mag_stats
    #}}}


# subclasses implementing different models for the reduced-rank gpmi
# ==================================================================

#{{{ subclasses for different models

#{{{ Matern for GP, no mean function
class Matern(AbstractModel):
    """Model y = GP(x) + e where GP(x) is a manifold GP with Matern kernel."""       

    def kernelSetup(self, smoothness):

        self.smoothness = smoothness


    def spectralDensity(self, rho, w, grad = False):
        """Spectral Density for Matern kernel.
        
           w: sqrt(eigenvalues)
           rho: lengthscale parameter

        """
        
        D = 2  # dimension
        #v = 3.0/2.0  # smoothness
        v = self.smoothness  # smoothness

        alpha = (2**D * np.pi**(D/2) * gamma(v + (D/2)) * (2*v)**v) / gamma(v)
        delta = 1.0 / rho**(2*v)
        grad_delta = -2*v / rho**(2*v + 1)

        beta = 2*v / (rho**2)  + 4*(np.pi**2)*(w**2) 
        grad_beta = -4.0*v / (rho**3)

        expo = -(v + (D/2))
        result = (alpha*delta) * beta**expo

        if grad == True:

            grad_result = ( alpha*grad_delta * (beta**expo) ) \
                        + ( alpha*delta * expo*beta**(expo-1) * grad_beta )

            return result, grad_result

        else:

            return result, None


    def setMeanFunction(self, smooth = 10.0):
        """Set mean function."""
        self.meanFunction = np.array([0])
        self.grad_meanFunction = np.array([[0,0,0]])
#}}}

#}}}

