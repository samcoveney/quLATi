"""
   gpmi.py

   Implementation of Gaussian Process Manifold Interpolation (GPMI).

   Contains:
      class AbstractModel
      class Matern(AbstractModel)

   Created: 05-Feb-2021
   Author:  Sam Coveney

"""

#{{{ module imports
from abc import ABC, abstractmethod
from functools import wraps

import numpy as np
from scipy import linalg
from scipy.optimize import minimize
from scipy.special import gamma
#}}}


# abstract class
# ==============

class AbstractModel(ABC):
    """ Gaussian Process Manifold Interpolation."""

    #{{{ everything in common: init, del, reset, and abstract methods
    def __init__(self, X, Tri, Q, V, gradV, JAX = False):
        """Initialize class with manifold data."""

        self.X = X
        self.Tri = Tri
        self.Q = Q
        self.V = V
        self.gradV = gradV

        self.reset()

        # convenient way to choose between JAX and non-JAX, so that jax can remain a soft dependency
        if JAX == True:
            try:
                from .likelihood_jax import LLH, jit, value_and_grad
                self.JAX = True
                AbstractModel.LLH = LLH
                
                # To use jax with lbfgs, wrap returns and cast to numpy array - https://github.com/google/jax/issues/1510
                newllh = value_and_grad(self.LLH)
                self.LLH = lambda x, y: [ np.array(val) for val in jit(newllh)(x, y) ]

            except ImportError as e:
                JAX = False
                print("[ERROR]:", e)

        if JAX == False:
            from .likelihood import LLH
            self.JAX = False
            AbstractModel.LLH = LLH


        super().__init__()


    def __del__(self):
        """Method to cleanup when we delete the instance of this class."""
        pass


    def reset(self):
        """Resets parameters and predictions."""

        self.HP, self.nugget = None, None
        self.post_mean, self.post_var = None, None
        self.grad_post_mean, self.grad_post_var = None, None


    @abstractmethod
    def spectralDensity(self, rho):
        """Spectral Density function."""
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
        """Scales and centers y data."""
        if std == False:
            return (y - self.y_mean) / self.y_std
        else:
            return (y / self.y_std)


    def unscale(self, y, std = False):
        """Un-scales and un-centers y data into original scale."""
        if std == False:
            return (y * self.y_std) + self.y_mean
        else:
            return (y * self.y_std)
    #}}}

    #{{{ optimize the hyperparameters
    def optimize(self, nugget, restarts):
        """Optimize the hyperparameters.
        
           Arguments:
           nugget -- value of the nugget, if None then train nugget, if number then fix nugget
           restart -- how many times to restart the optimizer.

           NOTES:
           * values for hyperparameters for training are supplied as log(HP) to the loglikelihood function
           * currently using Newton-CG method needing LLH gradients, hopefully these LLH gradients wrt log(HP) are correct...
           * initial guesses are picked at random from predefined ranges set in this function
        
        """

        # print formatting
        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str(x)[:5] % x
        hdr = "Restart | " + " len " + " | " + " var " + " | "

        # initial guesses for hyperparameters
        dguess = np.random.uniform(np.log(10), np.log(100), size = restarts).reshape([1,-1]).T
        sguess = np.random.uniform(np.log(10), np.log(100), size = restarts).reshape([1,-1]).T
        guess = np.hstack([dguess, sguess])

        # nugget
        if nugget is None:  # nugget not supplied; we must train on nugget
            fixed_nugget = 0.0
            nguess = np.random.uniform(np.log(1e-4), np.log(1e-2), size = restarts).reshape([1,-1]).T
            guess = np.hstack([guess, nguess])
            hdr = hdr + " nug " + " | " 
        else:
            fixed_nugget = np.abs(nugget)

        # run optimization code
        print("Optimizing hyperparameters...")

        for ng, g in enumerate(guess):
            optFail = False
            try:
                bestStr = "   "
                if self.JAX == True:
                    res = minimize(self.LLH, g, args = (fixed_nugget), method = 'L-BFGS-B', jac = True) # for jax with joint func and grad
                else:
                    res = minimize(self.LLH, g, args = (fixed_nugget), method = 'Nelder-Mead')
                #print("res:", res)

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
                
                if ng == 0: print(hdr)
                print(" {:02d}/{:02d} ".format(ng + 1, restarts),
                      "| %s" % ' | '.join(map(str, [fmt(i) for i in np.exp(res.x)])),
                      "| {:s} llh: {:.3f}".format(bestStr, -1.0*np.around(res.fun, decimals=4)))

            except TypeError as e:
                optFail = True

        # save results of optimization
        try:
            if nugget is None:
                self.HP = np.exp(bestRes.x[:-1])
                self.nugget = np.exp(bestRes.x[-1])
            else:
                self.HP = np.exp(bestRes.x)
                self.nugget = fixed_nugget
            
            print("\nHyperParams:", ", ".join(map(str, [fmt(i) for i in self.HP])), \
                  "\n  (nugget: {:f})".format(self.nugget))

        except UnboundLocalError as e:
            print("ERROR:", e)
            print("This probably means that the optimization failed.")
            self.HP, self.nugget = np.array([np.nan, np.nan]), np.nan

    #}}}

    #{{{ posterior distribution
    def posterior(self, indices = None, pointwise = True):
        """Calculates GP posterior mean and variance and returns *total* posterior mean and square root of pointwise variance

           NOTES:
           * the returned *total* posterior mean is the sum of the GP posterior mean and the mean function

        """

        # FIXME: Need to implement a way to calculate the posterior for a subset of vertices (usefully, for first self.X.shape[0] positions i.e. vertices)
        #        This is because we will want to take posterior samples, and so need the full posterior covariance

        # spectral density
        SD = self.spectralDensity( self.HP[0] ) # NOTE: multiply SD by signal variance
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
            return self.unscale(Ef), self.unscale(np.sqrt(Vf), std = True)
        else: 
            return self.unscale(Ef), self.unscale(np.sqrt(np.diag(Vf)), std = True)

    #}}}

    #{{{ posterior samples
    def posteriorSamples(self, num, nugget = 1e-10):
        """Returns posterior samples from the whole model (samples from the posterior + mean function)
        
        Arguments:
        num -- how many posterior samples to calculate
        
        """

        if self.post_var is None or self.post_mean is None:
            print("[ERROR]: need to call posterior first, and succeed in saving full posterior covariance.")

        if self.post_var.ndim != 2:
            print("[ERROR]: require full posterior variance to calculate samples")

        print("Calculating {:d} samples from posterior distribution...".format(num))

        print("WARNING: vertices only, not centroids.")
        N = self.X.shape[0]
        PM, PV = self.post_mean[0:N].copy(), self.post_var[0:N][:,0:N].copy()

        while True:
            try:
                print("  (adding nugget {:1.0e} to posterior var diagonal for stability)".format(nugget))
                np.fill_diagonal(PV, np.diag(PV) + nugget)
                #print("  Cholesky decomp")
                L = np.linalg.cholesky(PV)
                break
            except:
                nugget = nugget * 10
        
        #print("Calculate samples")
        u = np.random.randn(N, num)
        samples = PM[:,None] + L.dot(u)

        return self.unscale(samples)
        
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
        SD = self.spectralDensity( self.HP[0] ) # NOTE: multiply SD by signal variance
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

        return self.unscale(self.grad_post_mean, std = True) # NOTE: not a standard deviation, but should not be shifted by data centering

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

                mean, var = self.grad_post_mean[i], self.grad_post_var[i]

                # slower
                samples = np.random.multivariate_normal(mean, var, size = numSamples)
                mag_grad = np.linalg.norm(self.unscale(samples, std = True), axis = 1)

                # statistics
                p = np.percentile(mag_grad, [9, 25, 50, 75, 91])
                m, s = np.mean(mag_grad), np.std(mag_grad)
                mag_stats[ii, 0] = m
                mag_stats[ii, 1] = s
                mag_stats[ii, 2:] = p

        return mag_stats
    #}}}


# subclasses implementing different models
# ========================================

#{{{ Matern kernel
class Matern(AbstractModel):
    """Model y = GP(x) + e where GP(x) is a manifold GP with Matern kernel."""       

    def kernelSetup(self, smoothness):
        """Pre calculations for Matern kernel."""

        self.smoothness = smoothness

        D = 2  # dimension
        self.const1 = (2**D * np.pi**(D/2) * gamma(self.smoothness + (D/2)) * (2*self.smoothness)**self.smoothness) / gamma(self.smoothness)
        self.const2 = 4*(np.pi**2)*(self.Q) # Q replaces w**2 where w = sqrt(q)
        self.const3 = -(self.smoothness + (D/2))


    def spectralDensity(self, rho):
        """Spectral Density for Matern kernel."""

        beta = 2*self.smoothness / (rho**2)  + self.const2 
        result = (self.const1/rho**(2*self.smoothness)) * beta**self.const3

        return result

#}}}

