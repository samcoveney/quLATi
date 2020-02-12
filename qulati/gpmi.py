"""
   gpmi.py

   Module implementing Gaussian Process Manifold Interpolation (GPMI).

   Contains:
      class GPMI()

   Created: 03-Feb-2020
   Author:  Sam Coveney

"""

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

#}}}


class AbstractModel(ABC):
    """Gaussian Process Manifold Interpolation."""

    #{{{ decorators for checking methods arguments
    class Decorators(object):
        """Class of methods to use as decorators."""

        @classmethod
        def check_posterior_args(self, method):
            """Checks that posterior functions are supplied with correct arguments, and that optimize was called."""
            @wraps(method)
            def _wrapper(self, *args, **kwargs):

                errMsg = "\n[ERROR] {:s}: ".format(method.__name__)
                errRtn = "\n[return] None, None\n"

                s2pred = kwargs["s2pred"] if "s2pred" in kwargs else args[0]

                if (s2pred is None) and (self.s2 is not None):
                    print(errMsg + "s2pred not supplied." + errRtn)
                    return None, None

                if (s2pred is not None) and (self.s2 is None):
                    print(errMsg + "s2pred supplied but <set_data> did not receive s2." + errRtn)
                    return None, None

                if self.HP is None or self.nugget is None:
                    print(errMsg + "must call <optimize> before any predictions." + errRtn)
                    return None, None

                return method(self, *args, **kwargs)
            return _wrapper

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


    def getMeanFunction(self):
        """Return value of the mean function and gradient of mean function."""
        return self.meanFunction, self.grad_meanFunction

    
    # {{{ data handling including scalings
    def set_data(self, y, yerr, vertex, s2 = None):
        """Set data for interpolation, scaled and centred."""

        # set mean and stdev of y
        self.y_mean, self.y_std = np.mean(y), np.std(y)

        self.y = self.scale(y, std = False)
        self.yerr = self.scale(yerr, std = True)

        self.vertex = vertex
        self.s2 = s2

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

    #{{{ covariance matrix
    def kernelMatrix(self, HP, nugget, vertex, s2, yerr, grad = False):
        """Returns the kernel matrix between all supplied vertex locations.
        
           Arguments:
           HP -- hyperparameters
           nugget -- nugget to add to diagonal
           s2 -- values of s2 for each row
           yerr -- values of yerr for each row

           Keyword arguments:
           grad -- whether to return the matrix kernel derivatives wrt hyperparameters

           NOTES:
           * always builds a square matrix
           * even if HP includes the nugget value, the nugget value in 'nugget' is used
        
        """

        if grad: grad_cov = np.empty([self.vertex.shape[0], self.vertex.shape[0], HP.shape[0]])

        res = self.spectralDensity( HP[0], np.sqrt(self.Q), grad = grad )
        if grad: SD, gradSD = res[0], res[1]
        else: SD = res
        
        # gradient wrt lengthscale
        if grad: grad_cov[:,:,0] = HP[1] * self.V[vertex].dot(np.diag(gradSD)).dot(self.V[vertex].T)

        cov = self.V[vertex].dot(np.diag(SD)).dot(self.V[vertex].T)

        # gradient wrt variance
        if grad: grad_cov[:,:,1] = np.copy(cov)

        cov = HP[1]*cov
            
        # multiply kernel by RBF(s2)
        if s2 is not None:
            dists = pdist(s2/HP[-1], "sqeuclidean") # last HP is lengthscale for S2
            Ks2 = squareform( np.exp(-dists) )
            np.fill_diagonal(Ks2, 1.0)
            cov = np.multiply(cov, Ks2)
    
            # gradient wrt s2_lengthscale
            if grad: grad_cov[:,:,-1] = 2.0*(squareform(dists)/HP[-1])*cov

        # add nugget to diagonal
        np.fill_diagonal(cov, np.diag(cov) + nugget + yerr**2)

        # gradient wrt nugget
        if grad and self.nugget is None: grad_cov[:,:,2] = np.eye(self.vertex.shape[0])

        if grad:
            return cov, grad_cov
        else:
            return cov
    #}}}

    #{{{ negative loglikelihood
    # NOTE: if a nugget is provided to LLH via the optimizer use that value as the nugget (fixed), else take value from HP (train nugget)
    def LLH(self, guess, nugget, grad = False):
        """Return the negative loglikelihood.
        
           Arguments:
           guess -- log(hyperparameter values) for training
           nugget -- value of the nugget, which should be either a positive number (do not train) or None (train)
           grad -- return gradient of LLH wrt guess

           NOTES:
           * this function receives guess = log(HP), therefore derivatives wrt guess are returned as dLLH/dg = dLLH/dHP * dHP/dguess
           * the nugget passed to kernelMatrix will be either 'nugget' (if nugget is a number) or HP[2] (if nugget is None)
        
        """
        
        # FIXME: might be easier to set the stored values in the class, and have kernelMatrix use these
        HP = np.exp(guess)
        nug = nugget if nugget is not None else HP[2] # if training on the nugget, it is always HP[2]

        if grad == True:
            K, grad_K = self.kernelMatrix(HP, nug, self.vertex, self.s2, self.yerr, grad = grad)
        else:
            K = self.kernelMatrix(HP, nug, self.vertex, self.s2, self.yerr, grad = grad)

        try: # more stable

            L = linalg.cho_factor(K)        
            invK_y = linalg.cho_solve(L, self.y)
            logDetK = 2.0*np.sum(np.log(np.diag(L[0])))        

            cho_success = True

        except: # less stable, but almost always works

            cho_success = False

            try:
                invK = np.linalg.inv(K)
                invK_y = invK.dot(self.y)
                logDetK = np.log( linalg.det(K) )

            except np.linalg.linalg.LinAlgError as e:
                print("\n[WARNING]: Matrix not PSD for", HP, ", not fit.\n")

                return None
            except ValueError as e:
                print("\n[WARNING]: Ill-conditioned matrix for", HP, ", not fit.\n")
                return None
                
        llh = 0.5 * ( logDetK + ( self.y.T ).dot( invK_y ) + self.y.shape[0]*np.log(2*np.pi) )

        if grad == False:
            return llh
        else:
        
            # FIXME: something must be going wrong here, because cannot get check_grad to show a match
            grad_llh = np.empty(guess.size)
            for hp in range(guess.size):
                grad_hp = grad_K[:,:,hp]

                #if guess.size == 4: imShow(grad_hp)

                if cho_success:
                    invK_grad_hp = linalg.cho_solve(L, grad_hp)
                else:
                    invK_grad_hp = invK.dot(grad_hp)

                grad_llh[hp] = 0.5 * ( - (self.y.T).dot(invK_grad_hp).dot(invK_y) + np.trace(invK_grad_hp) )

                if ~np.isfinite(grad_llh[hp]):
                    print("\n[WARNING]: gradient(LLH) not finite", HP, ", not fit.\n")
                    return None

            return llh, grad_llh*HP

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

        fmt = lambda x: '+++++' if abs(x) > 1e3 else '-----' if abs(x) < 1e-3 else str(x)[:5] % x

        # initial guesses for hyperparameters
        dguess = np.random.uniform(np.log(1.0), np.log(100), size = restarts).reshape([1,-1]).T
        sguess = np.random.uniform(np.log(3), np.log(5), size = restarts).reshape([1,-1]).T
        nguess = np.random.uniform(np.log(1e-3), np.log(1e-2), size = restarts).reshape([1,-1]).T
        S2guess = np.random.uniform(np.log(10), np.log(100), size = restarts).reshape([1,-1]).T

        hdr = "Restart | " + " len " + " | " + " var " + " | "
        guess = np.hstack([dguess, sguess])

        self.nugget = nugget

        if nugget is None:  # nugget not supplied; we must train on nugget
            guess = np.hstack([guess, nguess])
            hdr = hdr + " nug " + " | " 
        else:
            nugget = np.abs(nugget)
        if self.s2 is not None:  # s2 values supplied; we must train with s2
            guess = np.hstack([guess, S2guess])
            hdr = hdr + " s2l " + " | " 

        # run optimization code
        print("Optimizing Hyperparameters...\n" + hdr)

        for ng, g in enumerate(guess):
            optFail = False
            try:
                bestStr = "   "
                res = minimize(self.LLH, g, args = (nugget, True), method = 'Newton-CG', jac=True) # make use of gradLLH
                #res = minimize(self.LLH, g, args = (nugget, False), method = 'Nelder-Mead') 

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
        self.nugget = self.HP[2] if nugget is None else nugget 

        print("\nHyperParams:", ", ".join(map(str, [fmt(i) for i in self.HP])), \
              "(nugget: {:1.3f})\n".format(self.nugget))

    #}}}

    #{{{ posterior distribution
    @Decorators.check_posterior_args
    def posterior(self, s2pred = None):
        """Calculates GP posterior mean and variance and returns *total* posterior mean and square root of pointwise variance
        
           Arguments:
           s2pred -- single value of s2 to use for prediction

           NOTES:
           * the returned *total* posterior mean is the sum of the GP posterior mean and the mean function

        """

        print("Calculating posterior distribution...")

        # make covariance for data
        covTrain = self.kernelMatrix(self.HP, self.nugget, self.vertex, self.s2, self.yerr)
        L = linalg.cho_factor(covTrain)

        if s2pred is not None: # multiPacing prediction
            
            print("predicting for s2pred: {:0.3f}".format(s2pred))
            s2pred = np.array([[s2pred]])

            # stack (mesh vertices, data vertices) and (s2pred, data s2) and make a big covariance matrix 
            fullInds = np.hstack( [np.arange(self.V.shape[0]), self.vertex] )
            fullS2 = np.vstack([np.repeat(s2pred, self.V.shape[0])[:,None], self.s2])
            covBig = self.kernelMatrix(self.HP, 0, fullInds, fullS2, 0.0) # NOTE: nugget = 0

            # slice out the cross covariance and prediction covariance
            crossCov = covBig[0:self.V.shape[0]][:, -self.vertex.shape[0]:]  # rows: all mesh vertices; cols: data vertices
            predCov = covBig[0:self.V.shape[0]][:, 0:self.V.shape[0]]

        else: # singlePacing prediction

            # prediction covar: all mesh locations
            predCov = self.kernelMatrix(self.HP, 0, np.arange(self.V.shape[0]), None, 0.0) # NOTE: nugget = 0
            crossCov = predCov[:, self.vertex]


        # calculate GP posterior
        Ef = crossCov.dot( linalg.cho_solve(L, self.y) )
        Vf = predCov - crossCov.dot( linalg.cho_solve(L, crossCov.T) )
        self.post_mean, self.post_var = Ef, Vf

        return self.meanFunction + self.unscale(Ef), self.unscale(np.sqrt(np.diagonal(Vf)), std = True)
    #}}}

    #{{{ posterior gradient
    @Decorators.check_posterior_args
    def posteriorGradient(self, s2pred = None):
        """Calculates GP posterior mean and variance *of gradient* and returns posterior mean and square root of pointwise variance
        
           Arguments:
           s2pred -- single value of s2 to use for prediction
        
           NOTES:
           * the returned *total* posterior mean *of gradient* is the sum of the GP posterior mean of gradient and the mean function gradient
           * the GP posterior variance *of gradient* is only calculated pointwise, giving a 3x3 matrix at every vertex

        """

        print("Calculating posterior distribution of gradient...")

        # make covariance for data
        covTrain = self.kernelMatrix(self.HP, self.nugget, self.vertex, self.s2, self.yerr)
        L = linalg.cho_factor(covTrain)

        # gradient of kernel at all mesh vertices
        # ---------------------------------------
        #print("Calculating kernel spatial gradient...")
        SD = self.HP[1] * self.spectralDensity( self.HP[0], np.sqrt(self.Q) )
        beta = np.einsum('i, ijk -> ijk', SD, self.gradV)
        gradKern = np.einsum('ijk, li -> kjl', beta, self.V[self.vertex]) 
        #print("gradKern.shape:", gradKern.shape)

        if self.s2 is not None:
        
            print("predicting for s2pred: {:0.3f}".format(s2pred))
            s2pred = np.array([[s2pred]])

            dists = cdist(self.s2/self.HP[-1], s2pred/self.HP[-1], "sqeuclidean") # last HP is lengthscale for S2
            Ks2 = np.exp(-dists).flatten()
            gradKern = np.einsum("k, ijk -> ijk", Ks2, gradKern)

        # posterior mean of gradient
        # --------------------------
        alpha = linalg.cho_solve(L, self.y)  #  K(X,X)^-1 y
        v = np.einsum('kjl, l -> kj', gradKern, alpha)
        #print("grad posterior mean shape:", v.shape)
       
        # posterior variance of gradient
        # ------------------------------
        # vVar = grad2Kern - zeta, where zeta = gradK K^-1 gradK

        grad2Kern = np.einsum('i, ijk, ilk -> kjl', SD, self.gradV, self.gradV)

        invK = linalg.cho_solve(L, np.identity(covTrain.shape[0]))

        # doing zeta = np.einsum('klm , mp , kjp -> klj', gradKern, invK, gradKern) in two stages; much faster
        zeta = np.einsum('klm , mp -> klp', gradKern, invK) 
        zeta = np.einsum('klp , kjp -> klj', zeta, gradKern)

        vVar = grad2Kern - zeta
        #print("grad posterior variance shape:", vVar.shape)

        self.grad_post_mean, self.grad_post_var = v, vVar

        #print("\n[WARNING] the variance of posterior gradient is a 3x3 matrix for each vertex.\n")
        return self.grad_meanFunction + self.unscale(v, std = True) # ,  self.unscale(self.unscale(vVar, std = True)) # FIXME: it is not clear how to return UQ on CV at the moment

    #}}}

    #{{{ posterior samples
    @Decorators.check_posterior_exists
    def posteriorSamples(self, num):
        """Returns posterior samples from the whole model (samples from the posterior + mean function)
        
        Arguments:
        num -- how many posterior samples to calculate
        
        """

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

    #{{{ posteriorStatistics
    def gradientStatistics(self, electrodes = False, s2pred = None):
        """Calculate statistics for magnitude of posterior gradients, returns mesh idx and these statistics."""
    
        
        if electrodes == False: # all vertices
            idx = list(range(self.V.shape[0]))
            prnt = "" # "for all vertices"
        else:
            if s2pred is not None: # all electrodes where we have an observation for the s2 we are predicting at
                idx = self.vertex[self.s2.flatten() == s2pred]
                prnt = " for all electrodes with observation for s2pred"
            else:
                idx = self.vertex # all electrodes
                prnt = " for all electrodes"

        print("Calculating posterior distribution gradient magnitudes{:s}...".format(prnt))
        print("  (statistics: mean, stdev, 9th, 25th, 50th, 75th, 91st percentiles)")
        
        mag_stats = np.empty([len(idx), 7])

        if self.grad_post_mean is not None and self.grad_post_var is not None:

            for ii, i in enumerate(idx):

                meanF = 0 if self.grad_meanFunction.shape[0] == 1 else self.grad_meanFunction[i]

                mean, var = meanF + self.grad_post_mean[i], self.grad_post_var[i]

                samples = np.random.multivariate_normal(mean, var, size = 2000)

                mag_grad = np.linalg.norm(self.unscale(samples, std = True), axis = 1)

                p = np.percentile(mag_grad, [9, 25, 50, 75, 91])
                m, s = np.mean(mag_grad), np.std(mag_grad)
                mag_stats[ii, 0] = m
                mag_stats[ii, 1] = s
                mag_stats[ii, 2:] = p

                if False:

                    plt.hist(mag_grad, bins = 30);
                    for pi in p:
                        plt.axvline(pi, color = "pink")
                    plt.axvline(m, color = "red")
                    plt.axvline(m + 2*s, color = "red", linestyle = "--")
                    plt.axvline(m - 2*s, color = "red", linestyle = "--")
                    plt.show()

        return mag_stats
    #}}}

    #{{{screePlot
    def screePlot(self):
    
        if self.HP is None:
            print("\n[ERROR]: must call optimize before <screePlot> is called\n")
            return

        val = self.spectralDensity(np.sqrt(self.Q), self.HP[0])
        cumsum = np.cumsum(val)/ np.sum(val)

        for val in [0.95, 0.997]:

            plt.axhline(val)

            try:
                cumsum_where = np.argwhere(cumsum > val).min()
                plt.axvline(cumsum_where)
                print("{:1.3f} variance explained with {:d} eigenfunctions".format(val, cumsum_where + 1))
            except:
                pass

        plt.scatter(np.arange(cumsum.shape[0]), cumsum, color = "red")
        plt.title("Scree Plot of explained variance")
        plt.show()
    #}}}


class Matern52(AbstractModel):
    """Model y = GP(x) + e where GP(x) is a manifold GP with Matern52 kernel."""       

    def spectralDensity(self, rho, w, grad = False):
        """Spectral Density for Matern kernel.
        
           w: sqrt(eigenvalues)
           rho: lengthscale parameter

        """
        
        D = 2  # dimension
        v = 5.0/2.0  # smoothness

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

            return result

        # hard wired simpler form for v = 5.2 and D = 2; does not seem to give any speed ups at all
        #a = np.sqrt(5.)/rho
        #result_2 = 10.*np.pi*a**5 * ( a**2 + 4*np.pi**2*w**2 )**(-7/2)


    def setMeanFunction(self, smooth = 10.0):
        """Set mean function."""
        self.meanFunction = np.array([0])
        self.grad_meanFunction = np.array([[0,0,0]])


class splinesAndMatern52(Matern52): # NOTE: inherets Matern52 class but reimplements setMeanFunction
    """Model y = m(x) + GP(x) + e where m(x) are splines on the manifold."""       

    def setMeanFunction(self, smooth = 10.0):
        """Manifold splines using Laplacian eigenfunctions f(x) = f0 + sum_k=1..N { beta_k * G(x, x_k) }
           
           NOTES:
           * this mean is in the original units. It therefore adjusts scalings so the GP has centred and scaled residuals to fit.
           * the stored y data is modified, such that it is the residuals after fitting this mean
           * G does not include the first eigenpair

        """

        print("Calculating mean function with manifold splines...")

        # unscale y data into original units
        self.y = self.unscale(self.y)
        self.yerr = self.unscale(self.yerr, std = True)

        # 1. solve for coefficients f0 and beta
        # NOTE: could optimize for "smooth" parameter

        G = self.V[self.vertex, 1:].dot(np.diag(1.0/self.Q[1:])).dot(self.V[self.vertex,1:].T) + smooth*np.eye(self.y.shape[0])
        res = np.linalg.lstsq( np.hstack([G, np.ones(G.shape[0])[:,None]]), self.y, rcond = None ) # NOTE: used least squares, ignored <e, beta> = 0
        beta, f0 = res[0][0:-1], res[0][-1]

        # 2. make prediction at all vertices and save result

        G = self.V[:, 1:].dot(np.diag(1.0/self.Q[1:])).dot(self.V[self.vertex,1:].T)
        self.meanFunction = f0 + beta.dot(G.T)

        gradG = np.einsum( "ijk, li -> klj" , self.gradV[1:, :, :], ((1.0/self.Q[1:])*self.V[self.vertex,1:]) ) 
        self.grad_meanFunction = np.dot(beta, gradG)

        # rescale y data
        y = self.y - self.meanFunction[self.vertex]
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y, self.yerr = self.scale(y), self.scale(self.yerr, std = True)


class basisAndMatern52(Matern52): # NOTE: inherets Matern52 class but reimplements setMeanFunction
    """Model y = m(x) + GP(x) + e where m(x) is regression with eigenfunctions basis.

       NOTES:
       * this mean is in the original units. It therefore adjusts scalings so the GP has centred and scaled residuals to fit.
       * the stored y data is modified, such that it is the residuals after fitting this mean
       * the regression basis does not include the first eigenpair

    """

    def setMeanFunction(self, num = 2**2):
        """Regression y = f0 + beta*eigenfunctions( j = 1 .. 1 + num )..."""

        print("Calculating mean function with as regression using eigenfunction basis...")

        # unscale y data into original units
        self.y = self.unscale(self.y)
        self.yerr = self.unscale(self.yerr, std = True)

        res = np.linalg.lstsq( np.hstack( [ self.V[self.vertex, 1:1+num], np.ones(self.y.shape[0])[:,None] ] ), self.y, rcond = 0 )
        beta, f0 = res[0][0:-1], res[0][-1]

        plt.plot(beta); plt.show()

        self.meanFunction = f0 + np.dot(beta, self.V[:,1:1+num].T)
        self.grad_meanFunction = np.einsum("i, ijk -> kj" , beta, self.gradV[1:1+num,:,:]) 

        # rescale y data
        y = self.y - self.meanFunction[self.vertex]
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y, self.yerr = self.scale(y), self.scale(self.yerr, std = True)


