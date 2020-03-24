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

                #s2pred = kwargs["s2pred"] if "s2pred" in kwargs else args[0]
                if "s2pred" in kwargs:
                    s2pred = kwargs["s2pred"]
                else:
                    try:
                        s2pred = args[0]
                    except:
                        s2pred = None

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

        self.gradnugget = None

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

    #}}}
    
    # {{{ s2 kernel function
    # should be an abstract method that is implemented elsewhere
    def s2kernel(self, s2i , s2j, l, grad = False):
        """Covariance between s2 values: fixed as RBF kernel for now."""
           
        dists = cdist(s2i[:,None]/l, s2j[:,None]/l, "sqeuclidean") # last HP is lengthscale for S2
        Ks2 = np.exp(-dists)

        # gradient wrt s2_lengthscale
        if grad:
            gradKs2 = 2.0*(dists/l)*Ks2
            return Ks2, gradKs2
        else:
            return Ks2, None
    #}}}

    # {{{ data handling including scalings
    def set_data(self, y, yerr, vertex, grady = np.empty([0,3]), gradyerr = np.empty([0]), gradvertex = np.empty([0], dtype = int), s2 = None, grads2 = np.empty([0])):
        """Set data for interpolation, scaled and centred."""

        # set mean and stdev of y
        self.y_mean, self.y_std = np.mean(y), np.std(y)

        self.y = self.scale(y, std = False)
        self.yerr = self.scale(yerr, std = True)
        self.grady = self.scale(grady, std = True) # careful not to subtract observation mean from observation gradients!
        self.gradyerr = self.scale(gradyerr, std = True)

        self.vertex = vertex
        self.gradvertex = gradvertex

        self.s2 = s2
        self.grads2 = grads2

        #print(self.y)
        #print(self.yerr) 
        #print(self.grady)
        #print(self.gradyerr)
        #input()

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
    def makeCovar(self, type, grad = False, N = None):
        """Returns the kernel matrix of the specified type.
        
           Arguments:
           type -- training covar, prediction covar, or cross covar

           Keyword arguments:
           grad -- whether to return the matrix kernel derivatives wrt hyperparameters

           NOTES:
           * nugget values passed to the kernel function will be set here depending on type
           * when using self.s2, currently prediction is only for a single specified s2 called self.s2pred
           * gradient observations do not work when self.s2 is not None at the moment
             -> for s2 values for gradient observations, will need to repeat self.grads2 3 times, so that grads2 is supplied for every component of the gradient
        
        """

        if type == "training":

            a = np.vstack( [ self.V[self.vertex], self.gradV[self.gradvertex,:,:].reshape(-1,self.V.shape[1]) ] )
            b = a

            if self.s2 is not None:
                a_s2 = np.hstack([self.s2, self.grads2])
            else:
                a_s2 = None
            b_s2 = a_s2

            # NOTE: error is properly prepared here
            err = np.hstack([self.yerr**2 + self.nugget, self.gradyerr.flatten()**2 + self.gradnugget]) 


        if type == "prediction":

            a = self.V if N is None else self.V[0:N]
            b = a

            a_s2 = None
            b_s2 = a_s2


        if type == "prediction_pointwise":
            # NOTE: exception, just calculate pointwise variance for K(X*, X*), don't need the cross terms

            SD, _ = self.spectralDensity( self.HP[0], np.sqrt(self.Q) )

            a = self.V
            b = a

            temp = (SD*a*b).sum(axis = 1)

            return self.HP[1]*temp


        if type == "cross":

            a = self.V if N is None else self.V[0:N]

            b = np.vstack( [ self.V[self.vertex], self.gradV[self.gradvertex,:,:].reshape(-1,self.V.shape[1]) ] )

            if self.s2 is not None:
                a_s2 = np.repeat(self.s2pred, self.V.shape[0]) 
                b_s2 = np.hstack([self.s2, self.grads2])
            else:
                a_s2 = None
                b_s2 = None



        if type == "grad_cross":
        
            a = self.gradV.reshape(-1,self.V.shape[1])
            b = np.vstack( [ self.V[self.vertex], self.gradV[self.gradvertex,:,:].reshape(-1,self.V.shape[1]) ] )


            if self.s2 is not None:
                a_s2 = np.repeat(self.s2pred, a.shape[0]) 
                b_s2 = np.hstack([self.s2, self.grads2])
            else:
                a_s2 = None
                b_s2 = None


        if type  == "grad_grad_prediction":
            # NOTE: exception, do not calclate the full posterior variance for derivative predictions, it is too big! Just pointwise.

            SD, _ = self.spectralDensity( self.HP[0], np.sqrt(self.Q) )
            #temp = np.einsum('i, ijk -> ijk', SD, self.gradV.T)
            #temp = np.einsum('ijk, ilk -> kjl', temp, self.gradV.T)
            #temp = np.einsum('j, ijk -> ijk', SD, self.gradV)
            #temp = np.einsum('ijk, ljk -> kil', temp, self.gradV)

            #temp = np.einsum('k, ijk -> ijk', SD, self.gradV)
            temp = SD * self.gradV
            temp = np.einsum('ijk, ilk -> ijl', temp, self.gradV)
            #temp = np.inner(temp, self.gradV) # probably correct, but memory error

            return self.HP[1]*temp
        

        # nuggets and observations errors for training covariance only
        if type != "training": err = 0

        # call to create the kernel matrix
        cov, grad_cov = self.kernelMatrix(a, b, a_s2, b_s2, err, grad = grad)

        if grad:
            return cov, grad_cov
        else:
            return cov
    #}}}

    #{{{ kernel matrix
    def kernelMatrix(self, a, b, a_s2, b_s2, err, grad = False):
        """Creates matrix of kernel entries, and gradient of entries wrt hyperparameters if requiered.
        
           NOTES:
           * kernelMatrix always uses the value passed in nugget as the nugget, so that a zero nugget can be passed for non-training-data covariances.
        
        """

        # for storing grad_kernel results
        if grad: grad_cov = np.empty([a.shape[0], b.shape[0], self.HP.shape[0]])

        # evaluate the spectral density
        SD, gradSD = self.spectralDensity( self.HP[0], np.sqrt(self.Q), grad = grad )
        
        # gradient wrt lengthscale
        if grad: grad_cov[:,:,0] = self.HP[1] * a.dot(np.diag(gradSD)).dot(b.T)

        #cov = a.dot(np.diag(SD)).dot(b.T)
        cov = a.dot( (b*SD).T ) # faster like this

        # gradient wrt variance
        if grad: grad_cov[:,:,1] = np.copy(cov)

        cov = self.HP[1]*cov
            
        # multiply kernel by RBF(s2)
        if self.s2 is not None:
    
            Ks2, gradKs2 = self.s2kernel(a_s2, b_s2, self.HP[-1], grad = grad) # returns Ks2, gradKs2

            # gradient wrt s2l
            if grad: grad_cov[:,:,-1] = gradKs2 * cov

            cov = Ks2 * cov

        # add nugget to diagonal
        np.fill_diagonal(cov, np.diag(cov) + err)

        # gradient wrt nugget # NOTE: added test to make sure we only add this on K_x_x and K_gx_gx
        if grad and self.nugget_train: grad_cov[:,:,self.nugget_index] = np.diag( np.hstack( [ np.ones(self.y.shape[0]), np.zeros(self.grady.flatten().shape[0]) ] ) )  # old: np.eye(a.shape[0])
        if grad and self.gradnugget_train: grad_cov[:,:,self.gradnugget_index] = np.diag( np.hstack( [ np.zeros(self.y.shape[0]), np.ones(self.grady.flatten().shape[0]) ] ) )

        if grad:
            return cov, grad_cov
        else:
            return cov, None
    #}}}

    #{{{ negative loglikelihood
    # NOTE: if a nugget is provided to LLH via the optimizer use that value as the nugget (fixed), else take value from HP (train nugget)
    def LLH(self, guess, grad = False):
        """Return the negative loglikelihood.
        
           Arguments:
           guess -- log(hyperparameter values) for training
           grad -- return gradient of LLH wrt guess

           NOTES:
           * this function receives guess = log(HP), therefore derivatives wrt guess are returned as dLLH/dg = dLLH/dHP * dHP/dguess
        
        """
        
        # FIXME: might be easier to set the stored values in the class, and have kernelMatrix use these
        self.HP = np.exp(guess)
        if self.nugget_train: self.nugget = self.HP[self.nugget_index]
        if self.gradnugget_train: self.gradnugget = self.HP[self.gradnugget_index]

        if grad == True:
            K, grad_K = self.makeCovar("training", grad = grad)
        else:
            K = self.makeCovar("training", grad = grad)

        y = self.y if self.grady is None else np.hstack([self.y, self.grady.flatten()]) 

        try: # more stable

            L = linalg.cho_factor(K)        
            invK_y = linalg.cho_solve(L, y)
            logDetK = 2.0*np.sum(np.log(np.diag(L[0])))        

            cho_success = True

        except: # less stable, but almost always works

            cho_success = False

            try:
                invK = np.linalg.inv(K)
                invK_y = invK.dot(y)
                logDetK = np.log( linalg.det(K) )

            except np.linalg.linalg.LinAlgError as e:
                print("\n[WARNING]: Matrix not PSD for", self.HP, ", not fit.\n")

                return None
            except ValueError as e:
                print("\n[WARNING]: Ill-conditioned matrix for", self.HP, ", not fit.\n")
                return None
                
        llh = 0.5 * ( logDetK + ( y.T ).dot( invK_y ) + y.shape[0]*np.log(2*np.pi) )  # NOTE: I'm not entirely sure about the constant when we have derivatives

        if grad == False:
            return llh
        else:
        
            grad_llh = np.empty(guess.size)
            for hp in range(guess.size):
                grad_hp = grad_K[:,:,hp]

                if cho_success:
                    invK_grad_hp = linalg.cho_solve(L, grad_hp)
                else:
                    invK_grad_hp = invK.dot(grad_hp)

                grad_llh[hp] = 0.5 * ( - (y.T).dot(invK_grad_hp).dot(invK_y) + np.trace(invK_grad_hp) )

                if ~np.isfinite(grad_llh[hp]):
                    print("\n[WARNING]: gradient(LLH) not finite", self.HP, ", not fit.\n")
                    return None

            return llh, grad_llh*self.HP # dLLH/d_guess = dLLH/d_HP * d_HP/d_guess = dLLH/d_HP * HP

    #}}}

    #{{{ optimize the hyperparameters
    def optimize(self, nugget, restarts, gradnugget = 0.0001):
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
        gnguess = np.random.uniform(np.log(1e-3), np.log(1e-2), size = restarts).reshape([1,-1]).T
        S2guess = np.random.uniform(np.log(10), np.log(100), size = restarts).reshape([1,-1]).T

        hdr = "Restart | " + " len " + " | " + " var " + " | "
        guess = np.hstack([dguess, sguess])

        # nugget
        self.nugget = nugget

        if nugget is None:  # nugget not supplied; we must train on nugget
            self.nugget_train = True
            guess = np.hstack([guess, nguess])
            self.nugget_index = guess.shape[1] - 1
            hdr = hdr + " nug " + " | " 
        else:
            nugget = np.abs(nugget)
            self.nugget_train = False

        # gradnugget
        if self.gradvertex.shape[0] == 0: gradnugget = 0.0 
        self.gradnugget = gradnugget

        if gradnugget is None:  # gradnugget not supplied; we must train on gradnugget
            self.gradnugget_train = True
            guess = np.hstack([guess, gnguess]) # NOTE: currently using same guess as nugget
            self.gradnugget_index = guess.shape[1] - 1
            hdr = hdr + " gnu " + " | " 
        else:
            gradnugget = np.abs(gradnugget)
            self.gradnugget_train = False

        # s2 lengthscale
        if self.s2 is not None:  # s2 values supplied; we must train with s2
            guess = np.hstack([guess, S2guess])
            hdr = hdr + " s2l " + " | " 

        # run optimization code
        print("Optimizing Hyperparameters...\n" + hdr)

        for ng, g in enumerate(guess):
            optFail = False
            try:
                bestStr = "   "
                #res = minimize(self.LLH, g, args = (True), method = 'Newton-CG', jac=True) # make use of gradLLH
                res = minimize(self.LLH, g, args = (False), method = 'Nelder-Mead') 

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
        self.gradnugget = self.HP[self.gradnugget_index] if gradnugget is None else gradnugget 

        print("\nHyperParams:", ", ".join(map(str, [fmt(i) for i in self.HP])), \
              "\n  (nugget: {:f})".format(self.nugget),
              "\n  (gradnugget: {:f})\n".format(self.gradnugget))

    #}}}

    #{{{ posterior distribution
    @Decorators.check_posterior_args
    def posterior(self, N = None, s2pred = None):
        """Calculates GP posterior mean and variance and returns *total* posterior mean and square root of pointwise variance
        
           Arguments:
           s2pred -- single value of s2 to use for prediction

           NOTES:
           * the returned *total* posterior mean is the sum of the GP posterior mean and the mean function

        """

        # FIXME: sort out how s2pred is handled
        self.s2pred = s2pred

        # make covariance matrices
        if N is None:
            print("Calculating pointwise posterior distribution...")
            predCov  = self.makeCovar("prediction_pointwise")
        else:
            print("Calculating full posterior distribution for first {:d} vertices...".format(N))
            # the point that be specifying N = X.shape[0], we can calculate this for the vertices only
            predCov  = self.makeCovar("prediction", N = N)

        covTrain = self.makeCovar("training")
        crossCov = self.makeCovar("cross", N = N)

        # NOTE: y must be prepared base on whether there are derivative observations or not
        y = self.y if self.grady is None else np.hstack([self.y, self.grady.flatten()]) 

        # calculate GP posterior mean
        L = linalg.cho_factor(covTrain)
        Ef = crossCov.dot( linalg.cho_solve(L, y) )
        self.post_mean = Ef

        # calculate GP posterior variance
        if N is None:
            temp = linalg.cho_solve(L, crossCov.T)
            temp = np.einsum("ij, ji -> i", crossCov, temp) # if I transpose the K^-1 y part, will the cross term only be pointwise?
            #temp = np.inner(crossCov, temp.T) # I'm quite sure this is correct, but memory error for large meshes
            Vf = predCov - temp

            return self.meanFunction + self.unscale(Ef), self.unscale(np.sqrt(Vf), std = True)

        else:
            
            Vf = predCov - crossCov.dot( linalg.cho_solve(L, crossCov.T) )

            self.post_var = Vf  # NOTE: only saved internal if a full posterior matrix was created, for first N vertices specified by N argument

            return self.meanFunction + self.unscale(Ef), self.unscale(np.sqrt(np.diagonal(Vf)), std = True)
    #}}}

    #{{{ posterior distribution of gradient
    @Decorators.check_posterior_args
    def posteriorGradient(self, s2pred = None):
        """Calculates GP posterior mean and variance *of gradient* and returns posterior mean and square root of pointwise variance
        
           Arguments:
           s2pred -- single value of s2 to use for prediction
        
           NOTES:
           * the returned *total* posterior mean *of gradient* is the sum of the GP posterior mean of gradient and the mean function gradient
           * the GP posterior variance *of gradient* is only calculated pointwise, giving a 3x3 matrix at every vertex

        """

        self.s2pred = s2pred

        print("Calculating posterior distribution of gradient...")

        covTrain = self.makeCovar("training")
        y = self.y if self.grady is None else np.hstack([self.y, self.grady.flatten()]) 
        gradKern = self.makeCovar("grad_cross")


        # posterior mean of gradient
        # --------------------------
        L = linalg.cho_factor(covTrain)
        alpha = linalg.cho_solve(L, y)  #  K(X,X)^-1 y
        self.grad_post_mean = gradKern.dot(alpha).reshape(-1,3)

        # posterior variance of gradient
        # ------------------------------
        grad2Kern = self.makeCovar("grad_grad_prediction") # N x (3 x 3)

        # zeta = gradKern.dot(linalg.cho_solve(L, gradKern.T)) # probably correct, but causes memory error because large arrays

        invK = linalg.cho_solve(L, np.identity(covTrain.shape[0]))
        #zeta = np.einsum('km , pm -> kp', gradKern, invK.T.copy()) 
        zeta = np.inner(gradKern, invK.T) # much faster than doing einsum above
        zeta = np.einsum('klp , kjp -> klj', zeta.reshape(-1, 3, y.shape[0]), gradKern.reshape(-1, 3, y.shape[0])) # reshape for cross terms on different directions

        self.grad_post_var = grad2Kern - zeta


        #print("\n[WARNING] the variance of posterior gradient is a 3x3 matrix for each vertex.\n")
        return self.grad_meanFunction + self.unscale(self.grad_post_mean, std = True) # ,  self.unscale(self.unscale(vVar, std = True)) # FIXME: it is not clear how to return UQ on CV at the moment

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
    def gradientStatistics(self, numSamples = 2000, centIdx = None, s2pred = None):
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

    #{{{screePlot
    def screePlot(self):
    
        if self.HP is None:
            print("\n[ERROR]: must call optimize before <screePlot> is called\n")
            return

        val, _ = self.spectralDensity(np.sqrt(self.Q), self.HP[0])
        cumsum = np.cumsum(val)/ np.sum(val)

        for val in [0.95, 0.997]:

            #plt.axhline(val)

            try:
                cumsum_where = np.argwhere(cumsum > val).min()
                #plt.axvline(cumsum_where)
                print("{:1.3f} variance explained with {:d} eigenfunctions".format(val, cumsum_where + 1))
            except:
                pass

        sum_at_256 = cumsum[255]
        plt.axvline(255)
        plt.axhline(100*sum_at_256) 
        print("{:1.3f} variance explained with {:d} eigenfunctions".format(sum_at_256*100, 256))

        plt.plot(np.arange(cumsum.shape[0]), 100*cumsum, color = "red")
        plt.xlabel("Eigenfunctions included")
        plt.ylabel("Percentage of variance explained")
        plt.title("{:1.1f}% of variance captured with {:d} eigenfunctions".format(sum_at_256*100, 256))
        plt.ylim(0,100)
        plt.xlim(0,self.Q.shape[0])
        plt.show()
    #}}}

#{{{ subclasses for different models

#{{{ Matern32 for GP, no mean function
class Matern32(AbstractModel):
    """Model y = GP(x) + e where GP(x) is a manifold GP with Matern32 kernel."""       

    def spectralDensity(self, rho, w, grad = False):
        """Spectral Density for Matern kernel.
        
           w: sqrt(eigenvalues)
           rho: lengthscale parameter

        """
        
        D = 2  # dimension
        v = 3.0/2.0  # smoothness

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

        # hard wired simpler form for v = 5.2 and D = 2; does not seem to give any speed ups at all
        #a = np.sqrt(5.)/rho
        #result_2 = 10.*np.pi*a**5 * ( a**2 + 4*np.pi**2*w**2 )**(-7/2)


    def setMeanFunction(self, smooth = 10.0):
        """Set mean function."""
        self.meanFunction = np.array([0])
        self.grad_meanFunction = np.array([[0,0,0]])
#}}}

#{{{ manifold splines for mean, Matern32 GP for residuals
class splinesAndMatern32(Matern32): # NOTE: inherets Matern32 class but reimplements setMeanFunction
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

        #gradG = np.einsum( "ijk, lk -> ijl" , self.gradV[:, :, 1:], ((1.0/self.Q[1:])*self.V[self.vertex,1:]) ) 
        gradG = np.inner( self.gradV[:, :, 1:], ((1.0/self.Q[1:])*self.V[self.vertex,1:]) )  # faster than above
        self.grad_meanFunction = gradG.dot(beta)

        # rescale y data
        y = self.y - self.meanFunction[self.vertex]
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y, self.yerr = self.scale(y), self.scale(self.yerr, std = True)
#}}}

#{{{ linear model of laplacian eigenfunction, Matern32 GP for residuals
class basisAndMatern32(Matern32): # NOTE: inherets Matern32 class but reimplements setMeanFunction
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
        self.grad_meanFunction = np.inner(beta, self.gradV[:,:,1:1+num])

        # rescale y data
        y = self.y - self.meanFunction[self.vertex]
        self.y_mean, self.y_std = np.mean(y), np.std(y)
        self.y, self.yerr = self.scale(y), self.scale(self.yerr, std = True)
#}}}

#}}}

