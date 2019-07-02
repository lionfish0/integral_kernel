# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import numpy as np
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math

class Mix_Integral_Matern32(Kern):
    """
      
    """

    def __init__(self, input_dim, variances=None, lengthscale=None, ARD=False, active_dims=None, name='mix_integral_matern32'):
        super(Mix_Integral_Matern32, self).__init__(input_dim, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)
            
        assert len(lengthscale)==(input_dim-1)/2

        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.


    def update_gradients_full(self, dL_dK, X, X2=None): 
        pass
        #raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")

    def indefK(self,x,xprime,l=2.0):
        return -(l**2 + l*(x-xprime)/np.sqrt(3))*np.exp(-np.sqrt(3)*(x-xprime)/l)

    def getint(self,s,t,sprime,tprime,lengthscale):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are."""
        l = lengthscale
        return self.indefK(t,tprime,l)-self.indefK(s,tprime,l)-self.indefK(t,sprime,l)+self.indefK(s,sprime,l)

        
    def getintoverlap(self,s,sprime,l):
        """
        for two identical regions get the covariance
        """
        return 2*l*(l*(np.exp((s-sprime)/l)-1)-s+sprime)


    def k_ff(self,x,xprime,lengthscale):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""
        l = lengthscale
        r = np.abs(x-xprime)
        return (1+np.sqrt(3)*r/l)*np.exp(-np.sqrt(3)*r/l)

    def indefKcross(self,x,xprime,l=2.0):
        return -(2*l/np.sqrt(3) + (x-xprime))*np.exp(-np.sqrt(3)*(x-xprime)/l)
    
    def k_xf(self,t,sprime,s,lengthscale):
        """Covariance between the gradient (latent value) and the actual (observed) value."""
        l = lengthscale
        return self.indefKcross(t,sprime,l)-self.indefKcross(s,sprime,l)

        
    def k(self,x,x2,idx,l):
        """Helper function to compute covariance in one dimension (idx) between a pair of points.
        The last element in x and x2 specify if these are integrals (0) or latent values (1).
        l = that dimension's lengthscale
        """
        
        if (x[-1]==0) and (x2[-1]==0):
            return self.k_xx(x[idx],x2[idx],x[idx+1],x2[idx+1],l)
        if (x[-1] == 0) and (x2[-1] == 1):
            return self.k_xf(x[idx],x2[idx],x[idx+1],l)
        if (x[-1] == 1) and (x2[-1] == 0):
            return self.k_xf(x2[idx],x[idx],x2[idx+1],l)                        
        if (x[-1]==1) and (x2[-1]==1):
            return self.k_ff(x[idx],x2[idx],l)
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"

    def calc_K_xx_wo_variance(self,X,X2):
        """Calculates K_xx without the variance term"""
        K_xx = np.ones([X.shape[0],X2.shape[0]]) #ones now as a product occurs over each dimension
        for i,x in enumerate(X):
            for j,x2 in enumerate(X2):
                for il,l in enumerate(self.lengthscale):
                    idx = il*2 #each pair of input dimensions describe the limits on one actual dimension in the data
                    K_xx[i,j] *= self.k(x,x2,idx,l)
        return K_xx

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        K_xx = self.calc_K_xx_wo_variance(X,X2)
        return K_xx * self.variances[0]

    def Kdiag(self, X):
        return np.diag(self.K(X))
