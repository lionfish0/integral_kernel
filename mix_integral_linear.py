# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import numpy as np
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math

class Mix_Integral_Linear(Kern):
    """
      
    """

    def __init__(self, input_dim, variances=None, ARD=False, active_dims=None, name='mix_integral_linear'):
        super(Mix_Integral_Linear, self).__init__(input_dim, active_dims, name)
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances) #this just takes a list of parameters we need to optimise.

    def update_gradients_full(self, dL_dK, X, X2=None): 
        pass
        #raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")


    def k_xx(self,t,tprime,s,sprime):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)"""
        return ((tprime**2 - sprime**2)*(t**2 - s**2))/4

    def k_ff(self,x,xprime):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""
        return x*xprime

    def k_xf(self,t,sprime,s):
        """Covariance between the gradient (latent value) and the actual (observed) value."""
        return (sprime)*(t**2/2 - s**2/2)

        
    def k(self,x,x2,idx):
        """Helper function to compute covariance in one dimension (idx) between a pair of points.
        The last element in x and x2 specify if these are integrals (0) or latent values (1).
        """
        
        if (x[-1]==0) and (x2[-1]==0):
            return self.k_xx(x[idx],x2[idx],x[idx+1],x2[idx+1])
        if (x[-1] == 0) and (x2[-1] == 1):
            return self.k_xf(x[idx],x2[idx],x[idx+1])
        if (x[-1] == 1) and (x2[-1] == 0):
            return self.k_xf(x2[idx],x[idx],x2[idx+1])                        
        if (x[-1]==1) and (x2[-1]==1):
            return self.k_ff(x[idx],x2[idx])
        assert False, "Invalid choice of latent/integral parameter (set the last column of X to 0s and 1s to select this)"

    def calc_K_xx_wo_variance(self,X,X2):
        """Calculates K_xx without the variance term"""
        K_xx = np.ones([X.shape[0],X2.shape[0]]) #ones now as a product occurs over each dimension
        for i,x in enumerate(X):
            for j,x2 in enumerate(X2):
                #for il,l in enumerate(self.lengthscale):
                for idx in range(0,len(x)-1,2):
                #idx = il*2 #each pair of input dimensions describe the limits on one actual dimension in the data
                    K_xx[i,j] *= self.k(x,x2,idx)
        return K_xx

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        K_xx = self.calc_K_xx_wo_variance(X,X2)
        return K_xx * self.variances[0]

    def Kdiag(self, X):
        return np.diag(self.K(X))
