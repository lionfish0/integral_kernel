# Written by Mike Smith michaeltsmith.org.uk

from __future__ import division
import numpy as np
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math

class Mix_Integral_Exponential(Kern):
    """
      
    """

    def __init__(self, input_dim, variances=None, lengthscale=None, ARD=False, active_dims=None, name='mix_integral_exponential'):
        super(Mix_Integral_Exponential, self).__init__(input_dim, active_dims, name)

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

    def getint(self,s,t,sprime,tprime,l):
        """
        get the covariance if not overlapping
        """
        if sprime<s:
            sprime,s = s,sprime
            tprime,t = t,tprime
        assert sprime>=t, "This function only handles disjoint integrals"
        return -l**2*(np.exp((s-sprime)/l)-np.exp((s-tprime)/l)-np.exp((t-sprime)/l)+np.exp((t-tprime)/l))

        
        
    def getintoverlap(self,s,sprime,l):
        """
        for two identical regions get the covariance
        """
        return 2*l*(l*(np.exp((s-sprime)/l)-1)-s+sprime)



    def k_xx(self,t,tprime,s,sprime,lengthscale):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""
     #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
            #first sort so u starts later
        if s<sprime:
            s,sprime = sprime,s
            t,tprime = tprime,t
        if tprime<s: #no overlap
            return self.getint(s,t,sprime,tprime,l)
        
        total = 0
        total+=self.getint(sprime,s,s,t,l)
        
        if tprime<t:
            total+=self.getintoverlap(s,tprime,l)
            total+=self.getint(s,tprime,tprime,t,l)
        else:
            total+=self.getintoverlap(s,t,l)
        if tprime>t:
            total+=self.getint(t,tprime,s,t,l)
        return total

    def k_ff(self,t,tprime,lengthscale):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""        
     #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        return np.exp(-np.abs(t-tprime)/l)


    def getcross(self,s,t,sprime,l):
        """ #TODO MOVE THIS INTO THE k_xf FUNCTION
        get the cross covariance
        """
        if sprime<=s:
            return l*np.exp((sprime-s)/l) - l*np.exp((sprime-t)/l)
        if sprime>=t:
            return - l*np.exp((s-sprime)/l) + l*np.exp((t-sprime)/l)
        #overlap
        return self.getcross(s,sprime,sprime,l)+self.getcross(sprime,t,sprime,l)
        
    def k_xf(self,t,tprime,s,lengthscale):
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that tprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
     #######   l = lengthscale * np.sqrt(2)###TO REINSTATE
        l = lengthscale
        """
        get the cross covariance
        """
        #if tprime<=t:
        #    return - l*np.exp((tprime-t)/l) + l*np.exp((tprime-s)/l)
        #if tprime>=s:
        #    return l*np.exp((t-tprime)/l) - l*np.exp((s-tprime)/l)
        #overlap
        #return getcross(t,tprime,tprime,l)+getcross(tprime,s,tprime,l)
        return self.getcross(-t,-s,-tprime,l)

        
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
