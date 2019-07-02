from integral_output_observed import Integral_Output_Observed as Integral
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math
from scipy.misc import factorial
import numpy as np
import math
import random
from hashlib import sha1
from numpy import uint8
import rectangles


#from http://machineawakening.blogspot.com/2011/03/making-numpy-ndarrays-hashable.html
class hashable(object):
    r'''Hashable wrapper for ndarray objects.

        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.

        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.

            wrapped
                The wrapped ndarray.

            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = np.array(wrapped) if tight else wrapped
        self.__hash = int(sha1(np.array(wrapped).view(uint8).copy(order='C')).hexdigest(), 16)

    def __eq__(self, other):
        return np.all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.

            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped

class ShapeIntegralHC(Kern):
    def __init__(self, input_dim, input_space_dim=None, active_dims=None, name='shapeintegralhc',lengthscale=None, variances=None,Nrecs=10,step=0.025,Ntrials=10,dims=2):
        super(ShapeIntegralHC, self).__init__(input_dim, active_dims, name)
        assert ((input_space_dim is not None)), "Need the input space dimensionality defining"
        kernel = Integral(input_dim=input_space_dim*2,lengthscale=lengthscale,variances=variances)
        self.lengthscale = Param('lengthscale', kernel.lengthscale, Logexp())
        self.variances = Param('variances', kernel.variances, Logexp()) 
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.
        
        
        self.kernel = kernel
        self.input_space_dim = input_space_dim
        self.rectangle_cache = {} #this is important, not only is it a speed up - we also get the same points for each shape, which makes our covariances more stable        
        
        self.Nrecs=Nrecs
        self.step=step
        self.Ntrials=Ntrials
        
        
    def add_to_cache(self,X,newX):
        self.rectangle_cache[hashable(X).__hash__()] = newX
        
    def delete_cache(self):
        self.rectangle_cache = {}
        
    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        
        #TODO
        if X2 is None:
            X2 = X

    #from functools import lru_cache
    #@lru_cache(maxsize=32)
    def cached_compute_newX(self,X):
        #return rectangles.compute_newX(X)
        if hashable(X).__hash__() in self.rectangle_cache:
            return self.rectangle_cache[hashable(X).__hash__()]
        else:
            newX = rectangles.compute_newX(X,Nrecs=self.Nrecs,step=self.step,Ntrials=self.Ntrials)
            self.add_to_cache(X,newX)
            return newX
    
    def K(self, X1, X2=None):
        symmetric = False
        if X2 is None:
            symmetric = True
            X2 = X1
        newX1, startindices1,allvolscales1,allvolcorrections1,_,_ = self.cached_compute_newX(X1)
        newX2, startindices2,allvolscales2,allvolcorrections2,_,_ = self.cached_compute_newX(X2)
        #fullK = self.kernel.K(np.r_[np.r_[tuple(newX1[:])],np.r_[tuple(newX2[:])]])
        fullK = self.kernel.K(np.r_[tuple(newX1[:])],np.r_[tuple(newX2[:])])
        #print(np.r_[tuple(allvolcorrections1)][:,None]@np.r_[tuple(allvolcorrections2)][None,:])
        fullK*=np.r_[tuple(allvolcorrections1)][:,None]@np.r_[tuple(allvolcorrections2)][None,:]
        
        K = np.zeros([len(newX1),len(newX2)])
        for i1,(X1,vscale1,s1,e1) in enumerate(zip(newX1,allvolscales1,startindices1[0:-1],startindices1[1:])):
            for i2,(X2,vscale2,s2,e2) in enumerate(zip(newX2,allvolscales2,startindices2[0:-1],startindices2[1:])):
                K[i1,i2] = np.sum(fullK[s1:e1,s2:e2])#*vscale1*vscale2
        if symmetric: #because the approximations are different for X1 and X2, even if it's supposed to be symmetric it won't be (quite)
            #we average the reflections to make it symmetric!
            K = (K + K.T)/2
        return K

    def Kdiag(self, X):
        return np.diag(self.K(X))
