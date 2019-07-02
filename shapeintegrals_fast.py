# Written by Mike Smith michaeltsmith.org.uk

from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math
from scipy.misc import factorial
import numpy as np
import math
import random

#TODO: Is it ok for us to just fill the rest of X in with zeros?
# these won't have any points chosen in those 0-volume areas... but ideally we should do something else? Put NaNs in????

#round (by probability)
#https://stackoverflow.com/questions/19045971/random-rounding-to-integer-in-python
def prob_round(x):
    sign = np.sign(x)
    x = abs(x)
    is_up = random.random() < x-int(x)
    round_func = math.ceil if is_up else math.floor
    return int(sign * round_func(x))


class ShapeIntegral(Kern):
    """
    
    """

    def __init__(self, input_dim, input_space_dim=None, active_dims=None, kernel=None, name='shapeintegral',Nperunit=100):#, lengthscale=None, variance=None):
        """
        NOTE: Added input_space_dim as the number of columns in X isn't the dimensionality of the space. I.e. for pentagons there
        will be 10 columns in X, while only 2 dimensions of input space.
        
        The lengthscale, variance, etc are set by specifying the kernel we'll use
        """
        super(ShapeIntegral, self).__init__(input_dim, active_dims, name)
        
        assert ((kernel is not None) or (input_space_dim is not None)), "Need either the input space dimensionality defining or the latent kernel defining (to infer input space)"
        if kernel is None:
            kernel = RBF(input_space_dim, lengthscale=lengthscale)
        else:
            input_space_dim = kernel.input_dim
        assert kernel.input_dim == input_space_dim, "Latent kernel (dim=%d) should have same input dimensionality as specified in input_space_dim (dim=%d)" % (kernel.input_dim,input_space_dim)
        
        
        #assert len(kern.lengthscale)==input_space_dim, "Lengthscale of length %d, but input space has %d dimensions" % (len(lengthscale),input_space_dim)

        #self.lengthscale = Param('lengthscale', kernel.lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        #self.variance = Param('variance', kernel.variance, Logexp()) #and here.
        #self.link_parameters(self.variance, self.lengthscale) #this just takes a list of parameters we need to optimise.
        
        self.kernel = kernel
        self.Nperunit = Nperunit
        self.input_space_dim = input_space_dim
        self.cached_points = {} #this is important, not only is it a speed up - we also get the same points for each shape, which makes our covariances more stable
        

    def simplexRandom(self,vectors):
        #vectors = np.array([[0,0],[0,2],[1,1]])
        """
        Compute random point in arbitrary simplex

        from Grimme, Christian. Picking a uniformly
        random point from an arbitrary simplex.
        Technical Report. University of M\:{u}nster, 2015.

        vectors are row-vectors describing the
        vertices of the simplex, e.g.
        [[0,0],[0,2],[1,1]] is a triangle
        """
        d = vectors.shape[1]
        n = vectors.shape[0]
        assert n == d+1, "Need exactly d+1 vertices to define a simplex (e.g. a 2d triangle needs 3 points, a 3d tetrahedron 4 points, etc). Currently have %d points and %d dimensions" % (n,d)

        zs = np.r_[1,np.random.rand(d),0]
        ls = zs**(1.0/np.arange(len(zs)-1,-1,-1))
        vs = np.cumprod(ls) #could skip last element for speed
        res = vectors.copy()
        res = np.zeros(d)
        for vect,l,v in zip(vectors.copy(),ls[1:],vs):
            res+=(1-l)*v*vect
        return res

    def simplexVolume(self, vectors):
        """Returns the volume of the simplex defined by the
        row vectors in vectors, e.g. passing [[0,0],[0,2],[2,0]]
        will return 2 (as this triangle has area of 2)"""
        assert vectors.shape[0]==self.input_space_dim+1, "For a %d dimensional space there should be %d+1 vectors describing the simplex" % (self.input_space_dim, self.input_space_dim)
        return np.abs(np.linalg.det(vectors[1:,:]-vectors[0,:]))/factorial(self.input_space_dim)
        
    def delete_cache(self):
        self.cached_points = {}
      
    def polytopeVolume(self, shape):
        Ncoords = self.input_space_dim*(self.input_space_dim+1)
        vol = 0
        for i in range(0,len(shape),Ncoords):
            vectors = shape[i:(i+Ncoords)].reshape(self.input_space_dim+1,self.input_space_dim)
            if np.isnan(vectors[0,0]):
                break
            vol += self.simplexVolume(vectors)
        return vol
            
            
            
    def placepoints(self,shape,Nperunit=100):
        """Places uniformly random points in shape, where shape
        is defined by an array of concatenated simplexes
        e.g. a 2x2 square (from [0,0] to [2,2]) could be built
        of two triangles:
        [0,0,0,2,2,0 ,2,2,0,2,2,0]"""
        
        #caching is vital to the algorithm's computed covariance matrix
        #we don't want to cause non-positive-definite matrices
        tuplehash = tuple(shape.tolist()) #TODO Add Nperunit to hash
        if tuplehash in self.cached_points:
            return self.cached_points[tuplehash]
            
            
        allps = []
        #each simplex in shape must have D*(D+1) coordinates, e.g. a triangle has 2*(2+1) = 6 coords (2 for each vertex)
        #e.g. a tetrahedron has 4 points, each with 3 coords = 12: 3*(3+1) = 12.
        
        Ncoords = self.input_space_dim*(self.input_space_dim+1)
        assert len(shape)%Ncoords == 0, "The number of coordinates (%d) describing the simplexes that build the shape must factorise into the number of coordinates in a single simplex in %d dimensional space (=%d)" % (len(shape), self.input_space_dim, Ncoords)
        
        totalvol = 0
        vols = []
        for i in range(0,len(shape),Ncoords):
            vectors = shape[i:(i+Ncoords)].reshape(self.input_space_dim+1,self.input_space_dim)
            if np.isnan(vectors[0,0]):
                break
            vol = self.simplexVolume(vectors)
            totalvol += vol
            vols.append(vol)

        totalexpectednpoints = 0
        totalnpoints = 0
        for i,vol in zip(range(0,len(shape),Ncoords),vols):
            vectors = shape[i:(i+Ncoords)].reshape(self.input_space_dim+1,self.input_space_dim)
            if np.isnan(vectors[0,0]):
                break
            npoints = prob_round(Nperunit*vol/totalvol)
            totalexpectednpoints += Nperunit*vol/totalvol
            totalnpoints += npoints
      #      if (totalnpoints<totalexpectednpoints-1) and (vol>0):
      #          npoints+=1
      #      if (totalnpoints>totalexpectednpoints+1) and (npoints>=1):
      #          npoints-=1
                
            points = np.array([self.simplexRandom(vectors) for i in range(npoints)])
            allps.extend(points)
            
        self.cached_points[tuplehash] = (np.array(allps),totalvol)
        return np.array(allps),totalvol
        
    def calc_K_xx_wo_variance(self,X,X2=None):
        """Calculates K_xx without the variance term
        
        X is in the form of an array, each row for one shape. each
        is defined by an array of concatenated simplexes
        e.g. a 2x2 square (from [0,0] to [2,2]) could be built
        of two triangles:
        [0,0,0,2,2,0 ,2,2,0,2,2,0]
        """
        
        if X2 is None:
            X2 = X

        ps = []
       

        pvols = []
        for s in X:
            s = s[~np.isnan(s)]
            pspoints, pvol = self.placepoints(s,self.Nperunit)
            ps.append(pspoints)
            pvols.append(pvol)
            
        qs = []     
        qvols = []            
        for s in X2:
            s = s[~np.isnan(s)]
            qspoints, qvol = self.placepoints(s,self.Nperunit)            
            qs.append(qspoints)
            qvols.append(qvol)
            
        K_xx = np.ones([len(ps),len(qs)])
        
        for i,(p,vp) in enumerate(zip(ps,pvols)):
            for j,(q,vq) in enumerate(zip(qs,qvols)): 
                if (len(p)==0) or (len(q)==0):
                    #print("Warning: no points in simplex. Assuming no covariance!")
                    v = 0 #what else can we do?
                else:
                    cov = self.kernel.K(p,q)
                    #print(np.min(p,0),np.max(p,0),np.min(q,0),np.max(q,0))
                    #print(vp,vq,np.sum(cov)/np.prod(cov.shape))
                    v = vp*vq*np.sum(cov)/np.prod(cov.shape) #(self.Nperunit*self.Nperunit)
                K_xx[i,j] = v #TODO Compute half and mirror
        return K_xx

    def update_gradients_full(self, dL_dK, X, X2=None):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        #self.variance.gradient = np.sum(self.K(X, X2)* dL_dK)/self.variance

        #now the lengthscale gradient(s)
        #print dL_dK

        if X2 is None:
            X2 = X
        ls_grads = np.zeros([len(X), len(X2), len(self.kernel.lengthscale.gradient)])
        var_grads = np.zeros([len(X), len(X2)])
       
        ps = []
        qs = []

        vols = []
        for s in X:
            s = s[~np.isnan(s)]
            pspoints, vol = self.placepoints(s,self.Nperunit)
            ps.append(pspoints)
            qspoints, _ = self.placepoints(s,self.Nperunit)            
            qs.append(qspoints)
            vols.append(vol)
        for i,(p,vp) in enumerate(zip(ps,vols)):
            for j,(q,vq) in enumerate(zip(qs,vols)):
                if (len(p)==0) or (len(q)==0):
                    pass
                    #print("Warning: no points in simplex. Assuming no covariance!")
                else:
                    self.kernel.update_gradients_full(np.ones([len(p),len(q)]), p, q)
                    ls_grads[i,j,:] = (vp*vq)*self.kernel.lengthscale.gradient/(len(p)*len(q))
                    var_grads[i,j] = (vp*vq)*self.kernel.variance.gradient/(len(p)*len(q))
        #print dL_dK.shape
        #print grads[:,:,0] * dL_dK
        lg = np.zeros_like(self.kernel.lengthscale.gradient)
        #find (1/N^2) * sum( gradient )
        for i in range(ls_grads.shape[2]):
            #print(np.prod(ls_grads[:,:,i].shape))
            lg[i] = np.sum(ls_grads[:,:,i] * dL_dK)#/np.prod(ls_grads[:,:,i].shape)
            
        vg = np.sum(var_grads[:,:] * dL_dK)#/np.prod(var_grads[:,:].shape)
            
        self.kernel.lengthscale.gradient = lg
        self.kernel.variance.gradient = vg


    def K(self, X, X2=None):
        K_xx = self.calc_K_xx_wo_variance(X,X2)            
        return K_xx #* self.kernel.variance[0]
            
        #if X2 is None: #X vs X
        #    K_xx = self.calc_K_xx_wo_variance(X)
        #    return K_xx * self.kernel.variance[0]
        #else: #X vs X2
        #    
        #    #pass #TODO
            

    def Kdiag(self, X):
        return np.diag(self.K(X))
