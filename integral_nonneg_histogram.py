
from dp4gp import dp4gp
import GPy
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import scipy
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dp4gp.utils import bin_data
from mix_integral import Mix_Integral
from mixed import Mixed

class DPGP_nonneg_integral_histogram(dp4gp.DPGP):
    """Using the histogram method"""
    
    def __init__(self,sens,epsilon,delta):   
        """
        DPGP_integral_histogram(sensitivity=1.0, epsilon=1.0, delta=0.01)
        
        sensitivity=1.0 - the amount one output can change
        epsilon=1.0, delta=0.01 - DP parameters
        """
        super(DPGP_nonneg_integral_histogram, self).__init__(None,sens,epsilon,delta)

    def prepare_model(self,Xtest,X,step,ys,scaling,variances=1.0,lengthscale=1,aggregation='mean',mechanism='laplace'):
        """
        Prepare the model, ready for making predictions
        """
        bincounts, bintotals, binaverages = bin_data(Xtest,X,step,ys,aggregation=aggregation)
        if aggregation=='median':
            raise NotImplementedError
        if aggregation=='mean':            
            sens_per_bin = self.sens/bincounts
        if aggregation=='sum':
            sens_per_bin = self.sens*np.ones_like(bincounts)
        if aggregation=='density':
            sens_per_bin = (self.sens/np.prod(step))*np.ones_like(bincounts)
            
        if mechanism=='gaussian':
            c = np.sqrt(2*np.log(1.25/self.delta)) #1.25 or 2 over delta?
            bin_sigma = c*sens_per_bin/self.epsilon #noise standard deviation to add to each bin       
            ##add DP noise to the binaverages
            dp_binaverages=binaverages+np.random.randn(binaverages.shape[0])*bin_sigma
        if mechanism=='laplace':
            #note the standard deviation is np.sqrt(2)*the scale parameter
            bin_sigma = np.array(sens_per_bin / self.epsilon) * np.sqrt(2)
            dp_binaverages=binaverages+np.random.laplace(scale=bin_sigma/np.sqrt(2),size=binaverages.shape)
        

        #we need to build the input for the integral kernel
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest+step
        newXtest[:,1::2] = Xtest

        #we don't want outputs that have no training data in.
        keep = ~np.isnan(dp_binaverages)
        finalXtest = newXtest[keep,:]
        final_dp_binaverages = dp_binaverages[keep]
        bin_sigma = bin_sigma[keep]

        
        #the integral kernel takes as y the integral... 
        #eg. if there's one dimension we're integrating over, km
        #then we need to give y in pound.km
        self.meanoffset = 0.0 ####np.mean(final_dp_binaverages)##########<<<<<<<<<<<<<<
        actual_dp_binaverages = final_dp_binaverages * np.prod(step)
        final_dp_binaverages[final_dp_binaverages<0]=0 #force non-negative!
        final_dp_binaverages-= self.meanoffset
        finalintegralbinaverages = final_dp_binaverages * np.prod(step) 

        #final_sigma = 2.0*(bin_sigma**2) #I've no idea but optimizer will find this later... #bin_sigma[keep]
        finalintegralsigma = bin_sigma * np.prod(step)  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        
        #generate the integral model
        self.scaling = scaling
        kernel = Mix_Integral(3,variances=variances*scaling**2,lengthscale=[lengthscale])
        #separate likelihood for each gaussian observation (To allow varying amounts of noise)
        #likelihood_fns = [GPy.likelihoods.Bernoulli()]
        #for varian in (finalintegralsigma**2+actual_dp_binaverages):
        #    likelihood_fns.append(GPy.likelihoods.Gaussian(variance=varian))
        likelihood_fns = [GPy.likelihoods.Bernoulli(),GPy.likelihoods.Gaussian(variance=1*scaling**2)]
        
        
        n_non_negs = 50
        
        #we add a kernel to describe the DP noise added
        actual_dp_binaverages[actual_dp_binaverages<0] = 0
        #print(finalintegralsigma.shape)
        varian = (finalintegralsigma**2)*scaling**2 #+actual_dp_binaverages
        #print(varian.shape)        
        varian = np.r_[varian,0.0001*np.ones(n_non_negs)]
        #print(varian.shape)
        kernel = kernel + GPy.kern.WhiteHeteroscedastic(input_dim=newXtest.shape[1], num_data=len(varian), variance=varian)
        ##print("INTEGRAL")
        
        ##print(finalintegralsigma**2,actual_dp_binaverages)
        
        
        X = finalXtest
        
        Y = scaling*finalintegralbinaverages[:,None]
        X = np.c_[X,np.zeros([len(X),1])]
        fn_idx = np.ones([len(X),1])
        #fn_idx = np.arange(len(X))[:,None]+1
        non_negs_X = np.linspace(np.min(X),np.max(X),n_non_negs)[:,None]
        non_negs_X = np.c_[non_negs_X,np.zeros([len(non_negs_X),1]),np.ones([len(non_negs_X),1])]
        non_negs_Y = np.ones([len(non_negs_X),1])
        non_negs_fn_idx = np.zeros_like(non_negs_Y)
        X = np.r_[X,non_negs_X]
        Y = np.r_[Y,non_negs_Y]
        fn_idx = (np.r_[fn_idx,non_negs_fn_idx]).astype(int)
        Y_metadata = {'likelihood_fn_index':fn_idx} 

        #print(X)
        #print(Y)
        #print(Y_metadata['likelihood_fn_index'])
        self.model = GPy.core.GP( 
            X, Y,        
            kernel = kernel, 
            inference_method = GPy.inference.latent_function_inference.EP(),
            likelihood = Mixed(likelihood_fns=likelihood_fns),
            Y_metadata = Y_metadata,normalizer=False,
        )
    
        self.model.sum.white_hetero.variance.fix() #fix the DP noise
        self.model.sum.mix_integral.lengthscale.constrain_bounded(1,15)
        ###################self.model.Gaussian_noise = 0.1 # seems to need starting at a low value!        
        return bincounts, bintotals, binaverages, sens_per_bin, bin_sigma, dp_binaverages
    
    def optimize(self,messages=True):
        #self.model.optimize_restarts(num_restarts=1,messages=messages)
        self.model.optimize()
     
    def draw_prediction_samples(self,Xtest,N=1):
        assert N==1, "DPGP_histogram only returns one DP prediction sample (you will need to rerun prepare_model to get an additional sample)"
        newXtest = np.zeros([Xtest.shape[0],2*Xtest.shape[1]])
        newXtest[:,0::2] = Xtest
        newXtest[:,1::2] = 0
        mean, cov = self.model.predict(newXtest)
        #print(self.scaling)     
        return mean/self.scaling+self.meanoffset, cov/(self.scaling**2)
