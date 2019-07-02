# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy import stats, special
from GPy.likelihoods import link_functions
from GPy.likelihoods.likelihood import Likelihood
from GPy.likelihoods import Gaussian
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from scipy import stats

class Mixed(Likelihood):
    """
    Mixed likelihoods
    Pass a list of likelihoods in likelihoods_fns. The Y_metadata
    will specify which likelihood each observation is associated with
    
    :param variance: variance value of the Gaussian distribution
    :param N: Number of data points
    :type N: int
    """
    def __init__(self, likelihood_fns=[], name='Mixed_noise'):
        #TODO Why do we need to specify a link function?
        super(Mixed, self).__init__(name=name, gp_link=link_functions.Identity())
        self.likelihood_fns = likelihood_fns
        #for lf in self.likelihood_fns:
        #    lf = 
        
        #TODO Optimise hyperparameters features
        #self.variance = Param('variance', variance, Logexp())
        #self.link_parameter(self.variance)

    def exact_inference_gradients(self, dL_dKdiag,Y_metadata=None):
        #TODO 
        return np.zeros(self.size)

    def _preprocess_values(self, Y):
        """
        Check if the values of the observations correspond to the values
        assumed by the likelihood function.
        """
        #TODO
        return Y

    def moments_match_ep(self, Y_i, tau_i, v_i, Y_metadata_i=None):
        """
        Moments match of the marginal approximation in EP algorithm

        :param i: number of observation (int)
        :param tau_i: precision of the cavity distribution (float)
        :param v_i: mean/variance of the cavity distribution (float)
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].moments_match_ep(Y_i, tau_i, v_i, Y_metadata_i)

    def predictive_mean(self, mu, sigma, Y_metadata=None):
        return mu #unclear what we should do here. #raise NotImplementedError

    def predictive_variance(self, mu, sigma, predictive_mean=None,Y_metadata=None):
        for f in self.likelihood_fns:
            if type(f)==Gaussian:
                return f.variance + sigma**2
        return sigma**2 #???

    def predictive_quantiles(self, mu, var, quantiles, Y_metadata=None):
        return np.nan

    def pdf_link(self, link_f, y, Y_metadata=None):
        """
        Likelihood function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: likelihood evaluated for this point
        :rtype: float
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].pdf_link(link_f, y, Y_metadata)

    def logpdf_link(self, link_f, y, Y_metadata):
        """
        Log likelihood function given link(f)

        .. math::
            \\ln p(y_{i}|\\lambda(f_{i})) = -\\frac{N \\ln 2\\pi}{2} - \\frac{\\ln |K|}{2} - \\frac{(y_{i} - \\lambda(f_{i}))^{T}\\sigma^{-2}(y_{i} - \\lambda(f_{i}))}{2}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: log likelihood evaluated for this point
        :rtype: float
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].logpdf_link(link_f, y, Y_metadata)

    def dlogpdf_dlink(self, link_f, y, Y_metadata=None):
        """
        Gradient of the pdf at y, given link(f) w.r.t link(f)

        .. math::
            \\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)} = \\frac{1}{\\sigma^{2}}(y_{i} - \\lambda(f_{i}))

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: gradient of log likelihood evaluated at points link(f)
        :rtype: Nx1 array
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].dlogpdf_dlink(link_f, y, Y_metadata)

    def d2logpdf_dlink2(self, link_f, y, Y_metadata=None):
        """
        Hessian at y, given link_f, w.r.t link_f.
        i.e. second derivative logpdf at y given link(f_i) link(f_j)  w.r.t link(f_i) and link(f_j)

        The hessian will be 0 unless i == j

        .. math::
            \\frac{d^{2} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{2}f} = -\\frac{1}{\\sigma^{2}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: Diagonal of log hessian matrix (second derivative of log likelihood evaluated at points link(f))
        :rtype: Nx1 array

        .. Note::
            Will return diagonal of hessian, since every where else it is 0, as the likelihood factorizes over cases
            (the distribution for y_i depends only on link(f_i) not on link(f_(j!=i))
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].d2logpdf_dlink2(link_f, y, Y_metadata)

    def d3logpdf_dlink3(self, link_f, y, Y_metadata=None):
        """
        Third order derivative log-likelihood function at y given link(f) w.r.t link(f)

        .. math::
            \\frac{d^{3} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{3}\\lambda(f)} = 0

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: third derivative of log likelihood evaluated at points link(f)
        :rtype: Nx1 array
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].d3logpdf_dlink3(link_f, y, Y_metadata)

    def dlogpdf_link_dvar(self, link_f, y, Y_metadata=None):
        """
        Gradient of the log-likelihood function at y given link(f), w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\sigma^{2}} = -\\frac{N}{2\\sigma^{2}} + \\frac{(y_{i} - \\lambda(f_{i}))^{2}}{2\\sigma^{4}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: float
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].dlogpdf_link_dvar(link_f, y, Y_metadata)

    def dlogpdf_dlink_dvar(self, link_f, y, Y_metadata=None):
        """
        Derivative of the dlogpdf_dlink w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d \\ln p(y_{i}|\\lambda(f_{i}))}{d\\lambda(f)}) = \\frac{1}{\\sigma^{4}}(-y_{i} + \\lambda(f_{i}))

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log likelihood evaluated at points link(f) w.r.t variance parameter
        :rtype: Nx1 array
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].dlogpdf_dlink_dvar(link_f, y, Y_metadata)

    def d2logpdf_dlink2_dvar(self, link_f, y, Y_metadata=None):
        """
        Gradient of the hessian (d2logpdf_dlink2) w.r.t variance parameter (noise_variance)

        .. math::
            \\frac{d}{d\\sigma^{2}}(\\frac{d^{2} \\ln p(y_{i}|\\lambda(f_{i}))}{d^{2}\\lambda(f)}) = \\frac{1}{\\sigma^{4}}

        :param link_f: latent variables link(f)
        :type link_f: Nx1 array
        :param y: data
        :type y: Nx1 array
        :param Y_metadata: Y_metadata not used in gaussian
        :returns: derivative of log hessian evaluated at points link(f_i) and link(f_j) w.r.t variance parameter
        :rtype: Nx1 array
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].d2logpdf_dlink2_dvar(link_f, y, Y_metadata)

    def dlogpdf_link_dtheta(self, f, y, Y_metadata=None):
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].dlogpdf_link_dtheta(link_f, y, Y_metadata)

    def dlogpdf_dlink_dtheta(self, f, y, Y_metadata=None):
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].dlogpdf_dlink_dtheta(link_f, y, Y_metadata)

    def d2logpdf_dlink2_dtheta(self, f, y, Y_metadata=None):
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].d2logpdf_dlink2_dtheta(link_f, y, Y_metadata)

    def samples(self, gp, Y_metadata=None):
        """
        Returns a set of samples of observations based on a given value of the latent variable.

        :param gp: latent variable
        """
        idx = Y_metadata_i['likelihood_fn_index'][0]
        return self.likelihood_fns[idx].samples(link_f, y, Y_metadata)

