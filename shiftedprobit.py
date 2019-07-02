import numpy as np
import scipy
from GPy.util.univariate_Gaussian import std_norm_cdf, std_norm_pdf
import scipy as sp
from GPy.util.misc import safe_exp, safe_square, safe_cube, safe_quad, safe_three_times
from GPy.likelihoods.link_functions import Probit

class ShiftedProbit(Probit):
    """
    .. math::

        g(f) = \\Phi^{-1} (mu)

    """
    def __init__(self,offset,scale):
        super(ShiftedProbit, self).__init__()
        self.offset=offset
        self.scale=scale
        
    def transf(self,f):
        shiftedf = (f-self.offset)*self.scale
        return std_norm_cdf(shiftedf)

    def dtransf_df(self,f):
        shiftedf = (f-self.offset)*self.scale
        return std_norm_pdf(shiftedf)

    def d2transf_df2(self,shiftedf):
        shiftedf = (f-self.offset)*self.scale
        return -f * std_norm_pdf(shiftedf)

    def d3transf_df3(self,f):
        shiftedf = (f-self.offset)*self.scale    
        return (safe_square(shiftedf)-1.)*std_norm_pdf(shiftedf)

    def to_dict(self):
        input_dict = super(ShiftedProbit, self)._to_dict()
        input_dict["class"] = "GPy.likelihoods.link_functions.ShiftedProbit"
        return input_dict
        
