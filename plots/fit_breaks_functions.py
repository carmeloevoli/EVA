import numpy as np
from iminuit import Minuit
from jacobi import propagate

from utils import load_data

class SBrokenPowerLaw:
    E0 = 1e3  # Constant for scaling

    def __init__(self, K, alpha, lgEb, dalpha, w):
        # Store the parameters in a dictionary for easier access
        self.K = K
        self.alpha = alpha 
        self.Eb = np.power(10., lgEb)
        self.dalpha = dalpha
        self.w = w

    def compute(self, E):
        y = self.K * (E / self.E0) ** -self.alpha
        y *= (1. + (E / self.Eb) ** self.w) ** (self.dalpha / self.w)
        return y
    

def fit_sbpl(initial_params, filename, minEnergy, maxEnergy, slope, doFixSmoothness = True):
    def chi2_function(K, alpha, lgEb, dalpha, w):
        params = [K, alpha, lgEb, dalpha, w]
        E, y_data, err_lo, err_up = load_data(filename, slope=slope, norm=1.0, min_energy=minEnergy, max_energy=maxEnergy)
        model = SBrokenPowerLaw(*params)
        
        y_model = np.array([model.compute(E_i) for E_i in E])
        chi2 = np.sum(np.where(y_model > y_data,
                            ((y_model - y_data) / err_up) ** 2,
                            ((y_model - y_data) / err_lo) ** 2))
        return chi2
        
    K, alpha, lgEb, dalpha, w = initial_params
    m = Minuit(chi2_function, K=K, alpha=alpha, lgEb=lgEb, dalpha=dalpha, w=w)

    m.limits['alpha'] = (-0.5, 0.5)
    m.limits['w'] = (1., 10.)
    m.limits['dalpha'] = (-0.3, 0.3)
    
    m.fixed['w'] = doFixSmoothness

    m.errordef = Minuit.LEAST_SQUARES
    m.simplex()
    m.migrad()
    m.hesse()

    dof = 0 # len(data_x) - m.nfit - 1

    # print(m) 

    return m.values, m.errors, m.covariance, m.fval, dof
 

def dump_sbpl(values, covariance, filename):    
    def get(E):
        def compute_model(E, params):
            model = SBrokenPowerLaw(*params)
            return model.compute(E) 

        y, ycov = propagate(lambda p: compute_model(E, p), values, covariance)
        y_errors = np.sqrt(np.diag(ycov))
        return y, y_errors

    E = np.logspace(2, 10, 4000)
    y, y_err = get(E)

    with open(f'output/{filename}', "w") as f:
        for i in range(len(E)):
            f.write(f"{E[i]:.2e} {y[i]:.3e} {y_err[i]:.3e}\n")


class DblSBrokenPowerLaw:
    E0 = 1e3  # Constant for scaling

    def __init__(self, K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2):
        # Store the parameters in a dictionary for easier access
        self.K = K
        self.alpha = alpha 
        self.Eb_1 = np.power(10., lgEb_1)
        self.dalpha_1 = dalpha_1
        self.w_1 = w_1
        self.Eb_2 = np.power(10., lgEb_2)
        self.dalpha_2 = dalpha_2
        self.w_2 = w_2

    def compute(self, E):
        y = self.K * (E / self.E0) ** -self.alpha
        y *= (1. + (E / self.Eb_1) ** self.w_1) ** (self.dalpha_1 / self.w_1)
        y *= (1. + (E / self.Eb_2) ** self.w_2) ** (self.dalpha_2 / self.w_2)
        return y


def fit_dblsbpl(initial_params, filename, minEnergy, maxEnergy, slope, doFixSmoothness = True):
    def chi2_function(K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2):
        params = [K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2]
        E, y_data, err_lo, err_up = load_data(filename, slope=slope, norm=1.0, min_energy=minEnergy, max_energy=maxEnergy)
        model = DblSBrokenPowerLaw(*params)
        
        y_model = np.array([model.compute(E_i) for E_i in E])
        chi2 = np.sum(np.where(y_model > y_data,
                            ((y_model - y_data) / err_up) ** 2,
                            ((y_model - y_data) / err_lo) ** 2))
        return chi2
        
    K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2 = initial_params
    m = Minuit(chi2_function, K=K, alpha=alpha, 
               lgEb_1=lgEb_1, dalpha_1=dalpha_1, w_1=w_1,
               lgEb_2=lgEb_2, dalpha_2=dalpha_2, w_2=w_2,
               )

    m.limits['alpha'] = (-0.6, 0.6)
    m.limits['w_1'] = (1., 10.)
    m.limits['w_2'] = (1., 10.)
    m.limits['dalpha_1'] = (-0.3, 0.3)
    m.limits['dalpha_2'] = (-0.3, 0.3)
    
    m.fixed['w_1'] = doFixSmoothness
    m.fixed['w_2'] = doFixSmoothness

    m.errordef = Minuit.LEAST_SQUARES
    m.simplex()
    m.migrad()
    m.hesse()

    dof = 0 # len(data_x) - m.nfit - 1

    #print(m) 

    return m.values, m.errors, m.covariance, m.fval, dof


def dump_dblsbpl(values, covariance, filename):    
    def get(E):
        def compute_model(E, params):
            model = DblSBrokenPowerLaw(*params)
            return model.compute(E) 

        y, ycov = propagate(lambda p: compute_model(E, p), values, covariance)
        y_errors = np.sqrt(np.diag(ycov))
        return y, y_errors

    E = np.logspace(2, 10, 4000)
    y, y_err = get(E)

    with open(f'output/{filename}', "w") as f:
        for i in range(len(E)):
            f.write(f"{E[i]:.2e} {y[i]:.3e} {y_err[i]:.3e}\n")


class TrpSBrokenPowerLaw:
    E0 = 1e3  # Constant for scaling

    def __init__(self, K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2, lgEb_3, dalpha_3, w_3):
        # Store the parameters in a dictionary for easier access
        self.K = K
        self.alpha = alpha 
        self.Eb_1 = np.power(10., lgEb_1)
        self.dalpha_1 = dalpha_1
        self.w_1 = w_1
        self.Eb_2 = np.power(10., lgEb_2)
        self.dalpha_2 = dalpha_2
        self.w_2 = w_2
        self.Eb_3 = np.power(10., lgEb_3)
        self.dalpha_3 = dalpha_3
        self.w_3 = w_3

    def compute(self, E):
        y = self.K * (E / self.E0) ** -self.alpha
        y *= (1. + (E / self.Eb_1) ** self.w_1) ** (self.dalpha_1 / self.w_1)
        y *= (1. + (E / self.Eb_2) ** self.w_2) ** (self.dalpha_2 / self.w_2)
        y *= (1. + (E / self.Eb_3) ** self.w_3) ** (self.dalpha_3 / self.w_3)
        return y
    

def fit_trpsbpl(initial_params, filename, minEnergy, maxEnergy, slope, doFixSmoothness = True):
    def chi2_function(K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2, lgEb_3, dalpha_3, w_3):
        params = [K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2, lgEb_3, dalpha_3, w_3]
        E, y_data, err_lo, err_up = load_data(filename, slope=slope, norm=1.0, min_energy=minEnergy, max_energy=maxEnergy)
        model = TrpSBrokenPowerLaw(*params)
        
        y_model = np.array([model.compute(E_i) for E_i in E])
        chi2 = np.sum(np.where(y_model > y_data,
                            ((y_model - y_data) / err_up) ** 2,
                            ((y_model - y_data) / err_lo) ** 2))
        return chi2
        
    K, alpha, lgEb_1, dalpha_1, w_1, lgEb_2, dalpha_2, w_2, lgEb_3, dalpha_3, w_3 = initial_params
    m = Minuit(chi2_function, K=K, alpha=alpha, 
               lgEb_1=lgEb_1, dalpha_1=dalpha_1, w_1=w_1,
               lgEb_2=lgEb_2, dalpha_2=dalpha_2, w_2=w_2,
               lgEb_3=lgEb_3, dalpha_3=dalpha_3, w_3=w_3,
               )

    m.limits['alpha'] = (-0.6, 0.6)
    m.limits['w_1'] = (1., 10.)
    m.limits['w_2'] = (1., 10.)
    m.limits['w_3'] = (1., 10.)
    m.limits['dalpha_1'] = (-0.5, 0.5)
    m.limits['dalpha_2'] = (-0.5, 0.5)
    m.limits['dalpha_3'] = (-0.5, 0.5)
    
    m.fixed['w_1'] = doFixSmoothness
    m.fixed['w_2'] = doFixSmoothness
    m.fixed['w_3'] = doFixSmoothness

    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.minos()

    dof = 0 # len(data_x) - m.nfit - 1

    print(m.errors)
    
    print(m) 

    return m.values, m.errors, m.covariance, m.fval, dof


def dump_trpsbpl(values, covariance, filename):    
    def get(E):
        def compute_model(E, params):
            model = TrpSBrokenPowerLaw(*params)
            return model.compute(E) 

        y, ycov = propagate(lambda p: compute_model(E, p), values, covariance)
        y_errors = np.sqrt(np.diag(ycov))
        return y, y_errors

    E = np.logspace(2, 10, 4000)
    y, y_err = get(E)

    with open(f'output/{filename}', "w") as f:
        for i in range(len(E)):
            f.write(f"{E[i]:.2e} {y[i]:.3e} {y_err[i]:.3e}\n")