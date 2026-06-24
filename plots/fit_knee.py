import numpy as np
from iminuit import Minuit
from jacobi import propagate

from utils import load_data

class KneeModel:
    R0 = 1e3  # Constant for scaling
    x, I_H, I_H_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,1,2), unpack=True)
    x, I_He, I_He_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,3,4), unpack=True)
    x, I_C, I_C_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,5,6), unpack=True)
    x, I_O, I_O_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,7,8), unpack=True)
    x, I_Mg, I_Mg_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,9,10), unpack=True)
    x, I_Si, I_Si_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,11,12), unpack=True)
    x, I_Fe, I_Fe_err = np.loadtxt('output/galactic_fluxes_fit.txt', usecols=(0,13,14), unpack=True)

    def __init__(self, K, alpha, lgRb, dalpha, w):
        # Store the parameters in a dictionary for easier access
        self.K = K
        self.alpha = alpha
        self.Rb = np.power(10., lgRb)
        self.dalpha = dalpha
        self.w = w

    def grapes_break(self, E, Z):
        Rb, dslope = 3e5, 0.29
        R = E / float(Z)
        s = 10.0
        return (1. + (R / Rb) ** s) ** (dslope / s)

    def knee1st(self, E, Z):
        R = E / float(Z)
        s = 10.0
        return (1. + (R / self.Rb) ** self.w) ** (self.dalpha / self.w)

    def compute(self, E):
        y_H = self.I_H * self.K * self.grapes_break(self.x, 1.) * self.knee1st(self.x, 1.)
        y_He = self.I_He * self.K * self.grapes_break(self.x, 2.) * self.knee1st(self.x, 2.)
        y_C = self.I_C * self.K * self.grapes_break(self.x, 6.) * self.knee1st(self.x, 6.)
        y_O = self.I_O * self.K * self.grapes_break(self.x, 8.) * self.knee1st(self.x, 8.)
        y_Mg = self.I_Mg * self.K * self.grapes_break(self.x, 12.) * self.knee1st(self.x, 12.)
        y_Si = self.I_Si * self.K * self.grapes_break(self.x, 14.) * self.knee1st(self.x, 14.)
        y_Fe = self.I_Fe * self.K * self.grapes_break(self.x, 26.) * self.knee1st(self.x, 26.)

        y_all = y_H + y_He + y_C + y_O + y_Mg + y_Si + y_Fe

        return np.exp(np.interp(np.log(E), np.log(self.x), np.log(y_all)))

def experiment_chi2(filename, params, norm):
    min_energy, max_energy = 1e5, 2e7

    E, y_data, err_lo, err_up = load_data(filename, slope=2.7, norm=norm, min_energy=min_energy, max_energy=max_energy)
    model = KneeModel(*params)
    
    y_model = np.array([model.compute(E_i) for E_i in E])
    chi2 = np.sum(np.where(y_model > y_data,
                           ((y_model - y_data) / err_up) ** 2,
                           ((y_model - y_data) / err_lo) ** 2))
    return chi2

def fit_knee(initial_params, minEnergy, maxEnergy):
    def chi2_function(K, alpha, lgEb, dalpha, w):
        chi2 = 0.
        params = [K, alpha, lgEb, dalpha, w]
        datasets = [
            ('TALE_QGSJET-II-04_all_energy.txt', 1),
            ('GAMMA_SIBYLL_all_energy.txt', 1),
        #     ('DAMPE_H_energy.txt', 1, fDAMPE),
        #     ('CREAM_H_energy.txt', 1, fCREAM),
        #     ('CREAM_Fe_energy.txt', 26, fCREAM),
        ]
        for filename, norm in datasets:
             chi2 += experiment_chi2(filename, params, norm)
        return chi2        

    K, alpha, lgEb, dalpha, w = initial_params
    m = Minuit(chi2_function, K=K, alpha=alpha, lgEb=lgEb, dalpha=dalpha, w=w)

    m.limits['K'] = (0.5, 2.0)
    m.limits['alpha'] = (-0.6, 0.6)
    m.limits['w'] = (1., 10.)
    m.limits['dalpha'] = (-1.5, 1.5)
   
    m.fixed['w'] = True

    m.errordef = Minuit.LEAST_SQUARES
    m.migrad()
    m.hesse()
    #m.minos()

    dof = 0 # len(data_x) - m.nfit - 1

    print(m.errors)
    
    print(m) 

    return m.values, m.errors, m.covariance, m.fval, dof

if __name__ == "__main__":
    # Initial parameters
    initial_params = [1, 0.1, 6.1, -1.0, 10.]
    values, errors, covariance, fval, dof = fit_knee(initial_params, 1e5, 1e8)