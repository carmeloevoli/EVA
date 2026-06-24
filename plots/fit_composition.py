import numpy as np
from iminuit import Minuit
from jacobi import propagate

class GalacticModel:
    R0 = 1e3  # Constant for scaling

    def __init__(self, I0_H, I0_He, I0_C, I0_O, I0_Mg, I0_Si, I0_Fe, alpha_H, alpha_He, alpha_N, Rb, dalpha, s):
        # Store the parameters in a dictionary for easier access
        self.params = {
            1: {"I0": I0_H, "alpha": alpha_H},
            2: {"I0": I0_He, "alpha": alpha_He},
            6: {"I0": I0_C, "alpha": alpha_N},
            8: {"I0": I0_O, "alpha": alpha_N},
            12: {"I0": I0_Mg, "alpha": alpha_N},
            14: {"I0": I0_Si, "alpha": alpha_N},
            26: {"I0": I0_Fe, "alpha": alpha_N},
        }
        self.Rb = Rb
        self.dalpha = dalpha
        self.s = s

    def compute(self, E, Z):
        if Z not in self.params:
            raise ValueError(f"Unsupported Z value: {Z}. Supported values are {list(self.params.keys())}.")

        # Access parameters based on Z value
        I0 = self.params[Z]["I0"]
        alpha = self.params[Z]["alpha"]

        # Compute R and model output y
        R = E / float(Z)
        y = I0 * (R / self.R0) ** -alpha
        y /= (1. + (R / self.Rb) ** self.s) ** (self.dalpha / self.s)

        return y


def experiment_chi2(filename, Z, params, norm=1.):
    E, y_data, err_lo, err_up = load_data(filename, slope=2.7, norm=norm, min_energy=0.7e3 * Z, max_energy=1e5 * Z)
    model = GalacticModel(*params)
    
    y_model = np.array([model.compute(E_i, Z) for E_i in E])
    chi2 = np.sum(np.where(y_model > y_data,
                           ((y_model - y_data) / err_up) ** 2,
                           ((y_model - y_data) / err_lo) ** 2))
    return chi2


def fit_phe(initial_params):
    def chi2_function(I0_H, I0_He, I0_C, I0_O, I0_Mg, I0_Si, I0_Fe, alpha_H, alpha_He, alpha_N, Rb, dalpha, s, fCALET, fDAMPE, fCREAM):
        chi2 = 0.
        params = [I0_H, I0_He, I0_C, I0_O, I0_Mg, I0_Si, I0_Fe, alpha_H, alpha_He, alpha_N, Rb, dalpha, s]
        datasets = [
            ('AMS-02_H_energy.txt', 1, 1),
            ('CALET_H_energy.txt', 1, fCALET),
            ('DAMPE_H_energy.txt', 1, fDAMPE),
            ('CREAM_H_energy.txt', 1, fCREAM),
            ('AMS-02_He_energy.txt', 2, 1),
            # ('CALET_He_energy.txt', 2, fCALET),
            ('DAMPE_He_energy.txt', 2, fDAMPE),
            ('CREAM_He_energy.txt', 2, fCREAM),
            ('AMS-02_C_energy.txt', 6, 1),
            ('CALET_C_energy.txt', 6, fCALET),
            ('CREAM_C_energy.txt', 6, fCREAM),
            ('AMS-02_O_energy.txt', 8, 1),
            ('CALET_O_energy.txt', 8, fCALET),
            ('CREAM_O_energy.txt', 8, fCREAM),
            ('AMS-02_Mg_energy.txt', 12, 1),
            ('CREAM_Mg_energy.txt', 12, fCREAM),
            ('AMS-02_Si_energy.txt', 14, 1),
            ('CREAM_Si_energy.txt', 14, fCREAM),
            ('AMS-02_Fe_energy.txt', 26, 1),
            ('CALET_Fe_energy.txt', 26, fCALET),
            ('CREAM_Fe_energy.txt', 26, fCREAM),
        ]
        for filename, Z, norm in datasets:
            chi2 += experiment_chi2(filename, Z, params, norm)
        return chi2

    I0_H, I0_He, I0_C, I0_O, I0_Mg, I0_Si, I0_Fe, alpha_H, alpha_He, alpha_N, Rb, dalpha, s, fCALET, fDAMPE, fCREAM = initial_params
    m = Minuit(chi2_function, I0_H=I0_H, I0_He=I0_He, I0_C=I0_C, I0_O=I0_O, I0_Mg=I0_Mg, I0_Si=I0_Si, I0_Fe=I0_Fe,
               alpha_H=alpha_H, alpha_He=alpha_He, alpha_N=alpha_N,
               Rb=Rb, dalpha=dalpha, s=s, 
               fCALET=fCALET, fDAMPE=fDAMPE, fCREAM=fCREAM)

    m.fixed['s'] = True
    m.limits['fCREAM'] = [0.7, 1.3]
    m.limits['fCALET'] = [0.9, 1.1]
    m.limits['fDAMPE'] = [0.9, 1.1]

    m.errordef = Minuit.LEAST_SQUARES
    m.simplex()
    m.migrad()
    m.hesse()

    print(m) 

    return m.values, m.errors, m.covariance, m.fval

def dump_fluxes(values, covariance):    
    print(f'fCREAM = {values["fCREAM"]:5.3f}')
    print(f'fCALET = {values["fCALET"]:5.3f}')
    print(f'fDAMPE = {values["fDAMPE"]:5.3f}')

    def get_flux(E, Z):
        def compute_model(E, params):
            p = params[0:13]
            model = GalacticModel(*p)
            return model.compute(E, Z)

        y, ycov = propagate(lambda p: compute_model(E, p), values, covariance)
        y_errors = np.sqrt(np.diag(ycov))

        return y, y_errors

    E = np.logspace(1, 9, 8000)
    I_H, I_H_err = get_flux(E, 1)
    I_He, I_He_err = get_flux(E, 2)
    I_C, I_C_err = get_flux(E, 6)
    I_O, I_O_err = get_flux(E, 8)
    I_Mg, I_Mg_err = get_flux(E, 12)
    I_Si, I_Si_err = get_flux(E, 14)
    I_Fe, I_Fe_err = get_flux(E, 26)

    with open("output/galactic_fluxes_fit.txt", "w") as f:
        for i in range(len(E)):
            f.write(f"{E[i]:.2e} {I_H[i]:.3e} {I_H_err[i]:.3e} "
                    f"{I_He[i]:.3e} {I_He_err[i]:.3e} "
                    f"{I_C[i]:.3e} {I_C_err[i]:.3e} "
                    f"{I_O[i]:.3e} {I_O_err[i]:.3e} "
                    f"{I_Mg[i]:.3e} {I_Mg_err[i]:.3e} "
                    f"{I_Si[i]:.3e} {I_Si_err[i]:.3e} "
                    f"{I_Fe[i]:.3e} {I_Fe_err[i]:.3e}\n")
            
def dump_all(values, covariance):    
    def get_all(E):
        def compute_model(E, params):
            p = params[0:13]
            model = GalacticModel(*p)
            all = model.compute(E, 1) 
            all += model.compute(E, 2) 
            all += model.compute(E, 6)
            all += model.compute(E, 8)
            all += model.compute(E, 12)
            all += model.compute(E, 14)
            all += model.compute(E, 26)
            return all

        y, ycov = propagate(lambda p: compute_model(E, p), values, covariance)
        y_errors = np.sqrt(np.diag(ycov))

        return y, y_errors

    E = np.logspace(1, 9, 8000)
    y, y_err = get_all(E)

    with open("output/galactic_all_fit.txt", "w") as f:
        for i in range(len(E)):
            f.write(f"{E[i]:.2e} {y[i]:.3e} {y_err[i]:.3e}\n")

def dump_light(values, covariance):    
    def get_all(E):
        def compute_model(E, params):
            p = params[0:13]
            model = GalacticModel(*p)
            all = model.compute(E, 1) 
            all += model.compute(E, 2) 
            return all

        y, ycov = propagate(lambda p: compute_model(E, p), values, covariance)
        y_errors = np.sqrt(np.diag(ycov))

        return y, y_errors

    E = np.logspace(1, 9, 8000)
    y, y_err = get_all(E)

    with open("output/galactic_light_fit.txt", "w") as f:
        for i in range(len(E)):
            f.write(f"{E[i]:.2e} {y[i]:.3e} {y_err[i]:.3e}\n")

def dump_lnA(values, covariance):    
    def get_lnA(E):
        def compute_model(E, params):
            p = params[0:13]
            model = GalacticModel(*p)
            all = model.compute(E, 1) 
            all += model.compute(E, 2) 
            all += model.compute(E, 6)
            all += model.compute(E, 8)
            all += model.compute(E, 12)
            all += model.compute(E, 14)
            all += model.compute(E, 26)
            lnA = np.log(4.) * model.compute(E, 2) 
            lnA += np.log(12.) * model.compute(E, 6)
            lnA += np.log(16.) * model.compute(E, 8)
            lnA += np.log(24.) * model.compute(E, 12)
            lnA += np.log(28.) * model.compute(E, 14)
            lnA += np.log(56.) * model.compute(E, 26)
            return lnA / all

        y, ycov = propagate(lambda p: compute_model(E, p), values, covariance)
        y_errors = np.sqrt(np.diag(ycov))

        return y, y_errors

    E = np.logspace(1, 9, 8000)
    y, y_err = get_lnA(E)

    with open("output/galactic_lnA_fit.txt", "w") as f:
        for i in range(len(E)):
            f.write(f"{E[i]:.2e} {y[i]:.3e} {y_err[i]:.3e}\n")

if __name__ == "__main__":
    # Initial parameters
    initial_params = [10e3, 9e3, 2e3, 3e3, 1e3, 1e3, 3e3, -0.12, -0.20, -0.20, 13e3, 0.21, 5., 1., 1., 1.]
    values, errors, covariance, fval = fit_phe(initial_params)
    dump_fluxes(values, covariance)
    dump_all(values, covariance)
    dump_lnA(values, covariance)
    dump_light(values, covariance)