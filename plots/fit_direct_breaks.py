import numpy as np
    
from fit_breaks_functions import fit_sbpl, fit_dblsbpl, dump_sbpl, dump_dblsbpl

def printit(values, errors):
    print(f'{values["dalpha"]:6.3e} {errors["dalpha"]:6.3e} {values["lgEb"]:6.3e} {errors["lgEb"]:6.3e}')

def printit2(values, errors):
    print(f'{values["dalpha_1"]:6.3e} {errors["dalpha_1"]:6.3e} {values["lgEb_1"]:6.3e} {errors["lgEb_1"]:6.3e}')
    print(f'{values["dalpha_2"]:6.3e} {errors["dalpha_2"]:6.3e} {values["lgEb_2"]:6.3e} {errors["lgEb_2"]:6.3e}')

if __name__== "__main__":
    filename = 'PAMELA_H_energy.txt'
    initial_params = [6e3, 0.20, 2.5, 0.40, 5.]
    values, errors, covariance, fval, dof = fit_sbpl(initial_params, filename, 1e2, 1e6, 2.7, doFixSmoothness=True)
    dump_sbpl(values, covariance, 'PAMELA_H_break.txt')
    printit(values, errors)

    filename = 'AMS-02_H_energy.txt'
    initial_params = [6e3, 0.20, 2.5, 0.40, 5.]
    values, errors, covariance, fval, dof = fit_sbpl(initial_params, filename, 1e2, 1e6, 2.7, doFixSmoothness=True)
    dump_sbpl(values, covariance, 'AMS-02_H_break.txt')
    printit(values, errors)

    filename = 'CREAM_H_energy.txt'
    initial_params = [9e3, -0.20, 4.1, -0.40, 5.]
    values, errors, covariance, fval, dof = fit_sbpl(initial_params, filename, 1e3, 1e6, 2.7, doFixSmoothness=True)
    dump_sbpl(values, covariance, 'CREAM_H_break.txt')
    printit(values, errors)

    filename = 'ISS-CREAM_H_energy.txt'
    initial_params = [9e3, -0.20, 4.1, -0.40, 5.]
    values, errors, covariance, fval, dof = fit_sbpl(initial_params, filename, 1e3, 1e6, 2.7, doFixSmoothness=True)
    dump_sbpl(values, covariance, 'ISS-CREAM_H_break.txt')
    printit(values, errors)

    filename = 'NUCLEON_H_energy.txt' # KLEM
    initial_params = [9e3, -0.20, 4.1, -0.40, 5.]
    values, errors, covariance, fval, dof = fit_sbpl(initial_params, filename, 1e3, 1e6, 2.7, doFixSmoothness=True)
    dump_sbpl(values, covariance, 'NUCLEON_H_break.txt')
    printit(values, errors)

    filename = 'CALET_H_energy.txt'
    initial_params = [8e3, 0.20, 2.5, 0.40, 5., 4., -0.40, 5.]
    values, errors, covariance, fval, dof = fit_dblsbpl(initial_params, filename, 1e2, 1e6, 2.7, doFixSmoothness=True)
    dump_dblsbpl(values, covariance, 'CALET_H_break.txt')
    printit2(values, errors)

    filename = 'DAMPE_H_energy.txt'
    initial_params = [8e3, 0.12, 2.8, 0.30, 5., 4., -0.3, 5.]
    values, errors, covariance, fval, dof = fit_dblsbpl(initial_params, filename, 1e2, 1e6, 2.7, doFixSmoothness=True)
    dump_dblsbpl(values, covariance, 'DAMPE_H_break.txt')
    printit2(values, errors)

