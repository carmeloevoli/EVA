import numpy as np
    
from fit_breaks_functions import fit_trpsbpl, dump_trpsbpl, fit_dblsbpl, dump_dblsbpl

def printit(values, errors):
    print(f'{values["dalpha"]:6.3e} {errors["dalpha"]:6.3e} {values["lgEb"]:6.3e} {errors["lgEb"]:6.3e}')


def printit2(values, errors):
    print(f'{values["dalpha_1"]:6.3e} {errors["dalpha_1"]:6.3e} {values["lgEb_1"]:6.3e} {errors["lgEb_1"]:6.3e}')
    print(f'{values["dalpha_2"]:6.3e} {errors["dalpha_2"]:6.3e} {values["lgEb_2"]:6.3e} {errors["lgEb_2"]:6.3e}')


if __name__== "__main__":
    # filename = 'TALE_QGSJET-II-04_all_energy.txt'
    # initial_params = [1e6, -0.15, 6.6, -0.3, 5., 7.3, 0.3, 5., 8.3, -0.2, 5.]
    # values, errors, covariance, fval, dof = fit_trpsbpl(initial_params, filename, 1e6, 1e9, 3.0, doFixSmoothness=True)
    # dump_trpsbpl(values, covariance, 'TALE_all_breaks.txt')   

    filename = 'GAMMA_SIBYLL_all_energy.txt'
    initial_params = [1e6, -0.15, 6.6, -0.3, 5., 7.3, 0.3, 5., 8.3, -0.2, 5.]
    values, errors, covariance, fval, dof = fit_trpsbpl(initial_params, filename, 1e6, 1e9, 3.0, doFixSmoothness=True)
    dump_trpsbpl(values, covariance, 'GAMMA_all_breaks.txt')   
    printit2(values, errors)

    # filename = 'TUNKA-133_QGSJET-01_all_energy.txt'
    # initial_params = [1e6, -0.15, 6.6, -0.3, 5., 7.3, 0.3, 5., 8.3, -0.2, 5.]
    # values, errors, covariance, fval, dof = fit_trpsbpl(initial_params, filename, 1e6, 1e9, 3.0, doFixSmoothness=True)
    # dump_trpsbpl(values, covariance, 'TUNKA_all_breaks.txt')   

    # filename = 'ICECUBE_SIBYLL_21_all_energy.txt'
    # initial_params = [1e6, -0.15, 6.6, -0.3, 5., 7.3, 0.3, 5., 8.3, -0.2, 5.]
    # values, errors, covariance, fval, dof = fit_trpsbpl(initial_params, filename, 1e6, 1e9, 3.0, doFixSmoothness=True)
    # dump_trpsbpl(values, covariance, 'ICECUBE_all_breaks.txt')   

    # filename = 'KGRANDE_QGSJET-II-04_all_energy.txt'
    # initial_params = [1e6, -0.15, 6.6, -0.3, 5., 7.3, 0.3, 5., 8.3, -0.2, 5.]
    # values, errors, covariance, fval, dof = fit_trpsbpl(initial_params, filename, 1e6, 3e9, 3.0, doFixSmoothness=True)
    # dump_trpsbpl(values, covariance, 'KGRANDE_all_breaks.txt')   

    # filename = 'KASCADE_QGSJET-01_all_energy.txt'
    # initial_params = [1e6, -0.15, 6.6, -0.3, 10., 7.3, 0.3, 10.]
    # values, errors, covariance, fval, dof = fit_dblsbpl(initial_params, filename, 1e6, 1e8, 3.0, doFixSmoothness=True)
    # dump_dblsbpl(values, covariance, 'KASCADE_QGSJET_all_breaks.txt')   

    # filename = 'TA_all_energy.txt'
    # initial_params = [1e6, -0.15, 6.6, -0.3, 10., 7.3, 0.3, 10., 8.3, -0.2, 10.]
    # values, errors, covariance, fval, dof = fit_trpsbpl(initial_params, filename, 1e6, 5e8, 3.0, doFixSmoothness=True)
    # dump_trpsbpl(values, covariance, 'TA_all_breaks.txt')   

