import crdb
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def log_column_names(tab):
    """Log column names from the table for debugging purposes."""
    logging.info("Column names in table:")
    for icol, col_name in enumerate(tab.dtype.fields):
        logging.info("%2i: %s", icol, col_name)

def dump_datafile(quantity, energy_type, exp_name, sub_exp_name, filename, combo_level=0, energy_convert_level=0):
    """Query data from CRDB and save filtered results to a file."""
    filepath = Path('crdb') / filename
    # Ensure the 'crdb' directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Searching for {quantity} as a function of {energy_type} measured by {exp_name}")

    # Query data
    tab = crdb.query(quantity, energy_type=energy_type, combo_level=combo_level, energy_convert_level=energy_convert_level, exp_dates=exp_name)
    
    # Get unique sub-experiment names and ADS codes
    sub_exp_names = set(tab["sub_exp"])
    logging.info("Number of datasets found: %d", len(sub_exp_names))
    logging.debug("Sub-experiment names: %s", sub_exp_names)
    
    ads_codes = set(tab["ads"])
    logging.debug("ADS codes: %s", ads_codes)

    # Filter data for the specified sub-experiment
    items = [i for i, sub in enumerate(tab["sub_exp"]) if sub == sub_exp_name]
    if not items:
        logging.error("No data found for sub-experiment '%s'.", sub_exp_name)
        raise ValueError(f"No data found for the specified sub-experiment: {sub_exp_name}")
    logging.info("Number of data entries: %d", len(items))

    # Write data to file
    logging.info(f"Dumping data to {filepath}")
    with open(filepath, 'w') as f:
        f.write("#source: CRDB\n")
        f.write(f"#Quantity: {quantity}\n")
        f.write(f"#EnergyType: {energy_type}\n")
        f.write(f"#Experiment: {exp_name}\n")
        f.write(f"#ADS: {tab['ads'][items[0]]}\n")
        f.write("#E_lo - E_up - y - errSta_lo - errSta_up - errSys_lo - errSys_up\n")
        
        for e_bin, value, err_sta, err_sys in zip(tab["e_bin"][items], tab["value"][items], tab["err_sta"][items], tab["err_sys"][items]):
            f.write(f"{e_bin[0]:10.5e} {e_bin[1]:10.5e} {value:10.5e} {err_sta[0]:10.5e} {err_sta[1]:10.5e} {err_sys[0]:10.5e} {err_sys[1]:10.5e}\n")
    logging.info("Data dump completed.\n")

if __name__ == "__main__":
    datasets = [
        ("H", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_H_rigidity.txt"),
        ("H", "EKN", "CALET", "CALET (2015/10-2021/12)", "CALET_H_kEnergy.txt"),
        ("H", "EK", "DAMPE", "DAMPE (2016/01-2018/06)", "DAMPE_H_kEnergy.txt"),
        ("H", "EKN", "CREAM", "CREAM-I+III (2004+2007)", "CREAM_H_kEnergy.txt"),   
        ("H", "EKN", "CREAM", "ISS-CREAM (2017/08-2019/02)", "ISS-CREAM_H_kEnergy.txt"),   
        ("H", "ETOT", "NUCLEON", "NUCLEON-KLEM (2015/07-2017/06)", "NUCLEON_H_energy.txt"),   
        ("H", "R", "PAMELA", "PAMELA (2006/07-2008/12)", "PAMELA_H_rigidity.txt"),   
        ("He", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_He_rigidity.txt"),
        ("He", "EK", "CALET", "CALET (2015/10-2022/04)", "CALET_He_kEnergy.txt"),
        ("He", "EK", "DAMPE", "DAMPE (2016/01-2020/06)", "DAMPE_He_kEnergy.txt"),
        ("He", "EKN", "CREAM", "CREAM-I+III (2004+2007)", "CREAM_He_kEnergyPerNucleon.txt"),   
        ("He", "R", "PAMELA", "PAMELA (2006/07-2008/12)", "PAMELA_He_rigidity.txt"),
        ("C", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_C_rigidity.txt"),
        ("C", "EKN", "CALET", "CALET (2015/10-2022/02)", "CALET_C_kEnergyPerNucleon.txt"),
        ("C", "EKN", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_C_kEnergyPerNucleon.txt"),   
        ("O", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_O_rigidity.txt"),
        ("O", "EKN", "CALET", "CALET (2015/10-2019/10)", "CALET_O_kEnergyPerNucleon.txt"),
        ("O", "EKN", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_O_kEnergyPerNucleon.txt"),   
        ("Mg", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_Mg_rigidity.txt"),
        ("Mg", "EKN", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_Mg_kEnergyPerNucleon.txt"),   
        ("Si", "R", "AMS02", "AMS02 (2011/05-2018/05)", "AMS-02_Si_rigidity.txt"),
        ("Si", "EKN", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_Si_kEnergyPerNucleon.txt"),   
        ("Fe", "R", "AMS02", "AMS02 (2011/05-2019/10)", "AMS-02_Fe_rigidity.txt"),
        ("Fe", "EKN", "CALET", "CALET (2016/01-2020/05)", "CALET_Fe_kEnergyPerNucleon.txt"),
        ("Fe", "EKN", "CREAM", "CREAM-II (2005/12-2006/01)", "CREAM_Fe_kEnergyPerNucleon.txt"),
        ("AllParticles", "ETOT", "NUCLEON", "NUCLEON-KLEM (2015/07-2017/06)", "NUCLEON_all_energy.txt"),
        ("AllParticles", "ETOT", "GAMMA", "GAMMA (2003/01-2007/12) SIBYLL", "GAMMA_SIBYLL_all_energy.txt"),
        ("AllParticles", "ETOT", "HAWC", "HAWC (2018-2019) QGSJet-II-04", "HAWC_QGSJET-II-04_all_energy.txt"),
        ("AllParticles", "ETOT", "TALE", "TALE (2014/06/-2016/03) QGSJet-II-03", "TALE_QGSJET-II-04_all_energy.txt"),
        ("AllParticles", "ETOT", "TUNKA", "TUNKA-133 Array (2009/10-2012/04) QGSJet01", "TUNKA-133_QGSJET-01_all_energy.txt"),
        ("AllParticles", "ETOT", "KASCADE", "KASCADE (1996/10-2002/01) QGSJet01", "KASCADE_QGSJET-01_all_energy.txt"),
        ("AllParticles", "ETOT", "KASCADE", "KASCADE (1996/10-2002/01) SIBYLL 2.1", "KASCADE_SIBYLL_21_all_energy.txt"),
        ("AllParticles", "ETOT", "KASCADE", "KASCADE-Grande (2003/01-2009/03) QGSJet-II-04", "KGRANDE_QGSJET-II-04_all_energy.txt"),
        ("AllParticles", "ETOT", "KASCADE", "KASCADE-Grande (2003/12-2011/10) SIBYLL2.3", "KGRANDE_SIBYLL_23_all_energy.txt"),        
        ("AllParticles", "ETOT", "IceTop", "IceTop (2016/05-2017/04) QGSJet-II-04", "ICETOP_QGSJET-II-04_all_energy.txt"),
        ("AllParticles", "ETOT", "IceTop", "IceTop (2016/05-2017/04) SIBYLL2.1", "ICETOP_SIBYLL_21_all_energy.txt"),
        ("AllParticles", "ETOT", "IceCube", "IceCube+IceTop (2010/06-2013/05) SIBYLL2.1", "ICECUBE_SIBYLL_21_all_energy.txt"),
        ("AllParticles", "ETOT", "Auger", "Auger SD750+SD1500 (2014/01-2018/08)", "AUGER_all_energy.txt"),
        ("AllParticles", "ETOT", "Telescope", "Telescope Array Hybrid (2008/01-2015/05)", "TA_all_energy.txt"),
        ("H-He-group", "ETOT", "Tibet", "Tibet III (2000/11-2004/10) QGSJet01", "TIBET_light_energy.txt"),
    ]

    try:
        for quantity, energy_type, exp_name, sub_exp_name, filename in datasets:
            dump_datafile(quantity, energy_type, exp_name, sub_exp_name, filename)
        logging.info("All datasets processed successfully.")
    except Exception as e:
        logging.critical("Processing stopped due to error: %s", e)
        logging.info("Program terminated due to a critical error.")
