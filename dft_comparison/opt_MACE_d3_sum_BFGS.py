

from ase.io import read, write
import os,csv
from mace.calculators import mace_mp
from ase.optimize import FIRE,BFGS
import numpy as np
from scipy.constants import h, k, c,N_A
from ase import units
import argparse
from ase.filters import FrechetCellFilter
from ase.calculators.dftd3 import DFTD3
from ase.calculators.mixing import SumCalculator


parser = argparse.ArgumentParser(description='Dir name.')
parser.add_argument('dir', type=str, help='Dir name')
parser.add_argument('--csv_file', type=str, default=None, help='CSV file to expand')
args = parser.parse_args()

cif_path = f'./DFT_calculations/{args.dir}'
cifs = os.listdir(cif_path)
cifs.sort()

mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda:1', )

d3_calc = DFTD3(xc='pbe',damping="bj")  # Using 'pbe' for the PBE functional
mace_d3_sum_calc = SumCalculator([mace_calc, d3_calc])

csv_file = './opt_MACE_for_CVs_BFGS/optimization_results_BFGS.csv'
for i,cif in enumerate(cifs):
    converged=True
    print(cif)
    atoms = read(os.path.join(cif_path, cif))
    initial = atoms.copy()
    atoms.calc=mace_d3_sum_calc
    # Set up the 
    ecf = FrechetCellFilter(atoms)
    optimizer = BFGS(ecf)
    not_opt=False
  
    optimizer.run(fmax=0.005, steps=25000)
    write(f"./opt_MACE_for_CVs_BFGS/{cif.split('.')[0]}.traj", atoms)
    
    converged=optimizer.converged()
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cif, converged])
  

