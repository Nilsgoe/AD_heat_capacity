from ase.io import read, write
import os,csv
from mace.calculators import mace_mp
from ase.optimize import BFGS
import numpy as np
import argparse
from ase.filters import FrechetCellFilter

# Function to read the first column of the CSV file and return a set of strings
def read_first_column_from_csv(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        return {row[0] for row in reader}

# Function to filter the list of strings by excluding those present in the first column of the CSV file
def exclude_strings_from_list(string_list, csv_file):
    strings_to_exclude = read_first_column_from_csv(csv_file)
    return [s for s in string_list if s not in strings_to_exclude]


parser = argparse.ArgumentParser(description='Dir name.')
parser.add_argument('dir', type=str, help='Dir name')
parser.add_argument('n_parts', type=int, help='Number of parts')
parser.add_argument('which_part', type=int, help='Which part')
parser.add_argument('--csv_file', type=str, default=None, help='CSV file to expand')
args = parser.parse_args()

cif_path = f'./ML/cifs/{args.dir}'
cifs = os.listdir(cif_path)
cifs.sort()
cif_parts = np.array_split(cifs, args.n_parts)
cifs=list(cif_parts[args.which_part])

if args.csv_file is not None:
    cifs = exclude_strings_from_list(cifs, args.csv_file)

mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda:0', )

        
for i,cif in enumerate(cifs):#[::10]):#cifs[:2])
    
    atoms = read(os.path.join(cif_path, cif))
    initial = atoms.copy()
    atoms.calc=mace_calc
    # Set up the optimizer
    ecf = FrechetCellFilter(atoms)
    optimizer = BFGS(ecf)
    opt=True
    try:
        # Run the optimization with a specified number of maximum steps
        optimizer.run(fmax=0.05, steps=1000)
    except Exception as e:
        print(f"Optimization failed: {e}")
    if optimizer.get_number_of_steps() >= 1000:
        print(f"Taking the structure after {1000} steps.")
        opt=False
    with open(f"{args.dir}_mace_bfgs.csv", mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([cif, opt])
    
    write(f"opt_cif_mace_hessian_bfgs/{cif.split('.')[0]}_mace_bfgs.traj",atoms)

    