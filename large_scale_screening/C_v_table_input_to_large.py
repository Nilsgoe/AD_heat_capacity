from ase.io import read, write
import os,csv
from mace.calculators import mace_mp
from ase.optimize import BFGS
import numpy as np
from scipy.constants import h, k, c,N_A
from ase import units
import argparse
from ase.filters import FrechetCellFilter

hbar = 1.054571817e-34  
THz = 1e12  
kB_eV = 8.617333262145e-5  

# Function to read the first column of the CSV file and return a set of strings
def read_first_column_from_csv(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        return {row[0] for row in reader}

# Function to filter the list of strings by excluding those present in the first column of the CSV file
def exclude_strings_from_list(string_list, csv_file):
    strings_to_exclude = read_first_column_from_csv(csv_file)
    return [s for s in string_list if s not in strings_to_exclude]

def heat_capacity(frequencies, T):
    C_v_total = 0.0
    counter=0
    for nu in frequencies:
        x = h * nu * 100 * c / (k * T)
        if np.isfinite(x) and x > 0:
            C_v_total += k * (x**2) * np.exp(x) / (np.exp(x) - 1)**2
        else:
            counter+=1
            pass
    print(f'Number of skiped frequencies: {counter}')
    return C_v_total * N_A

parser = argparse.ArgumentParser(description='Dir name.')
parser.add_argument('dir', type=str, help='input Dir name')
parser.add_argument('n_parts', type=int, help='Number of parts')
parser.add_argument('which_part', type=int, help='Which part')
parser.add_argument('--csv_file', type=str, default=None, help='CSV file to expand')
args = parser.parse_args()

cif_path = f'./ML/cifs/{args.dir}'
cifs = os.listdir(cif_path)
cifs.sort()
cif_parts = np.array_split(cifs, args.n_parts)
cifs=list(cif_parts[args.which_part])
print(cifs)
print(args.dir,args.which_part,len(cifs))



if args.csv_file is not None:
    cifs = exclude_strings_from_list(cifs, args.csv_file)

mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda:0', )

if args.csv_file is None or not os.path.exists(f'/./{args.csv_file}'):
    header=["File_Name","Atomic_weight","Number_of_atoms"]
    for j in range(250,401,10):
        header.append(f"Cv_molar_{j}")
    for j in range(250,401,10):
        header.append(f"Cv_gravimetric_{j}")
    header.append("Not_opt")
    with open(f"C_v_screening_opt_cell_{args.dir}_{args.which_part}.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(header)
    

for i,cif in enumerate(cifs):#[::10]):#cifs[:2])
    
    atoms = read(os.path.join(cif_path, cif))
    initial = atoms.copy()
    atoms.calc=mace_calc
    ecf = FrechetCellFilter(atoms)
    optimizer = BFGS(ecf)
    not_opt=False
    try:
        # Run the optimization with a specified number of maximum steps
        optimizer.run(fmax=0.05, steps=5000)
    except Exception as e:
        print(f"Optimization failed: {e}")

    # Retrieve the last known structure after max_steps
    if optimizer.get_number_of_steps() >= 5000:
        print(f"Taking the structure after {5000} steps.")
        not_opt=True
        atoms=initial.copy()
        atoms.calc=mace_calc
    n_atoms = len(atoms)
    masses = atoms.get_masses()
    try:
        hessian = mace_calc.get_hessian(atoms)  # Fill with your Hessian matrix values
        hessian=hessian.reshape(len(atoms)*3,len(atoms)*3)
        hessian += hessian.copy().T
        hessian/=2
        

        mass_weights = np.repeat(masses**-0.5, 3)

        omega2, vectors = np.linalg.eigh(mass_weights
                                        * hessian
                                        * mass_weights[:, np.newaxis])
        unit_conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
        energies = unit_conversion * omega2.astype(complex)**0.5

        modes = vectors.T.reshape(n_atoms * 3, n_atoms, 3)
        modes = modes * masses[np.newaxis, :, np.newaxis]**-0.5
        freq=np.real(energies)/ units.invcm
        
        
        print(i,cif)
        C_vs=[]
        # Example temperature
        for T in range(250,401,10):
            # Calculate heat capacity
            C_v = heat_capacity(freq, T)
            C_vs.append(C_v)
        C_vs_per_g=(C_vs/(sum(masses)))
        C_vs_molar=list(np.array(C_vs)/(len(atoms)))
        row=[cif,sum(masses),len(atoms)]+list(C_vs_molar)+list(C_vs_per_g)+[not_opt]
        
        with open(f"C_v_screening_opt_cell_{args.dir}_{args.which_part}.csv", 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(row)
    except:
        row=[cif,sum(masses),len(atoms),"to large for gpu"]
        
        with open(f"C_v_screening_opt_cell_{args.dir}_{args.which_part}.csv", 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(row)
        
    