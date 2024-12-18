from ase.io import read, write
import os,csv
from mace.calculators import mace_mp
import numpy as np
from scipy.constants import h, k, c,N_A
from ase import units
import argparse
from ase.calculators.dftd3 import DFTD3
from ase.build import make_supercell
import torch

hbar = 1.054571817e-34  
THz = 1e12  
kB_eV = 8.617333262145e-5 


def num_hess(system,calc):
    indicies =[i for i in range(len(system))]
    delta =1e-4
    ndim = 3
    hessian = np.zeros((len(indicies) * ndim, len(indicies) * ndim))
    atoms_h = system.copy()
    atoms_h.set_constraint()
    for i,index in enumerate(indicies):
        for j in range(ndim):
            atoms_i = atoms_h.copy()
            atoms_i.positions[index, j] += delta
            atoms_i.calc=calc
            forces_i = atoms_i.get_forces()
            
            atoms_j = atoms_h.copy()
            atoms_j.positions[index, j] -= delta
            atoms_j.calc=calc
            forces_j = atoms_j.get_forces()
            
            hessian[:, i * ndim + j] = -(forces_i - forces_j)[indicies].flatten() / (2 * delta)

    return hessian


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
        if nu<1e-3:
            counter+=1
            continue
        x = h * nu * 100 * c / (k * T)
        if np.isfinite(x) and x > 0:
            C_v_total += k * (x**2) * np.exp(x) / (np.exp(x) - 1)**2
        else:
            counter+=1
            pass
    print(f'Number of skiped frequencies: {counter}')
    return C_v_total * N_A


parser = argparse.ArgumentParser(description='Dir name.')
parser.add_argument('dir', type=str, help='Input dir name')
parser.add_argument('--csv_file', type=str, default=None, help='CSV file to expand')
parser.add_argument('--super', type=bool, default=None, help='Do you want to calculate it for 2x2x2 supercells')
args = parser.parse_args()

cif_path = f'./{args.dir}'
all_files = os.listdir(cif_path)

# Filter only files that end with '.cif'
cifs = [file for file in all_files if (file.endswith('.cif') or file.endswith('.traj'))]
cifs.sort()

if args.csv_file is not None:
    cifs = exclude_strings_from_list(cifs, args.csv_file)
    
mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda:1', )


if args.csv_file is None:
    header=["File_Name","Atomic_weight","Number_of_atoms"]
    for j in range(250,401,10):
        header.append(f"Cv_molar_{j}")
    for j in range(250,401,10):
        header.append(f"Cv_gravimetric_{j}")
    header.append(f"Opt")
    
    with open(f"C_v_screening_{args.dir}_s222_used_opt_sum_d3.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
    header.append(f"freq")
    with open(f"extended_C_v_screening_{args.dir}_s222_used_opt_sum_d3.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
    


transformation_matrix = [
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2]
        ]
d3_calc = DFTD3(xc='pbe',damping="bj")  # Using 'pbe' for the PBE functional

for i,cif in enumerate(cifs):
    print(cif)
    atoms = read(os.path.join(cif_path, cif))
    atoms.calc=None
    if args.super == True:
        atoms = make_supercell(atoms, transformation_matrix)
    n_atoms = len(atoms)
    masses = atoms.get_masses()
    try:
        hessian = mace_calc.get_hessian(atoms)  # Fill with your Hessian matrix values
        hessian=hessian.reshape(len(atoms)*3,len(atoms)*3)
        d3_hessian=num_hess(atoms,d3_calc).reshape(len(atoms)*3,len(atoms)*3)
        hessian= hessian+d3_hessian
        hessian += hessian.copy().T
        hessian/=2
        
        # motivated by ASE
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
        
        for T in range(250,401,10):
            C_v = heat_capacity(freq, T)
            C_vs.append(C_v)
        C_vs_per_g=(C_vs/(sum(masses)))
        C_vs_molar=list(np.array(C_vs)/(len(atoms)))
        row=[cif,sum(masses),len(atoms)]+list(C_vs_molar)+list(C_vs_per_g)+[True]
        
        with open(f"C_v_screening_{args.dir}_s222_used_opt_sum_d3.csv", 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        row = row+[list(freq)]
        with open(f"extended_C_v_screening_{args.dir}_s222_used_opt_sum_d3.csv", 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
    except Exception as e:
        row=[cif,sum(masses),len(atoms),"to large for gpu",e]
        
        with open(f"C_v_screening_{args.dir}_s222_used_opt_sum_d3.csv", 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        with open(f"extended_C_v_screening_{args.dir}_s222_used_opt_sum_d3.csv", 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)