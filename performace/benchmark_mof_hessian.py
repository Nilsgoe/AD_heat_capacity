import numpy as np
import csv,time,os
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators import mace_mp
import argparse

def calc_num_time(system,calc):
    s=time.time()
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

            # Numerical second derivative (central difference)
            
            hessian[:, i * ndim + j] = -(forces_i - forces_j)[indicies].flatten() / (2 * delta)

    e=time.time()
    diff= e-s
    return diff


parser = argparse.ArgumentParser(description='Process a CIF file.')
parser.add_argument('filename', type=str, help='The CIF filename to process')
parser.add_argument("csv",type=str,help="csv_file")
args = parser.parse_args()
print(f"Processing CIF file: {args.filename}")
print(args.filename)
cif_path = r'./3d_super_cells'
atoms = read(os.path.join(cif_path, args.filename))


mace_calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda', )

atoms.calc=mace_calc
s=time.time()
mace_calc.get_hessian(atoms)
e=time.time()
time_autograd=e-s
 
time_numerical = calc_num_time(atoms,mace_calc)



csv_file_path = f"./{args.csv}"

# Read the existing data and update the relevant row
rows = []
with open(csv_file_path, mode='r', newline='') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        if row[0] == args.filename:
            row[1] = len(atoms)
            row[2] = time_autograd
            row[3] = time_numerical
        rows.append(row)

# Write the updated row back to the CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Updated CSV with results for {args.filename}")
