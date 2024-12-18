import os
from ase.io import write, read

# File name and path
stru = "RSM0011.cif"
path = r'./DFT_calculations/DFT_structures_only_cif'
unit_cell = read(os.path.join(path, stru))

# Generate and save supercells for all configurations from 1x1x1 to 4x4x4
for a in range(1, 5):
    for b in range(1, 5):
        if b>a:
            continue
        for c in range(1, 5):
            if c>a or c>b:
                continue
            print(f'Generating supercell with repeat pattern {a} x {b} x {c}')
            
            # Create a supercell by repeating the unit cell a, b, and c times in the three dimensions
            supercell = unit_cell.repeat((a, b, c))
            
            # Save the supercell to a .cif file
            filename =  f'{stru.split(".")[0]}_supercell_{a}x{b}x{c}.cif'
            write(filename, supercell)

print("Supercells saved to .cif files.")
