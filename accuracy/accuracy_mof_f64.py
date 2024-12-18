### Calcing the accuray of the AD compared to numerical

import time,csv,os
from mace.calculators import mace_mp
import numpy as np
from ase.io import read

def mae(list1, list2):
  if len(list1) != len(list2):
    raise ValueError("Lists must have the same length")
  absolute_differences = np.abs(np.subtract(list1, list2))
  return np.mean(absolute_differences)

def rmse(list1, list2):
  if len(list1) != len(list2):
    raise ValueError("Lists must have the same length")

  # Calculate squared differences
  squared_differences = np.square(np.subtract(list1, list2))

  # Mean of squared differences
  mean_squared_difference = np.mean(squared_differences)

  # Return the square root of the mean
  return np.sqrt(mean_squared_difference)

calc = mace_mp(model="medium", dispersion=False, default_dtype="float64",device='cuda:1', )#device='cpu')
structures=['12022N2.cif',"20560N3.cif","AFR.cif","NPT.cif","RSM0059.cif","RSM0122.cif","RSM0788.cif","RSM1440.cif","RSM1854.cif",'SAS.cif']
path= r'./DFT_calculations/DFT_structures'

with open('accuracy_conversion_mofs_2.csv', 'w', newline='') as csvfile:
              writer = csv.writer(csvfile)
              writer.writerow(['Name','Systemsize', 'h', 'Max_diff',"RMSE","MAE"])
for stru in structures:
  initial = read(os.path.join(path, stru))
  initial.calc = calc
  
  s=time.time()
  h_autograd=calc.get_hessian(atoms=initial)
  print("h:",h_autograd)

  e=time.time()
  print(f"This system need {e-s} seconds")

  s=time.time()
  indicies =[i for i in range(len(initial))]
  deltas =[1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
  ndim = 3

  atoms_h = initial.copy()

  atoms_h.set_constraint()
  calc_f32 = mace_mp(model="medium", dispersion=False, default_dtype="float32",device='cuda:1', )
  for delta in deltas:
      hessian = np.zeros((len(indicies) * ndim, len(indicies) * ndim))
      for i,index in enumerate(indicies):
          for j in range(ndim):

              atoms_i = atoms_h.copy()
              atoms_i.positions[index, j] += delta
              atoms_i.calc=calc_f32
              forces_i = atoms_i.get_forces()
              
              atoms_j = atoms_h.copy()
              atoms_j.positions[index, j] -= delta
              atoms_j.calc=calc_f32
              forces_j = atoms_j.get_forces()
              
              hessian[:, i * ndim + j] = -(forces_i - forces_j)[indicies].flatten() / (2 * delta)

      e=time.time()
      hessian= hessian.reshape(-1,len(initial),3)

      print(f"This system need {e-s} seconds")

      max_diff=np.max(np.abs(hessian-h_autograd))
      rmse_= rmse(hessian,h_autograd)
      mae_ = mae(hessian,h_autograd)
      
      
      with open('accuracy_conversion_mofs.csv', 'a+', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow([stru,len(initial),delta,  max_diff,rmse_,mae_])
      csvfile.close()