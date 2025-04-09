# Imports
from pathlib import Path
import numpy as np
import h5py
import os
#from utils.checks import check_gaussian
from utils.processing import center_data, compute_covariance_matrices
from utils.processing import compute_sorted_eigenvalues, compute_kronecker_eigenvalues
from optimization.waterfilling import WaterFillingOptimizer
from utils.io_utils import plot_rate_distortion_curve
from mpi4py.futures import MPICommExecutor
from sklearn.preprocessing import PowerTransformer
import csv

# Load Gaussian Data
file_path = '30slice-precip-npn.h5'
with h5py.File(file_path, 'r') as h5_file:
  all_keys= list(h5_file.keys()) # use one of the keys 
  for dataset_name in [all_keys[-1]]:
    # Step 0: Create a folder to save the figures
    output_folder = "analysis_kronecker/checks_precip/transformed_data/"+str(dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    # Step 1: Obtain input data
    x = h5_file[dataset_name][:]
    print(f"Loaded '{dataset_name}' of shape: {x.shape}")
    m,n = x.shape

    # Processing pipeline
    x_centered = center_data(x)
    sigma_r, sigma_c = compute_covariance_matrices(x_centered, m, n)
    eigvals_r, eigvals_c = compute_sorted_eigenvalues(sigma_r, sigma_c)
    eigvals_r, eigvals_c = np.maximum(eigvals_r, 0), np.maximum(eigvals_c, 0)
    eigvals = compute_kronecker_eigenvalues(eigvals_r, eigvals_c)

    # Initialize Optimizer
    optimizer = WaterFillingOptimizer(eigvals, m, n)

    # Compute rate-distortion function for multiple distortion values # -10
    #d_values = np.logspace(-15, -11, num=15)#np.concatenate((np.logspace(-15, -10, 64), np.logspace(-10, 0, 64)))  # #np.linspace(0, 1e-15, 10)#np.concatenate((np.linspace(0, 1e-10, 50), np.linspace(1e-10, 0.00001, 3)))
    d_values = np.linspace(0, 0.03, 200) #np.logspace(np.log10(0.0001), np.log10(3), 100)
    r_of_d = [optimizer.compute_rate_distortion(d) for d in d_values]
    # Plot the curve
    print("\nRunning slice", dataset_name)
    plot_rate_distortion_curve(r_of_d, d_values, dataset_name)
    print("Rate is", r_of_d)
    
    csv_save_path = os.path.join(output_folder, "rate_distortion_data.csv")
    with open(csv_save_path, mode='w', newline='') as f:
      writer = csv.writer(f)
      writer.writerow(['Distortion_D', 'Rate_R'])  # header
      for D, R in zip(d_values, r_of_d):
        writer.writerow([D, R])



'''
# Load the data
input_path = Path(__file__).parent / "../../datasets/CLOUDf48.bin.f32"
input_path = Path(__file__).parent / "../../datasets/CLOUDf48_transformed.bin.f32"
all_data = np.fromfile(input_path, dtype=np.float32).reshape(100, 500, 500)
indices, m,n = all_data.shape

# we are specifically talking about the ith index
for index in range(23, 24): #10, 90, 5):
  x = all_data[index, :m, :n].reshape(m, n)
  
  # Transform image to gaussian
  #x = x**(50)
  #x = x.reshape(m, n)

  # Check Gaussian
  #check_gaussian(x, index)

  # Processing pipeline
  x_centered = center_data(x)
  sigma_r, sigma_c = compute_covariance_matrices(x_centered, m, n)
  eigvals_r, eigvals_c = compute_sorted_eigenvalues(sigma_r, sigma_c)
  eigvals_r, eigvals_c = np.maximum(eigvals_r, 0), np.maximum(eigvals_c, 0)
  eigvals = compute_kronecker_eigenvalues(eigvals_r, eigvals_c)

  # Initialize Optimizer
  optimizer = WaterFillingOptimizer(eigvals, m, n)

  # Compute rate-distortion function for multiple distortion values # -10
  #d_values = np.logspace(-15, -11, num=15)#np.concatenate((np.logspace(-15, -10, 64), np.logspace(-10, 0, 64)))  # #np.linspace(0, 1e-15, 10)#np.concatenate((np.linspace(0, 1e-10, 50), np.linspace(1e-10, 0.00001, 3)))
  d_values = np.logspace(0, 7, num=100)
  r_of_d = [optimizer.compute_rate_distortion(d) for d in d_values]
  # Plot the curve
  print("\nRunning index", index)
  plot_rate_distortion_curve(r_of_d, d_values, index)
  print("Rate is", r_of_d)
  #print("d is ", d_values)
'''