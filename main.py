# Imports
from pathlib import Path
import numpy as np
from utils.checks import check_gaussian
from utils.processing import center_data, compute_covariance_matrices
from utils.processing import compute_sorted_eigenvalues, compute_kronecker_eigenvalues
from optimization.waterfilling import WaterFillingOptimizer
from utils.io_utils import plot_rate_distortion_curve

# Load the data
input_path = Path(__file__).parent / "../../datasets/CLOUDf48.bin.f32"
all_data = np.fromfile(input_path, dtype=np.float32).reshape(100, 500, 500)
indices, m,n = all_data.shape
for index in range(30, 80):
  x = all_data[index, :m, :n].reshape(m, n)

  # Check Gaussian
  check_gaussian(x, index)

  # Processing pipeline
  x_centered = center_data(x)
  sigma_r, sigma_c = compute_covariance_matrices(x_centered, m, n)
  eigvals_r, eigvals_c = compute_sorted_eigenvalues(sigma_r, sigma_c)
  eigvals = compute_kronecker_eigenvalues(eigvals_r, eigvals_c)

  # Initialize Optimizer
  optimizer = WaterFillingOptimizer(eigvals, m, n)

  # Compute rate-distortion function for multiple distortion values # -10
  d_values = np.concatenate(([0], np.logspace(-22, -13, 40))) #np.linspace(0, 1e-15, 10)#np.concatenate((np.linspace(0, 1e-10, 50), np.linspace(1e-10, 0.00001, 3)))
  #print(type(d_values))
  r_of_d = [optimizer.compute_rate_distortion(d) for d in d_values]
  print("Rate is", r_of_d)
  # Plot the curve
  plot_rate_distortion_curve(r_of_d, d_values, index)