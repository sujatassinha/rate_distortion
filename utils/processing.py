import numpy as np
from scipy.linalg import eigh

def center_data(x):
  """
  Centers the matrix by subtracting the mean.
  
  Parameters:
    x (numpy.ndarray): Input matrix.

  Returns:
    numpy.ndarray: Centered matrix.
  """
  return x - np.mean(x)

def compute_covariance_matrices(x_centered, m, n):
  """
  Computes row and column covariance matrices.

  Parameters:
    x_centered (numpy.ndarray): Centered data matrix.
    m (int): Number of rows.
    n (int): Number of columns.

  Returns:
    tuple: (row covariance matrix, column covariance matrix)
  """
  # Unbiased estimator of the row covariance matrix, hence / by N-1
  sigma_r = (x_centered @ x_centered.T) / (n - 1) # M x M
  
  # Unbiased estimator of the column covariance matrix, hence / by M-1
  sigma_c = (x_centered.T @ x_centered) / (m - 1)  # N x N
  return sigma_r, sigma_c

def compute_sorted_eigenvalues(sigma_r, sigma_c):
  """
  Computes and sorts the eigenvalues of row and column covariance matrices.

  Parameters:
    sigma_r (numpy.ndarray): Row covariance matrix.
    sigma_c (numpy.ndarray): Column covariance matrix.

  Returns:
    tuple: (sorted eigenvalues of sigma_r, sorted eigenvalues of sigma_c)
  """
  # Perform eigenvalue decomposition on row and column covariance matrix
  eigvals_r, _ = eigh(sigma_r)  # shape: (M, )
  eigvals_c, _ = eigh(sigma_c)  # shape: (N, )
  
  # Sort in descending order
  idx_r = np.argsort(eigvals_r)[::-1]
  idx_c = np.argsort(eigvals_c)[::-1]
  eigvals_r = eigvals_r[idx_r]
  eigvals_c = eigvals_c[idx_c]

  return eigvals_r, eigvals_c

def compute_kronecker_eigenvalues(eigvals_r, eigvals_c):
  """
  Computes and sorts the Kronecker product of eigenvalues.

  Parameters:
    eigvals_r (numpy.ndarray): Sorted eigenvalues of sigma_r.
    eigvals_c (numpy.ndarray): Sorted eigenvalues of sigma_c.

  Returns:
    numpy.ndarray: Sorted eigenvalues from Kronecker product.
  """
  eigvals = np.outer(eigvals_r, eigvals_c).ravel()
  eigvals = np.sort(eigvals)[::-1]  # Sort in descending order
  return eigvals