import numpy as np
from scipy.linalg import eigh
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def compute_distortion_from_sets(S_x1x1, S_y2y2, H, lambda1, lambda2, I1, I2):
  """
  Compute distortions delta_x1 and delta_x2 using spectral-domain derivation.

  Inputs:
    S_x1x1: PSD of x1 [shape: (F, r)]
    S_y2y2: PSD of y2 [shape: (F, r)]
    H: Wiener filter [shape: (F, r)]
    lambda1, lambda2: positive scalars
    I1, I2: boolean masks [shape: (F, r)]

  Returns:
    delta_x1, delta_x2: distortions [shape: (r,)]
  """
  T, R = S_x1x1.shape
  delta_x1 = np.zeros(R)
  delta_x2 = np.zeros(R)

  for r in range(R):
    h2 = np.abs(H[:, r])**2

    # --- δ_x1 ---
    part1 = np.sum(1.0 / (lambda1 + lambda2 * h2[I1[:, r]]))
    part2 = np.sum(S_x1x1[~I1[:, r], r])
    delta_x1[r] = (1 / T) * (part1 + part2)

    # --- δ_x2 ---
    term1 = np.sum(h2[I1[:, r]] / (lambda1 + lambda2 * h2[I1[:, r]]))
    term2 = np.sum(S_x1x1[~I1[:, r], r] * h2[~I1[:, r]])
    term3 = np.sum(1.0 / lambda2 * np.ones_like(S_y2y2[I2[:, r], r]))
    term4 = np.sum(S_y2y2[~I2[:, r], r])
    
    delta_x2[r] = (1 / T) * (term1 + term2 + term3 + term4)

  return delta_x1, delta_x2


def compute_rate_minimum(S_x1x1, S_y2y2, H, lambda1, lambda2):
  """
  Compute the rate R_T under Sakrison-style spectral-domain constraints.

  Inputs:
    S_x1x1: PSD of x1 [shape: (F, r)]
    S_y2y2: PSD of y2 [shape: (F, r)]
    H: Wiener filter [shape: (F, r)]
    lambda1, lambda2: positive scalars

  Returns:
    R_T: rate per mode [shape: (r,)]
    I1, I2: boolean masks for valid frequency bins [shape: (F, r)]
  """
  T, R = S_x1x1.shape
  gain = lambda1 + lambda2 * np.abs(H)**2
  term1 = S_x1x1 * gain
  term2 = lambda2 * S_y2y2

  # Define I1 and I2 sets (indicator masks)
  I1 = term1 > 1
  I2 = term2 > 1

  R_T = np.zeros(R)
  for r in range(R):
    R1 = np.sum(np.log2(term1[I1[:, r], r]))
    R2 = np.sum(np.log2(term2[I2[:, r], r]))
    R_T[r] = (1 / (2 * T)) * (R1 + R2)

  return R_T, I1, I2


def gp_approximate_2d_field(x, axis=0, noise=1e-3):
  """
  Given a matrix X of shape (m, n), where each row (or column) is
  a realization of a Gaussian process, fit a GP to the common GP.

  Parameters:
    x (np.ndarray): The 2D matrix
    axis (int): 0 = GP over columns (rows as realizations)
                1 = GP over rows (columns as realizations)
    noise (float): Noise level for GP model

  Returns:
    inputs (np.ndarray): Grid locations
    mean_prediction (np.ndarray): Predicted GP mean
    std_prediction (np.ndarray): Predicted GP std
  """
  if axis == 0:
    m, n = x.shape
    inputs = np.tile(np.arange(n), m).reshape(-1, 1)     # shape (m*n, 1)
    outputs = x.reshape(-1, 1) 
  
  kernel = ConstantKernel(1.0) * RBF(length_scale=20.0, length_scale_bounds=(1e-2, 1e5))
  gp = GaussianProcessRegressor(kernel=kernel, alpha=noise**2, normalize_y=True)
  gp.fit(inputs, outputs)

  # Predict GP at each unique point
  unique_inputs = np.unique(inputs)
  pred_mean, pred_std = gp.predict(unique_inputs.reshape(-1, 1), return_std=True)
  return unique_inputs, pred_mean, pred_std


def decompose_row_column(x, alpha1=0.5, alpha2=0.5, max_iter=1000, tol=0, patience=10):
  """
  Given a 2D array x of shape (M, N), find x1(i) and x2(j) such that:
    x'(i,j) = x(i,j) - mean(x)
    is approximated by x1(i) + x2(j)
    
  Uses iterative Orthogonality Principle.
  
  Parameters:
    x (numpy.ndarray): Input matrix.
    alpha1, alpha2 (float): step sizes (learning rates) for X1 and X2 updates
    max_iter (int): Maximum number of iterations (default: 100)
    tol (float): Tolerance for convergence (default: 1e-8)

  Returns:
    tuple: (x1, x2)
      - x1 (np.ndarray): Row process of shape (M,)
      - x2 (np.ndarray): Column process of shape (N,)
  """
  m, n = x.shape

  # Center the data
  mean_x = np.mean(x)
  x_centered = x - mean_x

  # Initialize the row (x1) and column (x2) processes
  x1 = np.zeros(m, dtype=x.dtype)
  x2 = np.zeros(n, dtype=x.dtype)

  # Compute initial MSE
  mse_history = []
  _, current_mse = compute_mse(x, x1, x2, mean_x)
  mse_history.append(current_mse)

  # Patience tracking
  patience_counter = 0

  # Update x1 and x2 using a gradient-based approach.
  for iteration in range(max_iter):
    residual = x_centered - (x1[:, None] + x2[None, :])
    # --- Update X1 ---
    for i in range(m):
      x1[i] += alpha1 * np.mean(residual[i, :])
      
    # --- Update X2 ---
    for j in range(n):
      x2[j] += alpha2 * np.mean(residual[:, j])

    # Check new MSE
    _, new_mse = compute_mse(x, x1, x2, mean_x)
    mse_history.append(new_mse)

    # Print iteration info 
    #print(f"Iteration {iteration+1}/{max_iter}: MSE = {new_mse:.6f}")

    # Check immediate tolerance condition
    if abs(new_mse - current_mse) < tol:
      patience_counter += 1
    else:
      patience_counter = 0

    if patience_counter >= patience:
      print("Stopping early due to small MSE change.")
      break

    current_mse = new_mse

  return x1, x2, mean_x, mse_history

def approximate_field(x1, x2, mean_x):
  """
  Given 1D arrays x1 and x2, and a scalar mean mean_x = np.mean(x) compute the 2D approximation:
    x̂(i,j) = x1(i) + x2(j) + x_centered

  This reconstructs the original field using the additive decomposition model.

  Parameters:
    x1 (np.ndarray): Row process of shape (M,)
    x2 (np.ndarray): Column process of shape (N,)
    mean_x (scalar): Scalar mean of the original matrix

  Returns:
    np.ndarray: Reconstructed 2D matrix x̂ of shape (M, N)
  """
  x_hat = x1[:, None] + x2[None, :] + mean_x
  return x_hat

def compute_mse(x, x1, x2, mean_x):
    """
    Mean Squared Error between x and the approximation
      x_hat(i,j) = x1(i)+x2(j)+mean_x.

    Parameters:
      x (numpy.ndarray): Input matrix.
      x1 (np.ndarray): Row process of shape (M,)
      x2 (np.ndarray): Column process of shape (N,)
      mean_x (scalar): Scalar mean of the original matrix
    
    Returns:
      float: Mean squared error between x and its reconstruction x̂,
             computed as MSE = mean((x - x̂)^2)
    """
    x_hat = approximate_field(x1, x2, mean_x)
    mse = np.mean((x - x_hat)**2)
    return x_hat, mse

def whiten_two_processes(x1, x2):
  """
  """
  # If lengths differ, we choose a reference length for covariance estimation.
  l = min(len(x1), len(x2))
  x1 = x1[:l]
  x2 = x2[:l]

  # Estimate covarinace matrix
  var1 = np.var(x1, ddof=1)
  var2 = np.var(x2, ddof=1)
  cov12 = np.mean((x1 - np.mean(x1)) * 
                  (x2 - np.mean(x2)))
  Sigma = np.array([[var1,     cov12],
                    [cov12,    var2 ]], dtype=x1.dtype)
  
  # Invert the sqrt of Sigma
  # Easiest robust approach is eigen-decomposition:
  vals, vecs = np.linalg.eig(Sigma)  # Sigma = V diag(vals) V^T
  # sqrtD^{-1} = diag(1/sqrt(vals))
  sqrt_inv = np.diag(1.0 / np.sqrt(vals))
  Sigma_inv_sqrt = vecs @ sqrt_inv @ np.linalg.inv(vecs)
    
    

















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