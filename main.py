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











'''
# Center the data
mean_X = np.mean(X)
X_centered = X - mean_X

# Compute row covariance matrix
# unbiased estimator of the covariance matrix, hence / by N-1
Sigma_r = (X_centered @ X_centered.T) / (N - 1) # M X M

# Compute column covarinace matrix
# unbiased estimator of the covariance matrix, hence / by M-1
Sigma_c = (X_centered.T @ X_centered) / (M - 1)  # N X N

# Perform eigenvalue decomposition on row and column covariance matrix
eigvals_r, _ = eigh(Sigma_r)  # shape: (M, )
eigvals_c, _ = eigh(Sigma_c)  # shape: (N, )
#print("Sigma_r", Sigma_r.shape, "Sigma_c", Sigma_c.shape, "eigvals_r", eigvals_r.shape, "eigvals_c", eigvals_c.shape)

# Sort eigenvalues in descending order
idx_r = np.argsort(eigvals_r)[::-1]
idx_c = np.argsort(eigvals_c)[::-1]
eigvals_r = eigvals_r[idx_r]
eigvals_c = eigvals_c[idx_c]

# Build full covariance using Kronecker product (KLT basis)
eigvals = np.outer(eigvals_r, eigvals_c).ravel()
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
#print(eigvals.shape)
'''
# Water filling algorithm, binary search has time complexity N(logN)



'''
def find_alpha(eigvals, D, cumsums=0):
    """
    Find alpha >= 0 such that sum(min(eigvals[k], alpha)) = D,
    where eigvals is sorted in descending order, and 0 <= D <= sum(eigvals).

    This exploits the piecewise-linear property of phi(alpha).
    Returns alpha exactly in O(n) time.
    """
    n = len(eigvals)
    total_var = np.sum(eigvals)

    # Edge case: if distortion D is large enough to saturate all eigenvalues,
    # then alpha can exceed the largest eigenvalue => all d_k = lambda_k => sum=total_var.
    if D >= total_var:
        # In that scenario, alpha >= eigvals[0], meaning zero rate is needed.
        return eigvals[0]  # or any alpha >= eigvals[0]

    # cumsums[i] = sum of eigvals[0..i], inclusive
    cumsums = np.cumsum(eigvals)

    for m in range(n):
        # sum of the tail from (m+1) to (n-1):
        # i.e. sum(eigvals[m+1...n-1]) = total_var - cumsums[m]
        tail_sum = total_var - cumsums[m]

        # number of "big modes" that would be clipped to alpha is (m+1)
        # candidate alpha if exactly (m+1) eigenvalues are clipped:
        alpha_m = (D - tail_sum) / (m+1)

        if m == n-1:
            # We've included all eigenvalues. This must be the solution.
            return alpha_m

        # Check where alpha_m sits relative to eigvals[m] and eigvals[m+1].
        # By definition, alpha should be in [eigvals[m+1], eigvals[m]] to be consistent
        # with exactly (m+1) clipping.
        next_eigval = eigvals[m+1]  # safe because m < n-1
        curr_eigval = eigvals[m]

        if alpha_m >= next_eigval:
            # alpha_m is above the next break, so it's consistent with at least (m+1) modes clipped.
            if alpha_m <= curr_eigval:
                # Great, alpha_m is within [eigvals[m+1], eigvals[m]] => done.
                return alpha_m
            else:
                # alpha_m > curr_eigval => we haven't included enough modes in the clipping set.
                # We'll keep going. (m+1) was too small.
                pass
        else:
            # alpha_m < next_eigval => we've "over-included" modes. We actually need more modes.
            pass
        # Move on to m+1 in the loop.

    # Fallback (usually never reached unless numeric edge cases)
    # If we exit the loop, treat the last alpha as found:
    return 0.0

def water_filling(eigvals, D, cumsum_eigvals):
  n = len(eigvals)
  alpha = find_alpha(eigvals, D, cumsum_eigvals)
  mask = eigvals > alpha
  R_val = 0.5 * np.sum(np.log2(eigvals[mask] / alpha))
  R_val = R_val/(M*N) #bits/pixel
  return R_val
'''
'''
# compute rate distortion function
D_values = np.linspace(0, 1e-10, 50) 
R_of_D = []
it = 0
# Precompute cumulative sum for fast lookup
cumsum_eigvals = np.cumsum(eigvals)
for D in D_values:
  R_val = water_filling(eigvals, D, cumsum_eigvals)
  #print("R_val", R_val)
  R_of_D.append(R_val)

plt.figure(figsize=(8,6))
plt.plot(D_values, R_of_D, 'b-', label='Gaussian RD')
#plt.xlim([0, 15]) 
plt.xlabel("Distortion D")
plt.ylabel("Rate R(D) [bits/pixel]") # we have M*N samples
plt.title("Rate-Distortion for 2D Gaussian Field")
plt.legend()
plt.grid(True)
plt.legend()
plt.savefig('analysis/rd_curve_'+str(index)+'.jpeg', dpi=300)
plt.close()
'''




