# Imports
import h5py
import os
import numpy as np
from scipy.signal import lfilter
from utils.processing import compute_rate_minimum, compute_distortion_from_sets
from utils.io_utils import plot_data, qqplots, plot_psd, plot_spectral_analysis
from utils.statistical_approx import estimate_psd_csd
from utils.statistical_approx import compute_reconstruction_errors, decompose_to_wss_components
from utils.statistical_approx import compute_mmse_filter
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load Data
q = 400
file_path = '30slice-precip-npn.h5'
with h5py.File(file_path, 'r') as h5_file:
  all_keys= list(h5_file.keys()) # use one of the keys 
  for dataset_name in [all_keys[-1]]:
    # 1 Create a folder to save the figures
    output_folder = "temp_output/checks_precip/transformed_data/"+str(dataset_name)
    os.makedirs(output_folder, exist_ok=True)

    # 2 Obtain input data
    x = h5_file[dataset_name][:]
    m,n=x.shape
    print(f"Loaded '{dataset_name}' of shape: {x.shape}")

# Step 3: Approximate the two WSS correlated GRP
x1_list, x2_list, x_hat = decompose_to_wss_components(x, Q=q)
# 1. Obtain approximation error
error_metrics = compute_reconstruction_errors(x, x_hat)
for k, v in error_metrics.items():
  print(f"{k:30s}: {v:.6f}")
# 1. Plot the reconstruction data
colorbar_label = "Intensity"
title_name = f"For dataset {dataset_name}, total no. of non-zero elements in matrix is {len(x_hat[x_hat != 0])}"
save_path = os.path.join(output_folder, "x_hat.png")
plot_data(x_hat, colorbar_label, title_name, save_path)
# 2. Obtain M and N realizations of GRPs
x1 = np.array(x1_list) # Q realizations of length N
x2 = np.array(x2_list) # Q realizations of length M
title_names = "Process X1", "Process X2"
save_path = os.path.join(output_folder, "x1_hat.png")
plot_data(x1, colorbar_label, title_names[0], save_path)
save_path = os.path.join(output_folder, "x2_hat.png")
plot_data(x2, colorbar_label, title_names[1], save_path)

# Or plot if x1 and x2 are gaussian on average
save_path = os.path.join(output_folder, "qqplot_x1.png")
qqplots(np.mean(x1, axis=1), save_path)
save_path = os.path.join(output_folder, "qqplot_x2.png")
qqplots(np.mean(x2, axis=1), save_path)

# Step 5: Obtain PSD and CSD
# x1: shape (Q, N) -> Q realizations of x1 of length N
# x2: shape (Q, M) -> Q realizations of x2 of length M
_, Nfft = x1.shape
X1 = np.fft.rfft(x1, axis=1)  # shape: (q, 251)
X2 = np.fft.rfft(x2, axis=1)  # shape: (q, 251)
S_x1x1 = (X1 * np.conj(X1)) / Nfft # shape: (q, 251)
S_x2x2 = (X2 * np.conj(X2)) / Nfft # shape: (q, 251)
S_x2x1 = (X2 * X1.conjugate()) / Nfft # shape: (q, 251)
freqs = np.fft.rfftfreq(Nfft, d=1.0) # shape: (251,)
save_path = os.path.join(output_folder, "spectral_analysis_x1x2.png")
labels = ["PSD X1", "PSD X2", "|CSD X1 X2|"]
plot_spectral_analysis(freqs, S_x1x1, S_x2x2, S_x2x1, labels, save_path)

# Step 8: Use Weiner filter and obtain X1 and Y2
H = S_x2x1 / (S_x1x1 + 1e-100) # shape: (q, 251)
Y2 = X2 - H * X1  # shape: (q, 251)
S_y2y2 = (Y2 * np.conj(Y2)) / Nfft    # shape: (q, 251)
S_y2x1 = (Y2 * X1.conjugate()) / Nfft # shape: (q, 251)
save_path = os.path.join(output_folder, "spectral_analysis_x1y2.png")
labels = ["PSD X1", "PSD Y2", "|CSD X1 Y2|"]
plot_spectral_analysis(freqs, S_x1x1, S_y2y2, S_y2x1, labels, save_path)

# Step 9: Whiten the processes X1 and Y2 
W1 = X1 / np.sqrt(S_x1x1)           # shape: (q, 251)
W2 = Y2 / np.sqrt(S_y2y2 + 1e-100)  # shape: (q, 251)

# ------------------------------------------------------
def compute_rate_distortion(S_x1x1, S_y2y2, S_x2x1, H, T, lambda1, lambda2):
  """
  Computes R_T, δ_x1^T, δ_x2^T for given spectral densities and Lagrange multipliers.

  Parameters:
    S_x1x1: np.ndarray, shape (q, n_freqs)
    S_y2y2: np.ndarray, shape (q, n_freqs)
    S_x2x1: np.ndarray, shape (q, n_freqs)
    T: int, duration
    lambda1, lambda2: floats, Lagrange multipliers

  Returns:
    R_T, δ_x1^T, δ_x2^T: np.ndarray of shape (q,)
  """

  eps = 1e-100
  q, n_freqs = S_x1x1.shape
  abs_H_sq = np.abs(H)**2  # (n_freqs,)
  Lambda = lambda1 + lambda2 * abs_H_sq

  # Broadcast Lambda to shape (q, n_freqs)
  Lambda_b = np.broadcast_to(Lambda, (q, n_freqs))  # shape (q, n_freqs)
  abs_H_sq_b = np.broadcast_to(abs_H_sq, (q, n_freqs))

  # I1: S_x1x1 * Lambda > 1
  I1 = (S_x1x1 * Lambda_b) > 1
  # I2: lambda2 * S_y2y2 > 1
  I2 = (lambda2 * S_y2y2) > 1

  # R_T: Apply log where condition is satisfied
  term1 = np.zeros_like(S_x1x1)
  term2 = np.zeros_like(S_y2y2)
  term1[I1] = np.log2(S_x1x1[I1] * Lambda_b[I1] + eps)
  term2[I2] = np.log2(lambda2 * S_y2y2[I2] + eps)
  R_T = (1 / (2 * T)) * (np.sum(term1, axis=1) + np.sum(term2, axis=1))

  # δ_x1^T
  delta_x1_term1 = np.zeros_like(S_x1x1)
  delta_x1_term1[I1] = 1 / Lambda_b[I1]
  delta_x1_term2 = np.zeros_like(S_x1x1)
  delta_x1_term2[~I1] = S_x1x1[~I1]
  delta_x1_T = (1 / T) * np.sum(delta_x1_term1 + delta_x1_term2, axis=1)

  # δ_x2^T
  term1, term2 = np.zeros_like(S_x1x1), np.zeros_like(S_x1x1)
  term1[I1] = abs_H_sq_b[I1] / Lambda_b[I1]
  term2[~I1] = S_x1x1[~I1] * abs_H_sq_b[~I1]
  term3, term4 = np.zeros_like(S_y2y2), np.zeros_like(S_y2y2)
  term3[I2] = 1 / lambda2
  term4[~I2] = S_y2y2[~I2]

  delta_x2_T = (1 / T) * (
    np.sum(term1 + term2, axis=1) + np.sum(term3 + term4, axis=1)
  )
  return R_T, delta_x1_T, delta_x2_T

lambda1_values = np.logspace(-2, 5, 50)
lambda2_values = np.logspace(-2, 5, 50)

rate_grid = np.zeros((len(lambda1_values), len(lambda2_values)))
dist_x1_grid = np.zeros_like(rate_grid)
dist_x2_grid = np.zeros_like(rate_grid)

for i, lambda1 in enumerate(lambda1_values):
  for j, lambda2 in enumerate(lambda2_values):
    R, dx1, dx2 = compute_rate_distortion(S_x1x1, S_y2y2, S_x2x1, H, T=Nfft, lambda1=lambda1, lambda2=lambda2)
    rate_grid[i, j] = np.mean(R)
    dist_x1_grid[i, j] = np.mean(dx1)
    dist_x2_grid[i, j] = np.mean(dx2)

from mpl_toolkits.mplot3d import Axes3D
# 3D Surface: R_T vs. δ_x1 and δ_x2
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(dist_x1_grid, dist_x2_grid, rate_grid, cmap='viridis')
ax.set_xlabel("Distortion Process 1")
ax.set_ylabel("Distortion Process 2")
ax.set_zlabel("Rate")
ax.set_title("Rate-Distortion Surface")
plt.tight_layout()
save_path = os.path.join(output_folder, "rate_distortion_surface.png")
plt.savefig(save_path)
plt.legend()
plt.show()
plt.close()

#print("Rate is", rate_grid)
#print("D1 is", dist_x1_grid)

#here
# Flatten and compute total distortion
total_distortion = (dist_x1_grid + dist_x2_grid).flatten()
rate_flat = rate_grid.flatten()

# Plot the 2D curve
plt.figure(figsize=(8, 5))
# Plot 1
# Sort by total distortion for clean plotting
sorted_indices = np.argsort(total_distortion)
total_distortion_sorted = total_distortion[sorted_indices]
rate_sorted = rate_flat[sorted_indices]
plt.plot(total_distortion_sorted, rate_sorted, color='r', label='Rate vs Total Distortion')
# Plot 2
sorted_indices = np.argsort(dist_x1_grid.flatten())
distortion1_sorted = dist_x1_grid.flatten()[sorted_indices]
rate_sorted = rate_flat[sorted_indices]
plt.plot(distortion1_sorted, rate_sorted, color='b', label='Rate vs Distortion on X1')
# Plot 3
sorted_indices = np.argsort(dist_x2_grid.flatten())
distortion2_sorted = dist_x2_grid.flatten()[sorted_indices]
rate_sorted = rate_flat[sorted_indices]
plt.plot(distortion2_sorted, rate_sorted, color='g', label='Rate vs Distortion on X2')
plt.xlabel("Distortion")
plt.ylabel("Rate (bits/s)")
plt.title("Rate vs. Distortion Curve")
plt.grid(True)
plt.legend()
save_path = os.path.join(output_folder, "rate_distortion_curve.png")
plt.savefig(save_path)
plt.show()
plt.close()

# --- Compute Pareto Front ---
dist_total_flat = (dist_x1_grid + dist_x2_grid).flatten()
rate_flat = rate_grid.flatten()

# Sort by distortion (increasing)
sorted_idx = np.argsort(dist_total_flat)
dist_sorted = dist_total_flat[sorted_idx]
rate_sorted = rate_flat[sorted_idx]

# Build Pareto front: decreasing rate with increasing distortion
pareto_dist = []
pareto_rate = []

min_rate_so_far = np.inf
for d, r in zip(dist_sorted, rate_sorted):
  if r < min_rate_so_far:
    pareto_dist.append(d)
    pareto_rate.append(r)
    min_rate_so_far = r

pareto_dist = np.array(pareto_dist)
pareto_rate = np.array(pareto_rate)

# --- Plot with Pareto Front ---
# Create a DataFrame
pareto_df = pd.DataFrame({
    'Distortion': pareto_dist,
    'Rate': pareto_rate
})
# Save to CSV
csv_path = os.path.join(output_folder, "pareto_curve.csv")
pareto_df.to_csv(csv_path, index=False)
plt.figure(figsize=(8, 5))
plt.scatter(dist_total_flat, rate_flat, color='gray', alpha=0.3, label='All points')
plt.plot(pareto_dist, pareto_rate, color='crimson', linewidth=2.5, label='Pareto Front')
plt.xlabel("Total Distortion")
plt.ylabel("Rate (bits/s)")
plt.title("Rate vs. Total Distortion with Pareto Front")
plt.legend()
plt.grid(True)
save_path = os.path.join(output_folder, "rate_vs_total_distortion_pareto.png")
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()


print("D=", np.array(total_distortion))





'''
dist_total_grid = dist_x1_grid + dist_x2_grid  # shape: (len(λ1), len(λ2))
flat_dist = dist_total_grid.flatten()
flat_rate = rate_grid.flatten()
sorted_indices = np.argsort(flat_dist)[::-1]
D_sorted = flat_dist[sorted_indices]
R_sorted = flat_rate[sorted_indices]
R_min = np.minimum.accumulate(R_sorted)
D_curve = D_sorted[::-1]
R_curve = R_min[::-1]
plt.plot(D_curve, R_curve, color='red')
plt.xlabel("Distortion D")
plt.ylabel("Rate R(D)")
plt.title("Corrected Rate-Distortion Curve")
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(output_folder, "rate_distortion_curve.png")
plt.savefig(save_path)
plt.show()
plt.close()

dist_total_grid = dist_x1_grid + dist_x2_grid  # shape: (len(λ1), len(λ2))
# Flatten and sort by distortion
flat_dist = dist_total_grid.flatten()
flat_rate = rate_grid.flatten()
print("Distortion:", flat_dist.shape)
print("Rate", flat_rate.shape)
sorted_indices = np.argsort(flat_dist)
flat_dist_sorted = flat_dist[sorted_indices]
flat_rate_sorted = flat_rate[sorted_indices]

# Flip so we compute min(R) for decreasing D
reversed_D = flat_dist_sorted[::-1]
reversed_R = flat_rate_sorted[::-1]

# Get the *lower envelope* (min R for each increasing D)
min_R_reversed = np.minimum.accumulate(reversed_R)
final_D = reversed_D[::-1]
final_R = min_R_reversed[::-1]

# Plot classic R(D) shape
plt.figure(figsize=(6, 4))
plt.plot(final_D, final_R, color='red', linewidth=2)
plt.xlabel("Distortion $D$")
plt.ylabel("Rate $R(D)$")
plt.title("Rate-Distortion Curve (Lower Envelope)")
plt.grid(True)
plt.tight_layout()
save_path = os.path.join(output_folder, "rate_distortion_curve.png")
plt.savefig(save_path)
plt.show()
plt.close()
'''
# ----------------------------------------------













'''
from scipy.optimize import minimize
def find_min_rate_given_D(S_x1x1, S_y2y2, H, D_target=0, init_guess=(1e-1, 1e-1), verbose=False):
  T, R = S_x1x1.shape

  def objective(lmbdas):
    lambda1, lambda2 = lmbdas
    if lambda1 <= 0 or lambda2 <= 0:
      return 1e6  # Penalize negative lambdas
    R_T, I1, I2 = compute_rate_minimum(S_x1x1, S_y2y2, H, lambda1, lambda2)
    return np.sum(R_T) / (T * R)  # normalize rate

  def constraint(lmbdas):
    lambda1, lambda2 = lmbdas
    R_T, I1, I2 = compute_rate_minimum(S_x1x1, S_y2y2, H, lambda1, lambda2)
    delta_x1, delta_x2 = compute_distortion_from_sets(S_x1x1, S_y2y2, H, lambda1, lambda2, I1, I2)
    total_distortion = np.sum(delta_x1 + delta_x2)
    return D_target - total_distortion  # we want this to be >= 0

  # Initial guess (positive lambdas)
  x0 = np.array(init_guess)
  bounds = [(1e-6, 1e4), (1e-6, 1e4)]  # enforce positive lambda
  constraints = {'type': 'ineq', 'fun': constraint}
  result = minimize(
    objective, 
    x0, 
    method='trust-constr', 
    bounds=bounds, 
    constraints=[constraints],
    options={'disp': False}
    )

  if result.success:
    lambda1, lambda2 = result.x
    R_T, I1, I2 = compute_rate_minimum(S_x1x1, S_y2y2, H, lambda1, lambda2)
    delta_x1, delta_x2 = compute_distortion_from_sets(S_x1x1, S_y2y2, H, lambda1, lambda2, I1, I2)
    total_rate = np.sum(R_T) / (T * R)
    total_dist = np.sum(delta_x1 + delta_x2)
    if verbose:
      print(f"Success: D_target={D_target:.4f}, achieved D={total_dist:.4f}, R={total_rate:.4f}")
    return total_dist, total_rate
  else:
    if verbose:
      print("Optimization failed:", result.message)
    return np.inf, np.inf # invalid


D_values = np.logspace(np.log10(0.01), np.log10(0.8), 20)
R_list = []
D_list = []

for D in D_values:
  D_actual, R_actual = find_min_rate_given_D(S_x1x1, S_y2y2, H, D_target=D, verbose=True)
  R_list.append(R_actual)
  D_list.append(D_actual)

# Plot
save_path = os.path.join(output_folder, "rate_distortion_optimized.png")
plot_distortion_vs_rate(D_list, R_list, save_path)
'''
'''
X1_hat = W1 * np.sqrt(S_x1x1)
X2_hat = H * X1_hat + W2 * np.sqrt(S_y2y2 + 1e-100)

x1_recon = np.fft.irfft(X1_hat, axis=0)
x2_recon = np.fft.irfft(X2_hat, axis=0)
err_x1 = compute_reconstruction_errors(x1, x1_recon)
err_x2 = compute_reconstruction_errors(x2, x2_recon)

print("Reconstruction Error for x1_hat:")
for k, v in err_x1.items():
  print(f"{k:30s}: {v:.6f}")

print("\nReconstruction Error for x2_hat:")
for k, v in err_x2.items():
  print(f"{k:30s}: {v:.6f}")
'''