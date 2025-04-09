import numpy as np
import scipy.linalg
from scipy.linalg import toeplitz
from scipy.signal import lfilter

def estimate_psd_csd(x1, x2, fs=1.0):
  """
  Estimate PSD of x1, PSD of x2, and CSD between them using FFT.

  Parameters:
    x1 (np.ndarray): Q x N array (Q realizations of x1)
    x2 (np.ndarray): Q x M array (Q realizations of x2)
    fs (float): Sampling frequency

  Returns:
    psd_x1 (np.ndarray): Averaged PSD of x1, shape (N//2 + 1,)
    freq_x1 (np.ndarray): Frequency axis for x1
    psd_x2 (np.ndarray): Averaged PSD of x2, shape (M//2 + 1,)
    freq_x2 (np.ndarray): Frequency axis for x2
    csd_x1x2 (np.ndarray): Cross-spectral density, shape (L,), where L = min(N, M)//2 + 1
    freq_csd (np.ndarray): Frequency axis for CSD
  """
  Q, N = x1.shape
  _, M = x2.shape
  L = min(N, M)

  # PSD x1
  psd1_all = [np.abs(np.fft.rfft(x1[q, :]))**2 for q in range(Q)]
  psd_x1 = np.mean(psd1_all, axis=0)
  freq_x1 = np.fft.rfftfreq(N, d=1/fs)

  # PSD x2
  psd2_all = [np.abs(np.fft.rfft(x2[q, :]))**2 for q in range(Q)]
  psd_x2 = np.mean(psd2_all, axis=0)
  freq_x2 = np.fft.rfftfreq(M, d=1/fs)

  # CSD x1 and x2 (using aligned, truncated realizations)
  csd_all = []
  for q in range(Q):
    fft_x1 = np.fft.rfft(x1[q, :L])
    fft_x2 = np.fft.rfft(x2[q, :L])
    csd_q = fft_x1 * np.conj(fft_x2)
    csd_all.append(csd_q)
  csd_x1x2 = np.mean(csd_all, axis=0)
  freq_csd = np.fft.rfftfreq(L, d=1/fs)

  return psd_x1, freq_x1, psd_x2, freq_x2, csd_x1x2, freq_csd


def estimate_psd_csd_old(X, fs=1.0):
  """
  Estimate the mean, covariance, power spectral density (PSD), and cross spectral density (CSD)
  of two 1D WSS Gaussian processes x1 and x2 given a matrix X, where each row is a realization
  of x1 and each column is a realization of x2. FFT is used for all spectral estimates.

  Parameters:
    X (np.ndarray): 2D real-valued array of shape (M, N), where rows are samples of x1,
                    and columns are samples of x2.
    fs (float): Sampling frequency used to scale the frequency axis (default is 1.0).

  Returns:
    mean_x1 (np.ndarray): Estimated mean of x1, shape (N,)
    cov_x1 (np.ndarray): Toeplitz covariance matrix of x1, shape (N, N)
    psd_x1 (np.ndarray): Power spectral density of x1, shape (N//2 + 1,)
    freq_x1 (np.ndarray): Frequency axis corresponding to psd_x1, shape (N//2 + 1,)

    mean_x2 (np.ndarray): Estimated mean of x2, shape (M,)
    cov_x2 (np.ndarray): Toeplitz covariance matrix of x2, shape (M, M)
    psd_x2 (np.ndarray): Power spectral density of x2, shape (M//2 + 1,)
    freq_x2 (np.ndarray): Frequency axis corresponding to psd_x2, shape (M//2 + 1,)

    csd (np.ndarray): Cross spectral density between x1 and x2 using their mean realizations,
                      shape (L,) where L = min(M, N)//2 + 1
    freq_csd (np.ndarray): Frequency axis corresponding to csd, shape (L,)
  """
  M, N = X.shape

  # Means
  mean_x1 = np.mean(X, axis=0)  # Shape (N,)
  mean_x2 = np.mean(X, axis=1)  # Shape (M,)

  # Sample covariances
  cov_x1 = np.cov(X, rowvar=False)  # (N, N)
  cov_x2 = np.cov(X, rowvar=True)   # (M, M)

  # Toeplitz projection (WSS assumption)
  def toeplitz_from_cov(cov):
    N = cov.shape[0]
    r = [np.mean(np.diag(cov, k)) for k in range(N)]
    return scipy.linalg.toeplitz(r)

  cov_x1 = toeplitz_from_cov(cov_x1)
  cov_x2 = toeplitz_from_cov(cov_x2)

  # PSDs via FFT of autocovariance
  psd_x1 = np.abs(np.fft.rfft(cov_x1[0, :]))
  psd_x2 = np.abs(np.fft.rfft(cov_x2[0, :]))
  freq_x1 = np.fft.rfftfreq(N, d=1/fs)
  freq_x2 = np.fft.rfftfreq(M, d=1/fs)

  # Use aligned row and column pairs to estimate cross-covariance
  L = min(M, N)
  cross_cov = np.zeros(L)
  for tau in range(L):
    vals = [X[i, tau] * X[tau, i] for i in range(L - tau)]
    cross_cov[tau] = np.mean(vals)
  # FFT of cross-covariance to get CSD
  csd = np.fft.rfft(cross_cov)
  freq_csd = np.fft.rfftfreq(L, d=1/fs)

  return (
    mean_x1, cov_x1, psd_x1, freq_x1,
    mean_x2, cov_x2, psd_x2, freq_x2,
    csd, freq_csd
  )

def project_to_wss(signal):
  """
  Project a signal to its WSS form by estimating its autocovariance
  and reconstructing a version with Toeplitz structure.

  Parameters:
    signal (np.ndarray): 1D array

  Returns:
    np.ndarray: WSS-approximated signal (whitened-like)
  """
  N = len(signal)
  acov = np.correlate(signal, signal, mode='full')[N-1:] / N
  toeplitz_cov = scipy.linalg.toeplitz(acov[:N])
  eigvals, eigvecs = np.linalg.eigh(toeplitz_cov + 1e-6 * np.eye(N))
  projection = eigvecs @ eigvecs.T @ signal  # project into WSS space
  return projection

def decompose_to_wss_components(X, Q=None):
  """
  Decompose a matrix X into Q pairs of correlated 1D WSS Gaussian processes
  using SVD + WSS projection, and reconstruct X from them.

  Parameters:
    X (np.ndarray): 2D matrix (MxN)
    Q (int): number of components (default = min(M, N))

  Returns:
    x1_list (list of np.ndarray): Q components for x1 (shape N,)
    x2_list (list of np.ndarray): Q components for x2 (shape M,)
    X_reconstructed (np.ndarray): full-rank approximation of X
  """
  M, N = X.shape
  Q = Q or min(M, N)

  U, S, Vt = np.linalg.svd(X, full_matrices=False)
  x1_list = []
  x2_list = []
  X_reconstructed = np.zeros_like(X)

  for q in range(Q):
    x2_q = np.sqrt(S[q]) * U[:, q]
    x1_q = np.sqrt(S[q]) * Vt[q, :]

    x2_wss = project_to_wss(x2_q)
    x1_wss = project_to_wss(x1_q)

    x1_list.append(x1_wss)
    x2_list.append(x2_wss)
    X_reconstructed += np.outer(x2_wss, x1_wss)

  return x1_list, x2_list, X_reconstructed

def compute_reconstruction_errors(X, X_hat):
  """
  Compute a suite of error metrics between original and reconstructed matrix.

  Parameters:
    X (np.ndarray): Original matrix
    X_hat (np.ndarray): Reconstructed matrix

  Returns:
    dict: Dictionary of named error metrics
  """
  diff = X - X_hat
  mse = np.mean(diff ** 2)
  rmse = np.sqrt(mse)
  mae = np.mean(np.abs(diff))
  fro_err = np.linalg.norm(diff, ord='fro')
  fro_rel = fro_err / np.linalg.norm(X, ord='fro')
  max_err = np.max(np.abs(diff))
  
  return {
    'frobenius_error': fro_err,
    'relative_frobenius_error': fro_rel,
    'rmse': rmse,
    'mae': mae,
    'max_error': max_err,
  }

def compute_mmse_filter(x1_flat, x2_flat, L):
  """
  Compute MMSE-optimal FIR filter h such that x2 ≈ h * x1

  Parameters:
    x1_flat (np.ndarray): Flattened x1, shape (T,)
    x2_flat (np.ndarray): Flattened x2, shape (T,)
    L (int): Filter length

  Returns:
    h (np.ndarray): FIR filter of length L
  """
  T = len(x1_flat)
  assert T >= L

  # Estimate autocorrelation R_x1
  r_x1 = np.correlate(x1_flat, x1_flat, mode='full')
  mid = len(r_x1) // 2
  r_x1 = r_x1[mid:mid + L] / T
  R = toeplitz(r_x1)

  # Estimate cross-correlation R_x1x2
  r_x1x2 = np.correlate(x2_flat, x1_flat, mode='full')
  r_x1x2 = r_x1x2[mid:mid + L] / T

  # Solve Wiener-Hopf equation
  h = np.linalg.solve(R, r_x1x2)
  return h