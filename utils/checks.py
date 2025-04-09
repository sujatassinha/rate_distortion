import config
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

class GaussianRandomFieldChecker:
  def __init__(self, matrix, output_folder):
    """
    Initialize with a given MxN matrix
    """
    self.matrix = matrix
    self.output_folder = output_folder

  def check_gaussianity(self):
    """
    Checks whether the matrix follows a Gaussian distribution using:
    - Histogram + Gaussian fit
    - Q-Q plot (quantile-quantile)
    """
    values = self.matrix.flatten()

    # Plot histogram
    plt.figure(figsize=(12, 5))
    plt.hist(values, bins=50, density=True, alpha=0.6, color='b', label="Data Histogram")
    # Fit a Gaussian distribution
    mu, sigma = np.mean(values), np.std(values)
    x = np.linspace(np.min(values), np.max(values), 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r', label="Fitted Gaussian")
    plt.title("Histogram of Data Values vs. Gaussian Fit")
    plt.legend()
    save_path = os.path.join(self.output_folder, "histogram.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # Q-Q plot
    plt.figure(figsize=(6, 5))
    stats.probplot(values, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Data Values")
    save_path = os.path.join(self.output_folder, "qqplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

  def check_stationarity(self, window_size=50):
    """
    Checks stationarity by computing the variance of local means and variances in sub-windows.
    - A stationary field should have similar means/variances across different regions.    
    """
    M, N = self.matrix.shape
    sub_means, sub_vars = [], []

    # Divide matrix into non-overlapping sub-windows
    for i in range(0, M, window_size):
      for j in range(0, N, window_size):
        sub_matrix = self.matrix[i:i+window_size, j:j+window_size]
        sub_means.append(np.mean(sub_matrix))
        sub_vars.append(np.var(sub_matrix))

    # Plot histogram of means and variances
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(sub_means, bins=20, color='b', alpha=0.6)
    plt.title("Histogram of Local Means")

    plt.subplot(1, 2, 2)
    plt.hist(sub_vars, bins=20, color='r', alpha=0.6)
    plt.title("Histogram of Local Variances")
    save_path = os.path.join(self.output_folder, "histogram_local_mean_var.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    mean_variance = np.var(sub_means)
    print(f"Variance of means: {mean_variance}")

    return mean_variance < 1e-3  # Small variance of means indicates stationarity

  







'''
def check_gaussian(X, index):
  # Check if Gaussian
  data = X.flatten()
  # Q-Q Plot
  save_dir = config.FIG_DIR
  plt.figure(figsize=(6, 5))
  stats.probplot(data, dist="norm", plot=plt)
  plt.title("Q-Q Plot")
  plt.savefig(save_dir+'qqplots/'+str(index)+'.jpeg', dpi=300)
  plt.close()
'''