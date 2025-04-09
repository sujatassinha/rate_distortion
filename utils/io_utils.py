import config
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_spectral_analysis(freqs, S_x1x1, S_x2x2, S_x2x1, labels, save_path):
  # Plot PSDs
  plt.plot(freqs, np.mean(S_x1x1, axis=0).real, label=labels[0])
  plt.plot(freqs, np.mean(S_x2x2, axis=0).real, label=labels[1])
  plt.plot(freqs, np.abs(np.mean(S_x2x1, axis=0)).real, label=labels[2], color='tab:purple')
  plt.title("Spectral Densities")
  plt.xlabel("Frequency")
  plt.ylabel("Power")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(save_path)
  plt.show()
  plt.close()



def plot_psd(psd_x1, psd_x2, csd, fx1, fx2, fcsd, save_path):
  plt.figure(figsize=(10, 4))
  plt.plot(fx1, psd_x1, label='PSD of x1 (rows)')
  plt.plot(fx2, psd_x2, label='PSD of x2 (cols)')
  plt.plot(fcsd, np.abs(csd), label='|CSD of x1 and x2|')
  plt.xlabel('Frequency')
  plt.ylabel('Power')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()

def plot_all_psds_and_csd(freqs, S_x1x1, S_x2x2, S_x1x2, save_path, titles, max_r=10, log_scale=False):
  r = min(S_x1x1.shape[1], max_r)  # Plot up to max_r components

  # Pre-compute y-axis bounds
  def safe_log(y):
    y = np.nan_to_num(y, nan=1e-12, posinf=1e-12, neginf=1e-12)
    return 10 * np.log10(np.clip(y, 1e-12, None))
    #return 10 * np.log10(np.maximum(y, 1e-12))

  Y1 = np.array([safe_log(S_x1x1[:, k].real) if log_scale else S_x1x1[:, k].real for k in range(r)])
  Y2 = np.array([safe_log(S_x2x2[:, k].real) if log_scale else S_x2x2[:, k].real for k in range(r)])
  Y12 = np.array([safe_log(np.abs(S_x1x2[:, k])) if log_scale else np.abs(S_x1x2[:, k]) for k in range(r)])

  ymin = min(Y1.min(), Y2.min(), Y12.min())
  ymax = max(Y1.max(), Y2.max(), Y12.max())

  plt.figure(figsize=(18, 4))

  for i, (Y, title) in enumerate(zip([Y1, Y2, Y12], titles)):
    plt.subplot(1, 3, i + 1)
    for k in range(r):
      plt.plot(freqs, Y[k], label=f"[{k}]")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Power (dB)" if log_scale else "Power")
    plt.ylim(ymin, ymax)
    plt.xlim(freqs[0], freqs[-1])
    plt.legend()
    plt.grid(True)

  plt.tight_layout()
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()


def plot_error(mse_history, save_path):
  plt.figure(figsize=(8,6))
  plt.plot(list(range(len(mse_history))), mse_history, color='b', linestyle='solid', label='Reconstruction Error')
  plt.xlabel("epochs")
  plt.ylabel("MSE")
  plt.legend()
  plt.tight_layout() 
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()

def plot_data(data, colorbar_label, title_name, save_path):
  plt.figure(figsize=(8, 6))
  plt.imshow(data)  # Can use 'viridis' or 'gray'
  plt.colorbar(label=colorbar_label)  # Adds a color scale
  plt.title(title_name)
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()


def plot_bargraph(data, x_label, y_label, title_name, save_path):
  # Plot bar chart
  plt.figure(figsize=(8, 6))
  plt.bar(range(data.shape[0]), data, color='green', alpha=0.7)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(title_name)
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()

def qqplots(data, save_path):
  plt.figure(figsize=(8, 6))
  stats.probplot(data, dist="norm", plot=plt)
  plt.title("QQ Plot vs Normal Distribution")
  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()

def side_by_side_barplot(x1, x2, save_path):
  # Create bar chart
  fig, axs = plt.subplots(1, 2, figsize=(12, 4))

  # Bar plot for x1 (row components)
  axs[0].bar(range(len(x1)), x1, color='skyblue')
  axs[0].set_title('Components of x1')
  axs[0].set_xlabel('x1 Index')
  axs[0].set_ylabel('Value')
  axs[0].grid(True, linestyle='--', alpha=0.5)

  # Bar plot for x2 (column components)
  axs[1].bar(range(len(x2)), x2, color='salmon')
  axs[1].set_title('Components of x2')
  axs[1].set_xlabel('x2 Index')
  axs[1].set_ylabel('Value')
  axs[1].grid(True, linestyle='--', alpha=0.5)

  plt.savefig(save_path, dpi=300, bbox_inches='tight')
  plt.show()
  plt.close()


def plot_rate_distortion_curve(r_of_d, d_values, index=None, save_path=None, save_dir=config.FIG_DIR):
  plt.figure(figsize=(8,6))
  plt.plot(d_values, r_of_d, color='b', linestyle='solid', label='Gaussian RD')
  plt.xlabel("mse")
  plt.ylabel("bit rate")
  plt.legend()
  plt.tight_layout() # we have M*N samples
  plt.title("Rate-Distortion for 2D Gaussian Field")
  if save_path==None:
    plt.savefig(save_dir+'/rd_curve_'+str(index)+'.jpeg', dpi=300)
    plt.close()
  else:
    plt.savefig(save_path, dpi=300)
    plt.close()