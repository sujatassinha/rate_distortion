import config
import matplotlib.pyplot as plt

def plot_rate_distortion_curve(r_of_d, d_values, index):
  save_dir = config.FIG_DIR
  plt.figure(figsize=(8,6))
  plt.plot(d_values, r_of_d, 'b-', label='Gaussian RD')
  plt.xlabel("Distortion D")
  plt.ylabel("Rate R(D) [bits/pixel]") # we have M*N samples
  plt.title("Rate-Distortion for 2D Gaussian Field")
  plt.legend()
  plt.grid(True)
  plt.legend()
  plt.savefig(save_dir+'rd_curve_'+str(index)+'.jpeg', dpi=300)
  plt.close()