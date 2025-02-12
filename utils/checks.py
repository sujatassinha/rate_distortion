import config
import matplotlib.pyplot as plt
import scipy.stats as stats

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