import numpy as np

class WaterFillingOptimizer:
  """
  Class to compute the water-filling solution for a given distortion constraint.
  """

  def __init__(self, eigvals, m, n):
    """
    Initializes the WaterFillingOptimizer with eigenvalues and matrix dimensions.

    Parameters:
      eigvals (numpy.ndarray): Sorted eigenvalues in descending order.
      m (int): Number of rows in the original data.
      n (int): Number of columns in the original data.
    """
    self.eigvals = np.sort(eigvals)[::-1]  # Ensure eigenvalues are sorted
    self.m = m
    self.n = n
    self.cumsum_eigvals = np.cumsum(self.eigvals)  # Precompute cumulative sum

  def find_alpha(self, distortion):
    """
    Finds alpha >= 0 such that sum(min(eigvals[k], alpha)) = distortion.

    This exploits the piecewise-linear property of phi(alpha).
    Returns alpha exactly in O(n) time.

    Parameters:
      distortion (float): Given distortion constraint.

    Returns:
      float: Optimal alpha value.
    """
    num_eigvals = len(self.eigvals)
    total_var = np.sum(self.eigvals)
    
    # Edge case: If distortion is large enough to saturate all eigenvalues
    if distortion >= total_var:
      return self.eigvals[0]  # Alpha equals the largest eigenvalue

    # Compute cumulative sums
    cumsums = np.cumsum(self.eigvals)

    for m in range(num_eigvals):
      # Compute sum of remaining eigenvalues
      tail_sum = total_var - cumsums[m]

      # Compute candidate alpha for exactly (m+1) clipped eigenvalues
      alpha_m = (distortion - tail_sum) / (m + 1)

      if m == num_eigvals - 1:
        return alpha_m  # All eigenvalues are included, return the computed alpha

      # Check if alpha_m lies within [eigvals[m+1], eigvals[m]]
      next_eigval = self.eigvals[m + 1]  
      curr_eigval = self.eigvals[m]

      if next_eigval <= alpha_m <= curr_eigval:
        return alpha_m  # Found valid alpha

    return 0.0  # Fallback case (should not be reached)
  
  
  def compute_rate_distortion(self, distortion):
    """
    Computes the rate-distortion value using the water-filling approach.

    Parameters:
      distortion (float): Given distortion constraint.

    Returns:
      float: Computed rate-distortion value in bits per pixel.
    """
    alpha = self.find_alpha(distortion) 
    # Pick eigenvalues greater than alpha
    mask = self.eigvals > alpha # T/F, (MN,), numpy array
    selected_eigvals = self.eigvals[mask] # wherever it is T, numpy array
    rate_value =  0.5*np.sum(np.log(selected_eigvals / alpha))
    rate_value = rate_value/(self.m * self.n)  # Convert to bits per pixel

    return rate_value