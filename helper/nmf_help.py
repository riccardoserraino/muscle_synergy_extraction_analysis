from config.importer import *
from helper.config_help import *


def find_elbow(values, min_synergies):
    """
    Detect optimal point in variance-explained curve.
    
    Implementation:
    - Uses vector geometry to find point farthest from line
    connecting first and last points
    - Ensures minimum synergy constraint
    
    Rationale:
    The "elbow" represents the point of diminishing returns
    where adding more synergies provides little improvement
    
    Returns:
    Index of optimal synergy count
    """
    if len(values) < 3:
        return min_synergies - 1  # return index for minimum synergies
    
    # Convert to numpy arrays for vector operations
    x = np.arange(len(values))
    y = np.array(values)
    
    # Create reference line
    line_vec = np.array([x[-1], y[-1]]) - np.array([x[0], y[0]])
    line_norm = line_vec / np.linalg.norm(line_vec)
    
    # Calculate distances
    vectors = np.column_stack((x - x[0], y - y[0]))
    distances = np.abs(np.cross(line_norm, vectors))
    
    return max(np.argmax(distances), min_synergies - 1)


#------------------------------------------------------------------------


def apply_nmf(emg_data, n_components, init, max_iter, l1_ratio, alpha_W, random_state):
    nmf = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state) # Setting Sparse NMF parameters
    W = nmf.fit_transform(emg_data)  # Synergy activations
    H = nmf.components_  # Muscle patterns
    # Transpose W and H to match the correct shapes if needed
    if W.shape[0] != emg_data.shape[0]:
        W = W.T  # Ensure W has shape (n_samples, n_synergies)
    if H.shape[0] != n_components:
        H = H.T  # Ensure H has shape (n_synergies, n_muscles)
    Z = np.dot(W, H)  # Reconstructed signal
    rec_error = nmf.reconstruction_err_ # Reconstruction error
    return W, H, Z, rec_error


#------------------------------------------------------------------------


def compute_vaf(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_W, random_state):
    VAF_values = []

    for n in range(1, max_synergies + 1):
        W, H, Z, rec_err = apply_nmf(emg_data, n, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state)
        VAF = 1 - np.sum((emg_data - Z) ** 2) / np.sum(emg_data ** 2)
        VAF_values.append(VAF)
        print(f"VAF for {n} synergies: {VAF:.4f}")

    return VAF_values


