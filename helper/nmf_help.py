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
    
    return W, H


#------------------------------------------------------------------------


def compute_vaf(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_W, random_state):
    VAF_values = []

    for n in range(1, max_synergies + 1):
        W, H, Z, rec_err = apply_nmf(emg_data, n, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state)
        VAF = 1 - np.sum((emg_data - Z) ** 2) / np.sum(emg_data ** 2)
        VAF_values.append(VAF)
        print(f"VAF for {n} synergies: {VAF:.4f}")

    return VAF_values


#------------------------------------------------------------------------



def cross_validate_synergies(datasets, max_synergies=8, n_folds=3, min_synergies=1, 
                           init='nndsvd', max_iter=500, l1_ratio=0.7, alpha_W=0.001, 
                           random_state=None):
    """
    Perform cross-validation to determine the optimal number of synergies across multiple datasets.
    
    Parameters:
    - datasets: List of 3 EMG datasets (numpy arrays) to analyze
    - max_synergies: Maximum number of synergies to test
    - n_folds: Number of folds for cross-validation
    - min_synergies: Minimum number of synergies to consider
    - Other parameters: NMF algorithm parameters for sparse NMF
    
    Returns:
    - optimal_synergies: List of optimal synergy counts for each dataset
    - all_vaf_curves: VAF curves for each dataset and fold
    """
    
    # Validate inputs
    if len(datasets) != 3:
        raise ValueError("Exactly 3 datasets are required")
    if any(d.shape[1] < max_synergies for d in datasets):
        raise ValueError("Number of muscles must be >= max_synergies")
    
    optimal_synergies = []
    all_vaf_curves = {i: [] for i in range(3)}  # Store VAF curves for each dataset
    
    for i, emg_data in enumerate(datasets):
        print(f"\nProcessing dataset {i+1}")
        
        # Prepare for cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_vaf_curves = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(emg_data)):
            print(f"\nFold {fold+1}/{n_folds}")
            train_data = emg_data[train_idx]
            test_data = emg_data[test_idx]
            
            # Compute VAF curve for this fold
            vaf_curve = []
            
            for n in range(1, max_synergies + 1):
                # Train on training set
                W_train, H_train = apply_nmf(
                    train_data, n, init=init, max_iter=max_iter, 
                    l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state
                )
                
                # Reconstruct test data
                W_test = np.linalg.lstsq(H_train.T, test_data.T, rcond=None)[0].T
                reconstructed = W_test @ H_train
                
                # Compute VAF on test set
                vaf = 1 - np.sum((test_data - reconstructed) ** 2) / np.sum(test_data ** 2)
                vaf_curve.append(vaf)
                print(f"Dataset {i+1}, Fold {fold+1}: VAF for {n} synergies: {vaf:.4f}")
            
            fold_vaf_curves.append(vaf_curve)
        
        # Average VAF curves across folds
        avg_vaf_curve = np.mean(fold_vaf_curves, axis=0)
        all_vaf_curves[i] = avg_vaf_curve
        
        # Find optimal number of synergies
        optimal_idx = find_elbow(avg_vaf_curve, min_synergies)
        optimal_synergies.append(optimal_idx + 1)  # Convert from index to count
        
        print(f"\nDataset {i+1} optimal synergies: {optimal_synergies[-1]}")
    
    return optimal_synergies, all_vaf_curves