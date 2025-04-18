from config.importer import *
from helper.config_help import *


def find_elbow(values, min_synergies):
    """
    Detect convergence point in curve.
    
    Args:
        values: List of error values
        min_synergies: Minimum number of synergies to consider
    
    Outputs:
        Elbow point index
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
    """
    Applies Non-negative Matrix Factorization (NMF) to EMG data.

    Args:
        emg_data: Input EMG data (n_samples x n_muscles).
        n_components: Number of synergies to extract.
        init: Initialization method for NMF. 
        max_iter: Maximum number of iterations for NMF.
        l1_ratio: L1 ratio for sparse NMF.
        alpha_W: Regularization parameter for U matrix in NMF.
        random_state: Random seed for reproducibility.

    Outputs:
        U: Synergy activations over time (Neural drive matrix)
        S_m: Muscle patterns (Muscular synergy matrix)

    """
    nmf = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state) # Setting Sparse NMF parameters
    U = nmf.fit_transform(emg_data)         # Synergy activations over time (Neural drive matrix)
    S_m = nmf.components_                   # Muscle patterns (Muscular synergy matrix)
    
    # Transpose W and H to match the correct shapes if needed
    if U.shape[0] != emg_data.shape[0]:
        U = U.T         # Ensure U has shape (n_samples, n_synergies)
    if S_m.shape[0] != n_components:
        S_m = S_m.T     # Ensure S_m has shape (n_synergies, n_muscles)
    return U, S_m



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



def cross_validate_synergies(reps_dict, max_synergies, alpha, l1_ratio, min_synergies):
    """
    Determine optimal synergy count via cross-validation.
    
    Args: 
    
    Returns:
    (optimal_synergies, variance_results)
    """
    # Extract the repetitions from the dictionary
    rep0 = reps_dict['0']
    rep1 = reps_dict['1'] 
    rep2 = reps_dict['2']
    
    synergy_range = range(1, max_synergies + 1)
    results = []
    
    print("\nRunning cross-validation for synergy selection...")
    for n in synergy_range:
        fold_metrics  = []
        # Create all possible train/test splits
        for (X_train, X_test) in [
            (np.vstack([rep0, rep1]), rep2),
            (np.vstack([rep0, rep2]), rep1),
            (np.vstack([rep1, rep2]), rep0)
        ]:
            try:
                model = NMF(n_components=n, 
                            alpha_W=alpha, 
                            alpha_H=0, 
                            l1_ratio=l1_ratio, 
                            init='nndsvd', 
                            max_iter=500, 
                            random_state=42)
                W_train = model.fit_transform(X_train)
                
                # Calculate metrics
                X_recon = model.transform(X_test) @ model.components_
                recon_error = np.linalg.norm(X_test - X_recon, 'fro')
                var_explained = 1 - (recon_error**2 / np.sum(X_test**2))

                fold_metrics.append(var_explained)

            except Exception as e:
                print(f"Warning: {n} synergies failed - {str(e)}")
                fold_metrics.append(0)  # Append 0 variance if failed
                continue
        
        if fold_metrics:
            avg_variance = np.mean(fold_metrics)
            results.append((avg_variance))
            print(f"  - Synergies: {n}, Variance Explained: {avg_variance:.2%}")

    # Find elbow point in variance explained curve
    if len(results) > 0:
        optimal_idx = find_elbow(results, min_synergies)
        optimal_synergies = synergy_range[optimal_idx]
    else:
        optimal_synergies = min_synergies
    
    print(f"\nOptimal number of synergies: {optimal_synergies}")

    return optimal_synergies, results



#-----------------------------------------------------------



def sparsity_evaluation(reps_dict, n_synergies, alpha_values, l1_ratio):
    """
    Evaluate sparsity parameters for given synergy count.
    
    Implementation:
    - Tests range of alpha values
    - Tracks multiple quality metrics
    - Uses conservative sparsity on components
    - Detailed progress reporting
    
    Returns:
    List of dictionaries with evaluation metrics
    """

    X = np.vstack([reps_dict['0'], reps_dict['1'], reps_dict['2']])
    results = []

    print("\nEvaluating sparsity parameters...")
    for alpha in alpha_values:
        try:
            model = NMF(n_components=n_synergies, 
                        alpha_W=alpha, 
                        alpha_H=0, 
                        l1_ratio=l1_ratio,
                        init='nndsvd', 
                        max_iter=500, 
                        random_state=42)
            W = model.fit_transform(X)
            U = model.components_
            
            # Calculate multiple metrics
            recon_error = np.linalg.norm(X - W @ U, 'fro')
            var_explained = 1 - (recon_error**2 / np.sum(X**2))
            sparsity = (np.count_nonzero(U == 0) / U.size) * 100
            condition_number = np.linalg.cond(U)
            
            results.append({
                'alpha': alpha,
                'error': recon_error,
                'variance': var_explained,
                'sparsity': sparsity,
                'condition': condition_number
            })

            print(f"  - α={alpha:.3f}: Var={var_explained:.2%}, " 
                  f"Sparsity={sparsity:.1f}%, Cond={condition_number:.1f}")
        
        except Exception as e:
            print(f"  - α={alpha:.3f} failed: {str(e)}")
            continue

    return results



#-----------------------------------------------------------



def best_alpha(sparsity_results, var_weight, sparsity_weight, cond_weight):
    """
    Select best alpha based on multiple criteria.
    """
    if not sparsity_results:
        raise ValueError("No valid sparsity results")
    
    # Normalize metrics
    vars = np.array([r['variance'] for r in sparsity_results])
    spars = np.array([r['sparsity'] for r in sparsity_results])
    conds = np.array([r['condition'] for r in sparsity_results])
    
    # Higher variance and sparsity are better, lower condition number is better
    norm_vars = (vars - np.min(vars)) / (np.max(vars) - np.min(vars))
    norm_spars = (spars - np.min(spars)) / (np.max(spars) - np.min(spars))
    norm_conds = 1 - ((conds - np.min(conds)) / (np.max(conds) - np.min(conds)))
    
    # Combined score
    scores = (var_weight * norm_vars + 
              sparsity_weight * norm_spars + 
              cond_weight * norm_conds)
    
    best_idx = np.argmax(scores)
    return sparsity_results[best_idx]



#------------------------------------------------------------------------



def explain_frobenius_error(original, reconstructed):
    """
    Generates human-readable interpretation of reconstruction error metrics
    Returns dictionary with:
    - total_error: Original Frobenius norm
    - normalized_error: Error relative to signal magnitude
    - per_sample_error: Average error per sample per muscle
    - per_muscle_error: Average error per muscle channel
    """
    error = np.linalg.norm(original - reconstructed)
    n_samples, n_muscles = original.shape
    
    return {
        'total_frobenius_error': error,
        'normalized_error': error / np.linalg.norm(original),
        'per_sample_error': error / (n_samples * n_muscles),
        'per_muscle_error': [np.mean(np.abs(original[:,i] - reconstructed[:,i])) for i in range(n_muscles)]
    }



