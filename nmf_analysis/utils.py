"""
MUSCLE SYNERGY ANALYSIS TOOLKIT

This module provides tools for extracting muscle synergies from EMG data using Sparse Non-negative Matrix Factorization (NMF).
The implementation follows these key steps:
1. Data loading and preprocessing
2. Cross-validation to determine optimal synergy count
3. Sparsity optimization for cleaner muscle patterns
4. Parameter selection based on multiple criteria

The pipeline is designed to:
- Handle real-world EMG data from ROS bag files
- Automatically determine the simplest muscle coordination patterns
- Provide interpretable results through sparsity control
- Validate results through cross-validation
"""



import rosbag
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

#-----------------------------------------------------------
# Loads EMG data from a '.bag' file

def load_emg_data(bag_path, topic_name):
    """
    Load and preprocess EMG data from ROS bag files.
    
    Implementation:
    - Reads timestamped EMG data from specified ROS topic
    - Ensures proper array orientation (samples × muscles)
    - Handles potential transposition needs
    
    Usage:
    emg_data, timestamps = load_emg_data('path/to/data.bag', 'emg_topic')
    """
    timestamps = []
    emg_data = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, rawdata, timestamp in bag.read_messages(topics=[topic_name]):
            emg_data.append(rawdata.data)  # Extract EMG RMS values
            timestamps.append(timestamp.to_sec())  # Convert timestamp to seconds
    
    emg_data = np.array(emg_data)
    timestamps = np.array(timestamps)   

    # Ensure emg_data is shaped correctly (samples × muscles)
    if emg_data.shape[0] < emg_data.shape[1]:  # If (n_muscles, n_samples), transpose it
        emg_data = emg_data.T  # Now it’s (n_samples, n_muscles)
    return emg_data, timestamps


#-----------------------------------------------------------
# Choose the single pose to test the model 

def choose_pose(pinch, ulnar, power, topic_name):
    """
    Interactive pose selection and data loading.
    
    Implementation:
    - Presents menu of available poses
    - Validates user input
    - Loads all repetitions for selected pose
    - Provides loading feedback
    
    Usage:
    loaded_data = choose_pose(pinch_dict, ulnar_dict, power_dict, 'emg_topic')
    """
    pose_dicts = {
        '1': ('PINCH', pinch),
        '2': ('ULNAR', ulnar),
        '3': ('POWER', power)
    }
    while True:
        print("\nAvailable pose datasets:")
        print("1. Pinch")
        print("2. Ulnar")
        print("3. Power")
        
        choice = input("Select a pose dataset (1-3): ").strip()
        
        if choice in pose_dicts:
            pose_name, selected_pose = pose_dicts[choice]
            print(f'Pose chosen: {pose_name}.')
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    print("\nLoading pose data...")
    loaded_data = {}
    for rep_id, bag_path in selected_pose.items():
        try:
            emg_data, timestamps = load_emg_data(bag_path, topic_name)
            loaded_data[rep_id] = (emg_data, timestamps)
            print(f"  - Repository {rep_id} loaded successfully")
            print(f"    EMG shape: {emg_data.shape}, Timestamps: {len(timestamps)}")
        except Exception as e:
            print(f"Error loading {bag_path}: {str(e)}")
            loaded_data[rep_id] = (None, None)
    
    return loaded_data


#-----------------------------------------------------------
# Find elbow point in variance explained curve

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

#-----------------------------------------------------------
# Cross Validation analysis for optimal synergies number extraction 

def best_nsyn(reps_dict, max_synergies, alpha, l1_ratio, min_synergies):
    """
    Determine optimal synergy count via cross-validation.
    
    Implementation:
    - 3-fold cross-validation (leave-one-repetition-out)
    - Uses variance explained as primary metric
    - Applies light sparsity only to activation patterns
    - Robust error handling
    
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
# Sparsity parameters evaluation

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
# Best sparsity parameter (alpha) selection 
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


#-----------------------------------------------------------
# Optimization of parameters function (combines cv and sparsity analysis)
def optimize_parameters(reps_dict, max_synergies, alpha_values, l1_ratio, min_synergies, var_weight, sparsity_weight, cond_weight):
    """
    Selects the optimal alpha value based on multiple criteria.
    
    Implementation:
    1. Normalizes three key metrics to [0,1] range:
       - Variance explained (higher is better)
       - Sparsity percentage (higher is better)
       - Condition number (lower is better, so we invert)
    2. Computes weighted combined score
    3. Returns parameters with highest score
    
    Args:
        sparsity_results: List of dicts from sparsity_evaluation()
        var_weight: Importance of variance explained
        sparsity_weight: Importance of sparsity
        cond_weight: Importance of conditioning
    
    Returns:
        Dictionary with best parameters and metrics
    """

    print("\n=== Starting parameter optimization ===")

    # Initial CV with relaxed sparsity
    optimal_synergies, cv_results = best_nsyn(
        reps_dict,
        max_synergies,
        alpha=0.01,  # Very light sparsity
        l1_ratio=l1_ratio,
        min_synergies=min_synergies
    )
    
    # Sparsity optimization
    sparsity_results = sparsity_evaluation(
        reps_dict,
        optimal_synergies,
        alpha_values,
        l1_ratio
    )
    
    # Select best alpha
    if not sparsity_results:
        alpha_result  = {'alpha': 0.01, 'variance': 0, 'sparsity': 0}
        print("Warning: No valid sparsity results, using default alpha = 0.01")
    else:
        alpha_result  = best_alpha(sparsity_results, var_weight, sparsity_weight, cond_weight)
    
    print("\n=== Final Parameters ===")
    print(f"Optimal synergies: {optimal_synergies}")
    print(f"Best alpha: {alpha_result ['alpha']:.3f}")
    print(f"Variance explained: {alpha_result ['variance']:.2%}")
    print(f"Sparsity achieved: {alpha_result ['sparsity']:.1f}%")
    
    return optimal_synergies, alpha_result , cv_results, sparsity_results


#-----------------------------------------------------------
# Signal reconstruction using selected synergies
def reconstruct_signal(W, H, selected_synergies):
    """
    Reconstructs EMG signal using a subset of muscle synergies.
    
    Implementation:
    - Takes activation patterns (W) and synergy components (H)
    - Selects only the first 'selected_synergies' components
    - Performs matrix multiplication to reconstruct the signal
    - Maintains original data dimensions while reducing synergy count

    Args:
        W: Activation matrix (n_samples x n_synergies)
        H: Synergy components matrix (n_synergies x n_muscles)
        selected_synergies: Number of synergies to include (must be ≤ n_synergies)
    """
    # Select subset of synergies
    W_selected = W[:, :selected_synergies]
    H_selected = H[:selected_synergies, :]
    
    # Reconstruct signal via matrix multiplication
    Z_reconstructed = np.dot(W_selected, H_selected)
    return Z_reconstructed


#-----------------------------------------------------------
# Plotting function for results visualization
def plot_results(emg_data, Z_reconstructed, W_scaled, H, selected_synergies):
    """
    Visualizes EMG analysis results in a 4-panel comparative plot.
    
    Implementation:
    - Creates figure with 4 vertically stacked subplots
    - Uses consistent scaling for comparison
    - Automatically handles variable synergy counts
    - Preserves non-negativity of muscle activations
    
    Args:
        emg_data: Raw EMG (n_samples x n_muscles)
        Z_reconstructed: Reconstructed signal (n_samples x n_muscles)
        W_scaled: Normalized activations (n_samples x n_synergies)
        H: Synergy components (n_synergies x n_muscles)
        selected_synergies: Number of synergies 
        
    Produces:
        Interactive matplotlib figure with:
        1. Original EMG signals
        2. Reconstructed signals
        3. Synergy activations over time
        4. Muscle weightings per synergy
    """
    plt.figure(figsize=(10, 8))
    
    # Panel 1: Original EMG Signals
    plt.subplot(4, 1, 1)
    plt.plot(emg_data)
    plt.title('Original EMG Signals')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])  # Remove x-axis labels for cleaner visualization
    
    # Panel 2: Reconstructed EMG Signals
    plt.subplot(4, 1, 2)
    plt.plot(Z_reconstructed, linestyle='--')
    plt.title(f'Reconstructed EMG ({selected_synergies} Synergies)')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])
    
    # Panel 3: Synergy Activation Patterns
    plt.subplot(4, 1, 3)
    for i in range(selected_synergies):
        plt.plot(W_scaled[:, i], label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time')
    plt.ylabel('Activation')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])
    
    # Panel 4: Muscle Weighting Patterns
    plt.subplot(4, 1, 4)
    for i in range(selected_synergies):
        plt.plot(H[i, :], 'o-', label=f'Synergy {i+1}')
    plt.title('Muscle Weighting Patterns')
    plt.xlabel('Muscle Channel')
    plt.ylabel('Weight')
    plt.legend(loc='upper right', ncol=selected_synergies)

    plt.tight_layout()
    plt.show()


#-----------------------------------------------------------
# Applies Non-Negative Matrix Factorization (NMF) to extract synergies
def apply_nmf(emg_data, n_components, init, max_iter, l1_ratio, alpha_W, random_state):
    """
    Performs Sparse Non-Negative Matrix Factorization (NMF) on EMG data to extract muscle synergies.
    
    Implementation:
    - Initializes NMF with specified sparsity constraints
    - Decomposes EMG into activation patterns (W) and synergy components (H)
    - Automatically handles matrix orientation
    - Calculates reconstruction quality metrics
    
    Args:
        emg_data: Input EMG matrix (n_samples x n_muscles)
        n_components: Number of synergies to extract
        init: Initialization method ('nndsvd', 'nndsvda', etc.)
        max_iter: Maximum optimization iterations
        l1_ratio: Balance between L1/L2 regularization (0-1)
        alpha_W: Sparsity control for activation patterns
        random_state: Seed for reproducible results
        
    Returns:
        W: Activation patterns (n_samples x n_synergies)
        H: Synergy components (n_synergies x n_muscles)
        Z: Reconstructed EMG (n_samples x n_muscles)
        rec_error: Frobenius norm reconstruction error
    """
    nmf = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state) # Setting Sparse NMF parameters
    W = nmf.fit_transform(emg_data)  # Synergy activations
    H = nmf.components_  # Muscle patterns
    # Transpose W and H to match the correct shapes if needed
    if W.shape[0] != emg_data.shape[0]:
        W = W.T  # Ensure W has shape (n_samples, n_synergies)
    if H.shape[0] != n_components:
        H = H.T  # Ensure H has shape (n_synergies, n_muscles)
    Z = np.dot(W, H)  # Reconstructed signal
    rec_error = nmf.reconstruction_err_# Reconstruction error
    return W, H, Z, rec_error


#-----------------------------------------------------------
# Scales the synergy signal S_m to match the range of the original EMG signal for a better visual
def scale_synergy_signal(W, emg_data):
    """
    Normalizes synergy activations to match original EMG amplitude range.
    
    Implementation:
    - Linear scaling preserving activation dynamics
    - Maintains non-negativity constraint
    - Handles both single and multi-channel EMG
    
    Args:
        W: Activation matrix (n_samples x n_synergies)
        emg_data: Original EMG (n_samples x n_muscles)
    """
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    W_min = np.min(W)
    W_max = np.max(W)
    W_scaled = ((W - W_min) / (W_max - W_min)) * (emg_max - emg_min) + emg_min
    W_scaled = np.maximum(W_scaled, 0)  # Ensures W_scaled is non-negative
    return W_scaled


#-----------------------------------------------------------
# Frobenius error calculation and explanation
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


#-----------------------------------------------------------
# Analyze EMG data with optimized NMF parameters
def analyze_with_optimized_parameters(pose_data, optimal_synergies, alpha, l1_ratio):
    """
    Analyze EMG data with optimized NMF parameters, allowing user to select which repository to visualize.
    
    Implementation:
    1. Takes optimized parameters from previous analysis
    2. Presents menu of available repositories
    3. For selected repository:
       - Performs NMF decomposition
       - Reconstructs signal using specified synergy count
       - Scales activations for visualization
       - Displays comprehensive results
       - Generates comparison plots
    
    Args:
        pose_data: Dictionary containing loaded pose data {rep_id: (emg_data, timestamps)}
        optimal_synergies: Optimal number of synergies determined from optimization
        alpha: Optimal sparsity parameter (alpha_W)
        l1_ratio: Selected L1 ratio for regularization
    """
    # Get available repetitions (filter out any None/invalid data)
    available_reps = [k for k in pose_data.keys() if pose_data[k][0] is not None]
    if not available_reps:
        print("\n=== No valid repetitions available for analysis ===")
        return
    
    max_rep = max(available_reps)
    
    print("\n=== Now proceeding with reconstruction of repository selected ===")
    print("=== Based on optimized parameters: ===")
    print(f"- Optimal synergies: {optimal_synergies}")
    print(f"- Alpha (sparsity): {alpha:.3f}")
    print(f"- L1 ratio: {l1_ratio}")
    
    while True:
        # Display menu of available repetitions
        print("\n" + "="*50)
        print(f"Available repository (0-{max_rep}):")
        for rep in sorted(available_reps):
            print(f"{rep}. Analyze repository {rep}")
        print(f"{len(available_reps)}. Exit analysis")
        
        try:
            choice = input(f"\nSelect repository to analyze (0-{max_rep}) or {len(available_reps)} to exit: ").strip()
            
            if choice == str(len(available_reps)):
                print("\n=== Session ended ===")
                break
            elif choice in available_reps:
                print(f"\n=== Analyzing repository {choice} ===")
                emg_data, timestamps = pose_data[choice]
                
                # 1. Perform NMF decomposition with optimized parameters
                print("- Performing NMF decomposition...")
                W, H, _, rec_error = apply_nmf(
                    emg_data=emg_data,
                    n_components=optimal_synergies,
                    init='nndsvd',
                    max_iter=2000,
                    l1_ratio=l1_ratio,
                    alpha_W=alpha,
                    random_state=42
                )
                
                # 2. Reconstruct signal using specified function
                print("- Reconstructing EMG signal...")
                Z_reconstructed = reconstruct_signal(W, H, optimal_synergies)
                
                # 3. Scale activations for visualization
                print("- Scaling activation patterns...")
                W_scaled = scale_synergy_signal(W, emg_data)
                
                # 4. Display results
                print("\n=== Analysis Results ===")
                print("Matrix Shapes:")
                print(f"- Original EMG: {emg_data.shape} (samples x muscles)")
                print(f"- Reconstructed EMG: {Z_reconstructed.shape} (samples x muscles)")
                print(f"- Activation matrix (W): {W.shape} (samples x synergies)")
                print(f"- Synergy components (H): {H.shape} (synergies x muscles)")
                
                print("\n=== Reconstruction Quality Metrics ===")      
                error_metrics = explain_frobenius_error(emg_data, Z_reconstructed)
    
                print(f"1. Total Frobenius Norm: {error_metrics['total_frobenius_error']:.2f}")
                print(f"2. Normalized Error: {error_metrics['normalized_error']:.2%}")
                print(f"3. Mean Error Per Sample: {error_metrics['per_sample_error']:.4f} mV")
                print("4. Per-Muscle Mean Absolute Errors:")
                for i, err in enumerate(error_metrics['per_muscle_error']):
                    print(f"   - Muscle {i+1}: {err:.4f} mV")
                
                vaf = 1 - (error_metrics['total_frobenius_error']**2 / np.sum(emg_data**2))
                print(f"\n5. Variance Accounted For (VAF): {vaf:.2%}")
                
                print(f"- Reconstruction error: {rec_error:.2f}")
                variance = 1 - (rec_error**2 / np.sum(emg_data**2))
                print(f"- Variance explained: {variance:.2%}")
                
                # 5. Generate plots
                print("\nGenerating visualization...")
                plot_results(
                    emg_data=emg_data,
                    Z_reconstructed=Z_reconstructed,
                    W_scaled=W_scaled,
                    H=H,
                    selected_synergies=optimal_synergies
                )
                
                print(f"\n=== Completed analysis of Repository {choice} ===")
            else:
                print("\n! Invalid selection. Please enter a valid number.")
        except ValueError:
            print(f"\n! Please enter a valid number (0-{len(available_reps)})")


#-----------------------------------------------------------
# Loads EMG data from combined ROS .bag file
def load_combined_emg_data(selected_paths, topic_name):
    # Initialize empty lists for EMG data and timestamps
    emg_data_combined = []
    timestamps_combined = []
    

    for bag_path in selected_paths:
        # Load EMG data and timestamps
        emg_data, timestamps = load_emg_data(bag_path, topic_name)
        
        # Check and reshape data if necessary
        if emg_data.shape[0] < emg_data.shape[1]:
            emg_data = emg_data.T
        
        # Append data to the lists
        emg_data_combined.append(emg_data)
        timestamps_combined.append(timestamps)

    # Concatenate data into single arrays
    emg_data_combined = np.vstack(emg_data_combined)
    timestamps_combined = np.concatenate(timestamps_combined)

    return emg_data_combined, timestamps_combined


#-----------------------------------------------------------
# Interactive pose selection for combination
def choose_poses_interactively():
    """
    Interactive menu for pose selection
    Returns list of selected pose names
    """
    pose_options = {
        '1': 'pinch',
        '2': 'ulnar',
        '3': 'power'
    }
    
    selected = []
    while True:
        print("\nSelect poses to combine:")
        print("1. Pinch")
        print("2. Ulnar")
        print("3. Power")
        print("4. Done selecting")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '4':  # Exit condition
            if not selected:
                print("Please select at least one pose.")
                continue
            break
        elif choice in pose_options:
            pose = pose_options[choice]
            selected.append(pose)  # Allow repeated selections
            print(f"Added {pose} (total {selected.count(pose)} times)")
        else:
            print("Invalid choice. Please enter a number from 1 to 4.")
    
    return selected


#-----------------------------------------------------------
# Create combined datasets in the order pinch ulnar power
def load_all_poses(pinch, ulnar, power, topic_name):
    """
    Loads and combines all three poses automatically
    Returns dictionary with combined data for each repository
    """
    print("\nLoading pose data...")
    combined_data = {}
    
    # Get all available repository numbers (assuming same reps for all poses)
    rep_numbers = pinch.keys()  # ['0', '1', '2']
    
    for rep in rep_numbers:
        selected_paths = []
        
        # Always include all three poses
        selected_paths.append(pinch[rep])
        selected_paths.append(ulnar[rep])
        selected_paths.append(power[rep])
        
        # Load and combine data
        emg_combined, ts_combined = load_combined_emg_data(selected_paths, topic_name)
        combined_data[rep] = (emg_combined, ts_combined)
    
    return combined_data


#-----------------------------------------------------------

def load_selected_poses(pose_datasets, selected_poses, topic_name):
    """
    Loads and combines EMG data for selected poses dynamically.
    Returns dictionary with combined data per repetition.
    """
    print("\nLoading pose data...")

    # Get available repetitions (assuming all poses have the same rep structure)
    rep_numbers = list(pose_datasets["pinch"].keys())  

    # Initialize combined data dictionary
    combined_data = {}

    for rep in rep_numbers:
        selected_paths = []

        # Add selected poses' dataset paths
        for pose in selected_poses:
            dataset = pose_datasets.get(pose)
            if dataset:
                selected_paths.append(dataset[rep])

        # Load and combine EMG data
        if selected_paths:
            emg_combined, ts_combined = load_combined_emg_data(selected_paths, topic_name)
            combined_data[rep] = (emg_combined, ts_combined)

    return combined_data

