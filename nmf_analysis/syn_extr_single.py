# Import the utilities module containing all muscle synergy analysis functions
import utils

#-----------------------------------------------------------
# DATASET INITIALIZATION AND PARAMETERS CONFIGURATION
#-----------------------------------------------------------

# Define file paths for each grasp type's repetitions
# Using dictionaries with numeric keys for consistent access
pinch = {
    '0': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep0_pinch.bag',
    '1': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep1_pinch.bag',
    '2': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep2_pinch.bag',
}

ulnar = {
    '0': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep0_ulnar.bag',
    '1': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep1_ulnar.bag',
    '2': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep2_ulnar.bag',
}

power = {
    '0': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep0_power.bag',
    '1': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep1_power.bag',
    '2': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep2_power.bag',
}

pose_datasets = {
    "pinch": pinch,
    "ulnar": ulnar,
    "power": power
}


# ROS configuration
topic_name = 'emg_rms'  # The ROS topic where EMG RMS values are published

# Synergy extraction parameters
max_synergies = 8       # Physiological limit based on 8 muscle channels
min_synergies = 2       # Minimum meaningful number of synergies

# Sparsity optimization parameters
l1_ratio = 0.7          # Balance between L1 and L2 regularization (0.7 favors more sparsity)
alpha_values = [         # Range of sparsity parameters to test
    0.001, 
    0.005,               # Very mild sparsity
    0.01,                # Light sparsity  
    0.05,                # Moderate sparsity
    0.1                  # Strong sparsity
]

# Optimization weights (must sum to 1)
var_weight = 0.5        # Importance of reconstruction accuracy
sparsity_weight = 0.4   # Importance of sparse components  
cond_weight = 0.1       # Importance of numerical stability


#-----------------------------------------------------------
# DATA LOADING AND PREPARATION
#-----------------------------------------------------------

# Interactive pose selection and data loading
# This will prompt the user to select which pose type to analyze
pose_data = utils.choose_pose(pinch, ulnar, power, topic_name)

# Extract just the EMG data arrays, ignoring timestamps
# Creates dictionary with format: {'0': emg_data_rep0, '1': emg_data_rep1, ...}
reps_dict = {k: v[0] for k, v in pose_data.items() if v[0] is not None}


#-----------------------------------------------------------
# MUSCLE SYNERGY OPTIMIZATION PIPELINE
#-----------------------------------------------------------

# Run complete parameter optimization:
# 1. Determines optimal number of synergies via cross-validation
# 2. Finds best sparsity parameters for that synergy count
# 3. Returns all optimization results
optimal_synergies, alpha_result, cv_results, sparsity_results = utils.optimize_parameters(
    reps_dict,          # Dictionary of EMG data arrays
    max_synergies,      # Upper bound for synergy count
    alpha_values,       # Sparsity parameters to test
    l1_ratio,           # Regularization balance
    min_synergies,      # Lower bound for synergy count
    var_weight,         # Reconstruction accuracy weight
    sparsity_weight,    # Sparsity importance weight
    cond_weight         # Numerical stability weight
)

#'''
#-----------------------------------------------------------
# RECONSTRUCTION AND VISUALIZATION PIPELINE
#-----------------------------------------------------------

utils.analyze_with_optimized_parameters(
    pose_data,
    optimal_synergies,
    alpha_result['alpha'],
    l1_ratio
)
#'''



