# Code able to analyze multiple synergy combination

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
    0.001,               # Very mild sparsity
    0.005,
    0.01,                # Light sparsity  
    0.05,                # Moderate sparsity
    #0.1                  # Strong sparsity
]

# Optimization weights (must sum to 1)
var_weight = 0.6        # Importance of reconstruction accuracy
sparsity_weight = 0.3   # Importance of sparse components  
cond_weight = 0.1       # Importance of numerical stability


#-----------------------------------------------------------
# DATA LOADING AND PREPARATION
#-----------------------------------------------------------

# Ask user which poses they want to analyze (allowing repeats)
selected_poses = utils.choose_poses_interactively()

# Create combined datasets for each repetition based on user selection
combined_data = utils.load_selected_poses(pose_datasets, selected_poses, topic_name)

# Extract just EMG data for analysis
emg_data_dict = {k: v[0] for k, v in combined_data.items() if v[0] is not None}

#-----------------------------------------------------------
# MUSCLE SYNERGY OPTIMIZATION PIPELINE
#-----------------------------------------------------------

if emg_data_dict:
    print(f"\nAnalyzing combined poses)")
    
    # Run optimization
    optimal_synergies, alpha_result, cv_results, sparsity_results = utils.optimize_parameters(
        emg_data_dict,
        max_synergies,
        alpha_values,
        l1_ratio,
        min_synergies,
        var_weight,
        sparsity_weight,
        cond_weight
    )
    #'''
    # Visualization
    utils.analyze_with_optimized_parameters(
        combined_data,
        optimal_synergies,
        alpha_result['alpha'],
        l1_ratio
    )
    #'''
else:
    print("No valid data to analyze. Please check your pose selections and data paths.")