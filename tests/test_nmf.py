from analyser.nmf import *
from analyser.dataload import *
from helper.visualize_help import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/th_unibo/muscle_synergy_analysis/config/config.yaml"

loader = emgDataLoader(config_dir)

# Load emg data for all gestures
pinch_ulnar_power_dict = loader.combined_poses_dict()
pinch_ulnar_power_000, ts_000 = loader.combined_dataset('0', '0', '0')

extractor = emgNMF(pinch_ulnar_power_000, emg_data_dict=pinch_ulnar_power_dict)

# Evaluate Synergy number and Sparsity
optimal_synergies, optimal_alpha, cv_results, sparsity_results = extractor.synergy_sparsity_extractor()
W, H = extractor.SparseNMF(S_alpha_W=optimal_alpha['alpha'])
reconstructed_data = extractor.NMF_reconstruction(optimal_synergies, W, H)

# Plotting the results
plot_vaf(VAF_values=cv_results)
plot_all_results(pinch_ulnar_power_000, reconstructed_data, W, H, optimal_synergies)

