from analyser.nmf import *
from analyser.dataload import *
from helper.visualization_help import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/int_unibo/all_scripts/muscle_synergy_analysis/config/config.yaml"

loader = emgDataLoader(config_dir)

# Load emg data for all 3 gestures combined
pinch_ulnar_power_dict = loader.combined_poses_dict()
pinch_ulnar_power_000, ts_000 = loader.combined_dataset_3('0', '0', '0')

extractor = emgNMF(pinch_ulnar_power_000, emg_data_dict=pinch_ulnar_power_dict)

# Evaluate Synergy number and Sparsity
# optimal_synergies, optimal_alpha, optimal_l1, cv_results, sparsity_results = extractor.synergy_sparsity_extractor()
optimal_synergies = 3

# Sparse NMF application
s_U, s_S_m = extractor.SparseNMF()
s_reconstructed_data = extractor.NMF_reconstruction(optimal_synergies, s_U, s_S_m)
# Plotting sparse results
plot_all_results(pinch_ulnar_power_000, s_reconstructed_data, s_U, s_S_m, optimal_synergies)


# Classical NMF
U, S_m = extractor.ClassicNMF()
reconstructed_data = extractor.NMF_reconstruction(optimal_synergies, U, S_m)
# Plotting classical results
plot_all_results(pinch_ulnar_power_000, reconstructed_data, U, S_m, optimal_synergies)
