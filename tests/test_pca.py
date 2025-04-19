from analyser.nmf import *
from analyser.dataload import *
from analyser.pca import *
from helper.visualization_help import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/int_unibo/all_scripts/muscle_synergy_analysis/config/config.yaml"

loader = emgDataLoader(config_dir)

# Load emg data for all 3 gestures combined
pinch_ulnar_power_dict = loader.combined_poses_dict()
pinch_ulnar_power_000, ts_000 = loader.combined_dataset_3('0', '0', '0')


extractor = emgPCA(pinch_ulnar_power_000, emg_data_dict=pinch_ulnar_power_dict)

optimal_synergies = 3


S_m, U, mean = extractor.PCA()

reconstructed = extractor.PCA_reconstruction(U, S_m, mean, optimal_synergies)

plot_all_results(pinch_ulnar_power_000, reconstructed, U, S_m, optimal_synergies)

