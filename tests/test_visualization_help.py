from analyser.dataload import *
from helper.visualization_help import *


# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/int_unibo/all_scripts/muscle_synergy_analysis/config/config.yaml"


# Load YAML file
config = emgDataLoader(config_dir)._load_config()      


# Load emg data for a single gesture set
pinch_data_dict = emgDataLoader(config_dir, "pinch").single_pose_dict()

# Load emg data for all gestures
pinch_ulnar_power = emgDataLoader(config_dir).combined_poses_dict()

# Load emg data for a single gesture
pinch1_emg, pinch1_ts = emgDataLoader(config_dir, "pinch").single_dataset("1")

# Load emg data for combined gestures
pin_uln_pow_000_emg, pin_uln_pow_000_ts = emgDataLoader(config_dir).combined_dataset_3("0", "0", "0")



plot_signal_dict(pinch_data_dict, pose_name="pinch")
plot_signal_dict(pinch_ulnar_power, pose_name="pinch + ulnar + power")

plot_signal(pinch1_emg, 'emg data for pinch rep 1')
plot_signal(pin_uln_pow_000_emg, 'emg data for pinch ulnar power reps000')
