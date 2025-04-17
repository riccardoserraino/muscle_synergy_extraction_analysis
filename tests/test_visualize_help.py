from analyser.dataload import *
from helper.visualize_help import *


# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/int_unibo/all_scripts/muscle_synergy_analysis/config/config.yaml"

# Load YAML file
with open(config_dir, "r") as f:  # Fixed: Use self.config_path instead of global config_path
    config = yaml.safe_load(f)
    if not config:
        raise ValueError("YAML file is empty or not properly formatted.")
        

# Load emg data for a single gesture set
pinch_data_dict = DataLoader(config_dir, "pinch").single_pose_dict()

# Load emg data for all gestures
pinch_ulnar_power = DataLoader(config_dir).combined_poses_dict()


# Plot single pose (all repetitions)
plot_signal(pinch_data_dict, pose_name="Pinch")

# Plot combined data
plot_signal(pinch_ulnar_power, pose_name="Combined Poses")

# Plot single repetition
plot_signal({"0": pinch_data_dict["0"]})  # Just first rep


