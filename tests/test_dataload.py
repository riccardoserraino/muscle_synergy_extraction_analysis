from data.dataload import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/th_unibo/muscle_synergy_analysis/config/config.yaml"

# Load YAML file
with open(config_dir, "r") as f:  # Fixed: Use self.config_path instead of global config_path
    config = yaml.safe_load(f)
    if not config:
        raise ValueError("YAML file is empty or not properly formatted.")
        




# Load emg data for a single gesture
pinch_data_dict = MuscleSynergyDataLoader(config_dir, "pinch").single_pose()
ulnar_data_dict = MuscleSynergyDataLoader(config_dir, "ulnar").single_pose()
power_data_dict = MuscleSynergyDataLoader(config_dir, "power").single_pose()

# Load emg data for all gestures
pinch_ulnar_power = MuscleSynergyDataLoader(config_dir).combined_poses()

