from analyser.dataload import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/th_unibo/muscle_synergy_analysis/config/config.yaml"

# Load YAML file
config = emgDataLoader(config_dir)._load_config()      

# Load emg data for a single gesture set
pinch_dict = emgDataLoader(config_dir, "pinch").single_pose_dict()

# Load emg data for all gestures
pinch_ulnar_power = emgDataLoader(config_dir).combined_poses_dict()
    
# Load emg data for a single gesture
pinch1_emg, pinch1_ts = emgDataLoader(config_dir, "pinch").single_dataset("1")

# Load emg data for combined gestures
pinch_emg, pinch_ts = emgDataLoader(config_dir, "pinch").combined_dataset("0", "0", "0")
