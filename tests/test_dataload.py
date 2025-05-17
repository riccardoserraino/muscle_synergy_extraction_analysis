from analyser.dataload import *

# Initialization for yaml file directory
config_dir = "config/config.yaml"

# Load YAML file
config = emgDataLoader(config_dir)._load_config()      

# Load emg data for a single gesture set
pinch_dict = emgDataLoader(config_dir, "pinch").single_pose_dict()

# Load emg data for all gestures
pinch_ulnar_power = emgDataLoader(config_dir).combined_poses_dict()

# Load emg data for a single gesture
pinch1_emg, pinch1_ts = emgDataLoader(config_dir, "pinch").single_dataset("1")

# Load emg data for combined gestures
pin_uln_pow_000_emg, pin_uln_pow_000_ts = emgDataLoader(config_dir).combined_dataset_3("0", "0", "0")

# Load emg data for combined gestures with different repetitions and pose numbers
comb2_emg, comb2_ts= emgDataLoader(config_dir).combined_dataset(combination_name='combination_2', reps='1111')

