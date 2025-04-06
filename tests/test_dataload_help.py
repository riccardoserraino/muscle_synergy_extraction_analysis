from config.importer import *
from helper.config_help import *
from helper.dataload_help import *


# Load YAML file
with open("C:/Users/ricca/Desktop/th_unibo/muscle_synergy_analysis/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    if not config:
        raise ValueError("YAML file is empty or not properly formatted.")


# Get file paths for a single pose rep (pinch)
pinch0 = get_pose_paths(config, "pinch")["0"]      
pinch0_emg, pinch0_ts = load_emg_data(pinch0, config["emg_topic"])
if pinch0_emg.all() != None and pinch0_ts.all() != None:
    print(f"Loaded {pinch0_emg.shape[0]} samples and {pinch0_emg.shape[1]} muscles from {pinch0}.")
else:
    print("Failed to load data for the single selected gesture.")


# Get file paths for (pose_name) dataset
pinch_paths = get_pose_paths(config, "pinch")
ulnar_paths = get_pose_paths(config, "ulnar")
power_paths = get_pose_paths(config, "power")

poses_paths = [pinch_paths, ulnar_paths, power_paths]
poses_names = config["pinch_ulnar_power"]

# Load and concatenate EMG data from all repetitions of (pose_name)
dict_data_single = load_pose_repetitions(pinch_paths, config["emg_topic"], "pinch")
dict_data_multiple = load_combined_pose_repetitions(poses_paths, config["emg_topic"], poses_names)



