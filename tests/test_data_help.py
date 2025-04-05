from config.importer import *
from helper.config_help import *
from helper.data_help import *


# Load YAML file
with open("C:/Users/ricca/Desktop/th_unibo/muscle_synergy_analysis/config/config.yaml", "r") as f:
    config = yaml.safe_load(f)
    if not config:
        raise ValueError("YAML file is empty or not properly formatted.")


# Single pose reps loading convalidation ------------------------------------


# Get file paths for (pose_name) dataset
pinch_paths = get_pose_paths(config, "pinch")
ulnar_paths = get_pose_paths(config, "ulnar")
power_paths = get_pose_paths(config, "power")

poses_paths = [pinch_paths, ulnar_paths, power_paths]
poses_names = config["pinch_ulnar_power"]

# Load and concatenate EMG data from all repetitions of (pose_name)
dict_data_single = load_pose_repetitions(pinch_paths, config["emg_topic"], "pinch")
dict_data_multiple = load_combined_pose_repetitions(poses_paths, config["emg_topic"], poses_names)

