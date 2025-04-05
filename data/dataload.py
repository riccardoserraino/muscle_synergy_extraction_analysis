from config.importer import *
from utils.config_help import *
from utils.data_help import *

class MuscleSynergyDataLoader:
    """
    A class to handle loading of EMG data and configuration for muscle synergy analysis.
    """
    
    def __init__(self, config_path, pose_name=None):
        """
        Initialize the DataLoader with configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            pose_name (str): The pose name ("pinch", "ulnar", or "power").
        """
        self.config_path = config_path
        self.pose_name = pose_name  
      

    def single_pose(self): 
        """
        Load and preprocess EMG data for a single pose.

        Returns:  
            dict: Dictionary containing the loaded EMG data.
        """
        # Load YAML file
        with open(self.config_path, "r") as f:  # Fixed: Use self.config_path instead of global config_path
            config = yaml.safe_load(f)
            if not config:
                raise ValueError("YAML file is empty or not properly formatted.")
        
        # Get file paths for the specified pose dataset
        pose_paths = get_pose_paths(config, self.pose_name)

        # Load and concatenate EMG data from all repetitions
        dict_data = load_pose_repetitions(pose_paths, config["emg_topic"], self.pose_name)
        
        return dict_data
    

    

    def combined_poses(self):
        """
        Load and combine EMG data from multiple poses into a single dictionary.
        
        Returns:
            dict: Keys are repetition numbers, values are concatenated data from all poses.
        """
        # Load YAML file
        with open(self.config_path, "r") as f:  # Fixed: Use self.config_path instead of global config_path
            config = yaml.safe_load(f)
            if not config:
                raise ValueError("YAML file is empty or not properly formatted.")
        
        pinch_paths = get_pose_paths(config, "pinch")
        ulnar_paths = get_pose_paths(config, "ulnar")
        power_paths = get_pose_paths(config, "power")

        poses_paths = [pinch_paths, ulnar_paths, power_paths]
        poses_names = config["pinch_ulnar_power"]


        # Load and concatenate EMG data from all repetitions
        combined_data = load_combined_pose_repetitions(poses_paths, config["emg_topic"], poses_names)
        
        return combined_data  


        