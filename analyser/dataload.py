from config.importer import *
from helper.config_help import *
from helper.dataload_help import *

class emgDataLoader:
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
        self._config = None # Cache for loaded configuration


    def _load_config(self):
        """Helper method to load and cache configuration"""
        if self._config is None:
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f)
                if not self._config:
                    raise ValueError("YAML file is empty or not properly formatted.")
        return self._config
    


    def single_pose_dict(self): 
        """
        Load and preprocess EMG data for a single pose.

        Returns:  
            dict: Dictionary containing the loaded EMG data for all repetitions.
        """
        config = self._load_config()
        pose_paths = get_pose_paths(config, self.pose_name)
        return load_pose_repetitions(pose_paths, config["emg_topic"], self.pose_name)
    


    def combined_poses_dict(self):
        """
        Load and combine EMG data from multiple poses into a single dictionary.
        
        Returns:
            dict: Keys are repetition numbers, values are concatenated data from all poses.
        """
        config = self._load_config()
        pinch_paths = get_pose_paths(config, "pinch")
        ulnar_paths = get_pose_paths(config, "ulnar")
        power_paths = get_pose_paths(config, "power")

        poses_paths = [pinch_paths, ulnar_paths, power_paths]
        poses_names = ["pinch", "ulnar", "power"]  # Fixed: Use literal list instead of config key

        return load_combined_pose_repetitions(poses_paths, config["emg_topic"], poses_names) 



    def single_dataset(self, rep_id="0"):
        """
        Load a single dataset (one repetition) for the configured pose.
        
        Args:
            rep_id (str): The repetition ID to load ("0", "1", or "2")
            
        Returns:
            tuple: (emg_data, timestamps) for the specified repetition
        """
        config = self._load_config()
        pose_paths = get_pose_paths(config, self.pose_name)
        
        if rep_id not in pose_paths:
            raise ValueError(f"Repetition {rep_id} not found for pose {self.pose_name}")
            
        bag_path = pose_paths[rep_id]
        emg_data, timestamps = load_emg_data(bag_path, config["emg_topic"])
        
        print(f"\nLoaded single dataset for {self.pose_name} Rep {rep_id}:\n")
        print(f"  Samples: {emg_data.shape[0]}, Muscles: {emg_data.shape[1]}\n\n")
        
        return emg_data, timestamps
    


    def combined_dataset(self, pinch_rep="0", ulnar_rep="0", power_rep="0"):
        """Combine specific repetitions from different poses"""
        config = self._load_config()
        paths = [
            get_pose_paths(config, "pinch")[pinch_rep],
            get_pose_paths(config, "ulnar")[ulnar_rep],
            get_pose_paths(config, "power")[power_rep]
        ]

        emg_data_combined, timestamps_combined = load_combined_emg_data(paths, config["emg_topic"])
        print(f'\nLoading data...')
        print(f"\nLoaded combined dataset for gestures Rep {pinch_rep}{ulnar_rep}{power_rep}:")
        print(f"  Samples: {emg_data_combined.shape[0]}")
        print(f"  Muscles: {emg_data_combined.shape[1]}\n")

        return emg_data_combined, timestamps_combined

        