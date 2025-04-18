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
        poses_names = ["pinch", "ulnar", "power"]  

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
        
        print(f"\nLoading single dataset for {self.pose_name} rep {rep_id}...\n")
        print(f'Succesfully loaded {self.pose_name} for rep {rep_id}.')
        print(f"  -Samples: {emg_data.shape[0]}")
        print(f"  -sEMG: {emg_data.shape[1]}\n\n")
         
        return emg_data, timestamps
    


    def combined_dataset_3(self, pinch_rep="0", ulnar_rep="0", power_rep="0"):
        """Combine specific repetitions from different poses, 3 datasets in the order pinch, ulnar, power"""
        config = self._load_config()
        paths = [
            get_pose_paths(config, "pinch")[pinch_rep],
            get_pose_paths(config, "ulnar")[ulnar_rep],
            get_pose_paths(config, "power")[power_rep]
        ]

        emg_data_combined, timestamps_combined = load_combined_emg_data(paths, config["emg_topic"])
        print(f"\nLoading combined data for the 3 gestures of rep {pinch_rep}{ulnar_rep}{power_rep}...\n")
        print(f'Succesfully loaded pinch, ulnar, power for rep {pinch_rep}{ulnar_rep}{power_rep}.')
        print(f"  -Samples: {emg_data_combined.shape[0]}")
        print(f"  -sEMG: {emg_data_combined.shape[1]}\n\n")

        return emg_data_combined, timestamps_combined


    
    def combined_dataset(self, combination_name=None, reps=None):
        """
        Combine datasets from a predefined pose combination in config.yaml
        
        Args:
            combination_name (str): Key in config specifying which poses to combine 
                                (e.g., "pinch_ulnar_power"). If None, uses first combination.
            reps (str/list): Repetition indices for each pose. Defaults to "0" for all.
        
        Returns:
            tuple: (emg_data_combined, timestamps_combined)
        """

        config = self._load_config()
        
        # Get available combinations from config
        available_combinations = {k: v for k, v in config.items() 
                                if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v)}
        
        if not available_combinations:
            raise ValueError("No valid pose combinations found in config. Expected format: \n"
                        "combination_name: [pose1, pose2, ...]")
        
        # Select combination
        if combination_name is None:
            combination_name = next(iter(available_combinations))
            print(f"No combination specified. Using first available: '{combination_name}'")
        
        try:
            pose_names = available_combinations[combination_name]
        except KeyError:
            raise ValueError(f"Combination '{combination_name}' not found in config. "
                            f"Available: {list(available_combinations.keys())}")
        
        # Process reps
        if reps is None:
            reps = "0" * len(pose_names)
        elif isinstance(reps, (list, tuple)):
            reps = "".join(str(r) for r in reps)
        elif isinstance(reps, str):
            if len(reps) != len(pose_names):
                raise ValueError(f"Reps string length ({len(reps)}) must match "
                            f"number of poses ({len(pose_names)})")
        else:
            raise TypeError("reps must be str (e.g., '021'), list, or tuple")
        
        # Build paths
        paths = []
        for pose, rep in zip(pose_names, reps):
            try:
                paths.append(get_pose_paths(config, pose)[rep])
            except KeyError:
                raise ValueError(f"Pose '{pose}' or rep '{rep}' not found in config")
        
        # Load data
        emg_data_combined, timestamps_combined = load_combined_emg_data(paths, config["emg_topic"])
        
        print(f"\nLoading combination '{combination_name}'...")
        print(f"\nSuccesfully loaded poses: {', '.join(f'{p}(rep{rep})' for p, rep in zip(pose_names, reps))}")
        print(f"  -Samples: {emg_data_combined.shape[0]}")
        print(f"  -sEMG: {emg_data_combined.shape[1]}\n\n")

        return emg_data_combined, timestamps_combined
    



        