from config.importer import *


def get_pose_paths(yaml_config, pose_name):
    """
    Gets file paths for all three reps for a specific pose.
    
    Args:
        yaml_config (dict): Parsed YAML configuration
        pose_name (str): The pose name ("pinch", "ulnar", or "power")
        
    Returns:
        dict: Dictionary with pose reps as keys and full paths as values
    """
    base_path = yaml_config["base_data_dir"]
    pose_reps = yaml_config["single_pose"].get(pose_name, {})

    return {
        rep: os.path.join(base_path, filename)
        for rep, filename in pose_reps.items()
    }
      
