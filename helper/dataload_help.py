from config.importer import *
from helper.config_help import * 


#-------------------------------------------------------------------------------------------------------------


def load_emg_data(bag_path, topic_name):
    """
    Load and preprocess EMG data from ROS bag files.
    
    Implementation:
    - Reads timestamped EMG data from specified ROS topic
    - Ensures proper array orientation (samples x muscles)
    - Handles potential transposition needs
    
    Usage:
    emg_data, timestamps = load_emg_data('path/to/data.bag', 'emg_topic')
    """
    timestamps = []
    emg_data = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, rawdata, timestamp in bag.read_messages(topics=[topic_name]):
            emg_data.append(rawdata.data)  # Extract EMG RMS values
            timestamps.append(timestamp.to_sec())  # Convert timestamp to seconds
    
    emg_data = np.array(emg_data)
    timestamps = np.array(timestamps)   

    # Ensure emg_data is shaped correctly (samples × muscles)
    if emg_data.shape[0] < emg_data.shape[1]:  # If (n_muscles, n_samples), transpose it
        emg_data = emg_data.T  # Now it’s (n_samples, n_muscles)
    
    """if emg_data.all() != None and timestamps.all() != None:
        print(f"Loaded {emg_data.shape[0]} samples and {emg_data.shape[1]} muscles from {bag_path}.")
    else:
        print("Failed to load data for the single selected gesture.")"""
    return emg_data, timestamps


#-------------------------------------------------------------------------------------------------------------


def load_pose_repetitions(pose_paths_dict, topic_name, pose_name):
    """
    Loads all repetitions for a given pose into a dictionary.
    
    Args:
        pose_paths_dict (dict): Dictionary with rep numbers as keys and paths as values
                               (output from get_pose_paths)
        topic_name (str): ROS topic name containing EMG data
        
    Returns:
        dict: Dictionary with rep numbers as keys and 
              (emg_data, timestamps) tuples as values
    """
    loaded_data = {}
    
    print(f"\nLoading data for {len(pose_paths_dict)} {pose_name} repetitions...\n")
    for rep_id, bag_path in pose_paths_dict.items():
        try:
            emg_data, timestamps = load_emg_data(bag_path, topic_name)
            loaded_data[rep_id] = (emg_data, timestamps)
            print(f"  - Rep {rep_id} loaded successfully")
            print(f"    Samples: {emg_data.shape[0]}, Muscles: {emg_data.shape[1]}")
            print(f"    Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        except Exception as e:
            print(f"Error loading Rep {rep_id} from {bag_path}: {str(e)}")
            loaded_data[rep_id] = (None, None)
    print(f"\nLoaded {len(loaded_data)} repetitions for {pose_name}.\n")
    return loaded_data


#-------------------------------------------------------------------------------------------------------------


def load_combined_emg_data(selected_paths, topic_name):
    """
    Load and combine EMG data from multiple ROS bag files.

    Works like the single dataset loader but is able to load n paths data at once comcatenated.

    Usage:
    emg_data, timestamps = load_combined_emg_data('list_with_bag_data_paths', 'emg_topic')
    """
    # Initialize empty lists for EMG data and timestamps
    emg_data_combined = []
    timestamps_combined = []
    

    for bag_path in selected_paths:
        # Load EMG data and timestamps
        emg_data, timestamps = load_emg_data(bag_path, topic_name)
        
        # Check and reshape data if necessary
        if emg_data.shape[0] < emg_data.shape[1]:
            emg_data = emg_data.T
        
        # Append data to the lists
        emg_data_combined.append(emg_data)
        timestamps_combined.append(timestamps)

    # Concatenate data into single arrays
    emg_data_combined = np.vstack(emg_data_combined)
    timestamps_combined = np.concatenate(timestamps_combined)

    return emg_data_combined, timestamps_combined


#-------------------------------------------------------------------------------------------------------------


def load_combined_pose_repetitions(pose_paths_dicts, topic_name, pose_names):
    """
    Loads and combines EMG data from multiple poses with matching repository numbers.
    
    Args:
        pose_paths_dicts (list of dict): List of path dictionaries (one per pose)
        topic_name (str): ROS topic name containing EMG data
        pose_names (list of str): Names of poses being combined (e.g., ["pinch", "ulnar", "power"])
        
    Returns:
        dict: Dictionary with rep numbers as keys and 
              (combined_emg_data, combined_timestamps) tuples as values
    """
    combined_data = {}
    
    # Verify we have names for all poses
    if len(pose_names) != len(pose_paths_dicts):
        raise ValueError("Number of pose names must match number of pose path dictionaries")
    
    # Get all common repository numbers
    common_reps = set.intersection(*[set(d.keys()) for d in pose_paths_dicts])
    
    print(f"\nCombining data for poses: {', '.join(pose_names)}")
    print(f"Found {len(common_reps)} common repositories\n")
    
    for rep_id in common_reps:
        try:
            emg_parts = []
            ts_parts = []
            
            for pose_name, pose_dict in zip(pose_names, pose_paths_dicts):
                bag_path = pose_dict[rep_id]
                emg_data, timestamps = load_emg_data(bag_path, topic_name)
                
                # Ensure proper orientation
                if emg_data.shape[0] < emg_data.shape[1]:
                    emg_data = emg_data.T
                
                emg_parts.append(emg_data)
                ts_parts.append(timestamps)
                
                print(f"  - {pose_name.capitalize()} Rep {rep_id} loaded")
                print(f"    Samples: {emg_data.shape[0]}, Muscles: {emg_data.shape[1]}")
                print(f"    Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
            
            # Combine all pose data for this repository
            combined_emg = np.vstack(emg_parts)
            combined_ts = np.concatenate(ts_parts)
            
            combined_data[rep_id] = (combined_emg, combined_ts)
            print(f"\nCombined repository {rep_id}: Total samples {combined_emg.shape[0]}\n")
            
        except Exception as e:
            print(f"Error combining Rep {rep_id}: {str(e)}")
            combined_data[rep_id] = (None, None)
    
    print(f"\nSuccessfully combined {len(combined_data)} repositories.\n")
    return combined_data