from config.importer import *

def scale_synergy_signal(W, emg_data):
    """
    Normalizes synergy activations to match original EMG amplitude range.
    
    Implementation:
    - Linear scaling preserving activation dynamics
    - Maintains non-negativity constraint
    - Handles both single and multi-channel EMG
    
    Args:
        W: Activation matrix (n_samples x n_synergies)
        emg_data: Original EMG (n_samples x n_muscles)
    """
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    W_min = np.min(W)
    W_max = np.max(W)
    W_scaled = ((W - W_min) / (W_max - W_min)) * (emg_max - emg_min) + emg_min
    W_scaled = np.maximum(W_scaled, 0)  # Ensures W_scaled is non-negative
    return W_scaled



def plot_all_results(emg_data, Z_reconstructed, W, H, selected_synergies):
    """
    Visualizes EMG analysis results in a 4-panel comparative plot.
    
    Implementation:
    - Creates figure with 4 vertically stacked subplots
    - Uses consistent scaling for comparison
    - Automatically handles variable synergy counts
    - Preserves non-negativity of muscle activations
    
    Args:
        emg_data: Raw EMG (n_samples x n_muscles)
        Z_reconstructed: Reconstructed signal (n_samples x n_muscles)
        W_scaled: Normalized activations (n_samples x n_synergies)
        H: Synergy components (n_synergies x n_muscles)
        selected_synergies: Number of synergies 
        
    Produces:
        Interactive matplotlib figure with:
        1. Original EMG signals
        2. Reconstructed signals
        3. Synergy activations over time
        4. Muscle weightings per synergy
        """
    
    print(f'Plotting results...')

    W_scaled = scale_synergy_signal(W, emg_data)

    plt.figure(figsize=(10, 8))
    
    # Panel 1: Original EMG Signals
    plt.subplot(4, 1, 1)
    plt.plot(emg_data)
    plt.title('Original EMG Signals')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])  # Remove x-axis labels for cleaner visualization
    
    # Panel 2: Reconstructed EMG Signals
    plt.subplot(4, 1, 2)
    plt.plot(Z_reconstructed, linestyle='--')
    plt.title(f'Reconstructed EMG ({selected_synergies} Synergies)')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])
    
    # Panel 3: Synergy Activation Patterns
    plt.subplot(4, 1, 3)
    for i in range(selected_synergies):
        plt.plot(W_scaled[:, i], label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time')
    plt.ylabel('Activation')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])
    
    # Panel 4: Muscle Weighting Patterns
    plt.subplot(4, 1, 4)
    for i in range(selected_synergies):
        plt.plot(H[i, :], 'o-', label=f'Synergy {i+1}')
    plt.title('Muscle Weighting Patterns')
    plt.xlabel('Muscle Channel')
    plt.ylabel('Weight')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])

    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------



def plot_signal(data, name):

    plt.figure(figsize=(6, 4))

    plt.plot(data)
    plt.title(name)
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])  # Remove x-axis labels for cleaner visualization
    plt.legend()

    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------




def plot_signal_dict(emg_data_dict, pose_name=None):
    """
    Visualizes EMG data from a dictionary of repetitions.
    
    Args:
        emg_data_dict: Dictionary with rep numbers as keys and 
                      (emg_data, timestamps) tuples as values
        pose_name (str, optional): Name of the pose for title
    """
    if not isinstance(emg_data_dict, dict):
        raise ValueError("Input must be a dictionary of repetitions")
    
    num_reps = len(emg_data_dict)
    fig, axes = plt.subplots(num_reps, 1, figsize=(12, 3*num_reps))
    
    # Handle single repetition case
    if num_reps == 1:
        axes = [axes]
    
    title = 'EMG Signals' + (f' ({pose_name})' if pose_name else '')
    fig.suptitle(title, y=1.02)
    
    for ax, (rep_id, (emg_data, _)) in zip(axes, emg_data_dict.items()):
        if emg_data is None:
            continue
            
        ax.plot(emg_data)
        ax.set_title(f'Repetition {rep_id}')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_xticks([])
        
        # Add muscle labels if available
        if hasattr(emg_data, 'columns'):  # If pandas DataFrame
            ax.legend(emg_data.columns, loc='upper right', ncol=4)
    
    plt.tight_layout()
    plt.show()




#------------------------------------------------------------------------------------------


def plot_vaf(max_synergies=8, VAF_values=None):
    """
    Visualizes the Variance Accounted For (VAF) against the number of synergies.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_synergies+1), VAF_values, marker='o')
    plt.xlabel('Number of Synergies')
    plt.ylabel('VAF')
    plt.title('VAF vs Number of Synergies')
    plt.show()



#------------------------------------------------------------------------------------------




def plot_reconstruction(emg_data, Z_reconstructed, W, selected_synergies):
    """
    Visualizes EMG analysis results in a 3-panel comparative plot.
    
    Implementation:
    - Creates figure with 4 vertically stacked subplots
    - Uses consistent scaling for comparison
    - Automatically handles variable synergy counts
    - Preserves non-negativity of muscle activations
    
    Args:
        emg_data: Raw EMG (n_samples x n_muscles)
        Z_reconstructed: Reconstructed signal (n_samples x n_muscles)
        W_scaled: Normalized activations (n_samples x n_synergies)
        selected_synergies: Number of synergies 
        
    Produces:
        Interactive matplotlib figure with:
        1. Original EMG signals
        2. Reconstructed signals
        3. Synergy activations over time
        """
    
    print(f'Plotting results...')

    W_scaled = scale_synergy_signal(W, emg_data)

    plt.figure(figsize=(10, 6))
    
    # Panel 1: Original EMG Signals
    plt.subplot(3, 1, 1)
    plt.plot(emg_data)
    plt.title('Original EMG Signals')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])  # Remove x-axis labels for cleaner visualization
    
    # Panel 2: Reconstructed EMG Signals
    plt.subplot(3, 1, 2)
    plt.plot(Z_reconstructed, linestyle='--')
    plt.title(f'Reconstructed EMG ({selected_synergies} Synergies)')
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])
    
    # Panel 3: Synergy Activation Patterns
    plt.subplot(3, 1, 3)
    for i in range(selected_synergies):
        plt.plot(W_scaled[:, i], label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time')
    plt.ylabel('Activation')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])
    
    
    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------


def plot_synergies_separated(latent_space, syn_np, emg_data_np, rec_data_np):
    plt.figure(figsize=(10,6))
    scale_synergy_signal(syn_np, emg_data_np)
    for i in range(latent_space):
        plt.subplot(latent_space+1,1,i+1)
        plt.plot(syn_np[:,i], label=f'Synergy {i+1}')
        plt.legend()
    plt.subplot(latent_space+1,1,latent_space+1)
    plt.tight_layout()
    plt.plot(rec_data_np[:,0], 'k', label='Reconstructed EMG')
    plt.plot(emg_data_np[:,0], 'r', label='Original EMG')
    plt.legend()

    print("Synergy positivity check:", (syn_np >= 0).all())
    print("Synergy ranges:", [f"{i}: {syn_np[:,i].min():.2f}-{syn_np[:,i].max():.2f}" 
                            for i in range(latent_space)], '\n')
