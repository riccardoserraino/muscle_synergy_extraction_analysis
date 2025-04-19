from config.importer import *



def scale_synergy_signal(X, emg_data):
    """
    Normalizes synergy activations to match original EMG amplitude range.
    
    Args:
        X: Activation matrix (n_samples x n_sEMG)
        emg_data: Original EMG (n_samples x n_sEMG)
    """
    
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    X_min = np.min(X)
    X_max = np.max(X)
    X_scaled = ((X - X_min) / (X_max - X_min)) * (emg_max - emg_min) + emg_min
    X_scaled = np.maximum(X_scaled, 0)  # Ensures W_scaled is non-negative
    return X_scaled



#----------------------------------------------------------------------------------------



def plot_all_results(emg_data, Z_reconstructed, U, S_m, selected_synergies):
    """
    Visualizes EMG analysis results in a 4-panel comparative plot.
    
    Args:
        emg_data: Raw EMG (n_samples x n_muscles)
        Z_reconstructed: Reconstructed signal (n_samples x n_muscles)
        H_scaled: Normalized activations (n_samples x n_synergies) (U or H)
        W: Synergy components (n_synergies x n_muscles) (S_m or W)
        selected_synergies: Number of synergies 
        
    Produces:
        Interactive matplotlib figure with:
        1. Original EMG signals
        2. Reconstructed signals
        3. Synergy activations over time (U or H)
        4. Muscle weightings per synergy (S_m or W)
    """
    
    print(f'\nPlotting results...\n\n')

    U_scaled = scale_synergy_signal(U, emg_data)


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
    
    # Panel 3: Synergy Activation Patterns over time
    plt.subplot(4, 1, 3)
    for i in range(selected_synergies):
        plt.plot(U_scaled[:, i], label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time')
    plt.ylabel('Activation')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])
    
    # Panel 4: Synergy Weighting Patterns
    plt.subplot(4, 1, 4)
    for i in range(selected_synergies):
        plt.plot(S_m[i, :], 'o-', label=f'Synergy {i+1}')
    plt.title('Synergy Weighting Patterns')
    plt.xlabel('EMG Channel')
    plt.ylabel('Weight')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])

    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------



def plot_signal(data, name):
    """
    Visualizes a single EMG signal.
    
    Args:
        data: EMG signal data 
        name (str): Name of the signal for title
    """
    plt.figure(figsize=(8, 4))

    plt.plot(data)
    plt.title(name)
    plt.ylabel('Amplitude (mV)')
    plt.xticks([])  # Remove x-axis labels for cleaner visualization

    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------




def plot_signal_dict(emg_data_dict, pose_name):
    """
    Visualizes EMG data from a dictionary of repetitions.
    
    Args:
        emg_data_dict: Dictionary with rep numbers as keys and 
                      (emg_data, timestamps) tuples as values
        pose_name (str, optional): Name of the pose for title
    """

    print(f'\nPlotting EMG signals...')
    print("\n\n")
    if not isinstance(emg_data_dict, dict):
        raise ValueError("Input must be a dictionary of repetitions")
    

    num_reps = len(emg_data_dict)
    fig, axes = plt.subplots(num_reps, 1, figsize=(8, 1+2*num_reps))
    
    # Handle single repetition case
    if num_reps == 1:
        axes = [axes]
    
    
    for ax, (rep_id, (emg_data, _)) in zip(axes, emg_data_dict.items()):
        if emg_data is None:
            continue
            
        ax.plot(emg_data)
        ax.set_title(f'EMG signals - {pose_name} (rep {rep_id})')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_xticks([])
    
    plt.tight_layout()
    plt.show()




#------------------------------------------------------------------------------------------


def plot_error(max_synergies=8, error=None, error_name=None):
    """
    Visualizes in the reconstruction against the number of synergies.

    Args:
        max_synergies (int): Maximum number of synergies to plot
        error (list): List of errors for each synergy count
        error_name (str): Name of the error metric for the plot title
    """
    print(f'\nPlotting Error')
    print("\n\n")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_synergies+1), error, marker='o')
    plt.xlabel('Number of Synergies')
    plt.ylabel(f'{error_name} computed')
    plt.title(f'{error_name} vs Number of Synergies')
    plt.show()



#------------------------------------------------------------------------------------------




def plot_reconstruction(emg_data, Z_reconstructed, U, selected_synergies):
    """
    Visualizes EMG analysis results in a 3-panel comparative plot.
    
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
    
    print(f'\nPlotting results...')
    print("\n\n")

    U_scaled = scale_synergy_signal(U, emg_data)

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
        plt.plot(U_scaled[:, i], label=f'Synergy {i+1}')
    plt.title('Synergy Activation Over Time')
    plt.ylabel('Activation')
    plt.legend(loc='upper right', ncol=selected_synergies)
    plt.xticks([])
    
    
    plt.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------

# used for the autoencoders analysis
def plot_synergies_separated(latent_space, syn_np, emg_data_np, rec_data_np):
    """
    Visualizes synergy activations and reconstructed EMG signals.
    
    Args:
        latent_space (int): Number of synergies
        syn_np: Synergy activations (n_samples x n_synergies)
        emg_data_np: Original EMG data (n_samples x n_muscles)
        rec_data_np: Reconstructed EMG data (n_samples x n_muscles)
        
    Produces:
        1. Synergy activations over time
        2. Reconstructed EMG signals
        3. Original EMG signals shadow 
    """
    print("\n\n")

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




#------------------------------------------------------------------------------------------



def bar_chart_errors(pose_name, methods, single_list, list_computations):
    # Settings for the bar chart
    num_values = len(single_list)
    x = np.arange(num_values)
    bar_width = 0.20  # width of a single bar

    # Plot each computation
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, comp in enumerate(list_computations):
        ax.bar(x + i * bar_width, comp, bar_width, label=methods[i])
        for j, val in enumerate(comp):
            # Adjusted y-position (val + 0.01) to place it just above the bar
            # Added rotation=45 for 45 degree angle
            ax.text(x[j] + i * bar_width, val + 0.001, f'{val:.3f}', 
                    ha='center', va='bottom', fontsize=7, rotation=45)

    # Add labels and formatting
    ax.set_xlabel('N° synergies')
    ax.set_ylabel('Accuracy computed')
    ax.set_ylim(0.8, None)
    ax.set_title(f'Comparison of Results Across Methods - {pose_name}')
    ax.set_xticks(x + bar_width)  # center tick labels
    ax.set_xticklabels([f'{i}' for i in range(1, num_values + 1)])
    ax.legend()

    plt.ylim(0.8, None)
    plt.tight_layout()
    plt.show()