from config.importer import *
from helper.error_help import *


def apply_pca(emg_data, n_components, scale_U=False, random_state=None, svd_solver='full'):
    """
    Applies Principal Component Analysis (PCA).
    
    Args:
        emg_data: Input EMG data (n_samples x n_muscles)
        n_components: Number of synergies to extract
        random_state: Random seed for reproducibility
        scale_scores: Whether to scale scores by explained variance
        svd_solver: intialization of the svd method used ('full is the standard LAPACK solver)
    
    Outputs:
        components (S_m): Principal components (n_components x n_muscles) (Muscle activation weight)
        scores (U): Projection of data onto components (n_samples x n_components) (Synergies over time)
        explained_variance: Variance explained by each component
    """

    #preprocessing data for a clear reconstruction (centering and normalization)
    X_centered = emg_data - np.mean(emg_data, axis=0)
    X_normalized = (emg_data - np.mean(emg_data, axis=0)) / np.std(emg_data, axis=0)

    #model pca
    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    
    #extracting matrices
    U = pca.fit_transform(emg_data)             # Synergies over time
    S_m = pca.components_                       # Muscle patterns (Muscular synergy matrix)
    mean = pca.mean_
    
    # Scale scores by explained variance (makes them more comparable)
    if scale_U:
        U = U * np.sqrt(pca.explained_variance_ratio_)
    # Transpose to keep same structure as NMF function
    if S_m.shape[0] != n_components:
        S_m = S_m.T     # Ensure S_m has shape (n_synergies, n_muscles)
    
    #reconstruction based on the inverse transform
    X_transformed = pca.fit_transform(X_centered) # Neural matrix (synergies over time) adjusted for centering wrt original data and enforce positive values
    X_reconstructed = pca.inverse_transform(X_transformed) + np.mean(emg_data, axis=0) # the mean is added to enforce values of synergies and reconstruction being non negative as the original data

    """mse = np.mean((emg_data - X_reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse}")"""

    return S_m, U, mean, X_reconstructed



#-------------------------------------------------------------------------------------------



