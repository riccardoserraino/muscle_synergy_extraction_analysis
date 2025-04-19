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

    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    U = pca.fit_transform(emg_data)             # Synergies over time
    S_m = pca.components_                       # Muscle patterns (Muscular synergy matrix)

    mean = pca.mean_

    if scale_U:
        # Scale scores by explained variance (makes them more comparable)
        U = U * np.sqrt(pca.explained_variance_ratio_)
    
    # Transpose to keep same structure as NMF function
    if S_m.shape[0] != n_components:
        S_m = S_m.T     # Ensure S_m has shape (n_synergies, n_muscles)
    

    return S_m, U, mean



#-------------------------------------------------------------------------------------------



