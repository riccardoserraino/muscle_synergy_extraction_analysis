from config.importer import *
from helper.config_help import *
from helper.pca_help import *


class emgPCA:
    """
    A class to perform Principal Component Analysis (PCA) on EMG data.
    """

    def __init__(self, emg_data=None):
        """
        Initialize the PCA class with parameters and data.
        
        Parameters:
        - emg_data: Input EMG data for PCA analysis
        - emg_data_dict: Dictionary of EMG data for cross-validation
        """
        self.emg_data = emg_data
        
        # PCA parameters
        self.max_components = 8
        self.random_state = 42
        self.svd_solver='full'
    

    def PCA(self):
        """
        Perform PCA on the input data.
        
        Returns:
        - components: Principal components
        - scores: Projection scores
        - explained_variance: Variance explained by each component
        """
        print("\nApplying PCA...")
        S_m, U, mean, rec = apply_pca(
            emg_data=self.emg_data,
            n_components=self.max_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state
        )
        print("\n\n")
        return S_m, U, mean, rec
    


    def PCA_reconstruction(self, U, S_m, mean, n_components):
        """
        Reconstruct data using selected number of components.
        
        Returns:
        - reconstructed: Reconstructed data matrix
        """

        print("\nReconstructing data...")
        print("\n\n")

        # Select the first n_components
        U_rec = U[:, :n_components]
        S_m_rec = S_m[:n_components, :]
        
        # Reconstruct the data
        reconstructed = np.dot(U_rec, S_m_rec) + mean
        
        """original_mean = np.mean(self.emg_data)
        reconstructed_mean = np.mean(reconstructed)
        print(f"Mean difference: {original_mean - reconstructed_mean}")"""

        return reconstructed
    