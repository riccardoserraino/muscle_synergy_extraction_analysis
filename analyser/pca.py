from config.importer import *
from helper.config_help import *
from helper.pca_help import *


class emgPCA:
    """
    A class to perform Principal Component Analysis (PCA) on EMG data.
    """

    def __init__(self, emg_data=None, emg_data_dict=None):
        """
        Initialize the PCA class with parameters and data.
        
        Parameters:
        - emg_data: Input EMG data for PCA analysis
        - emg_data_dict: Dictionary of EMG data for cross-validation
        """
        self.emg_data = emg_data
        self.emg_data_dict = emg_data_dict
        
        # PCA parameters
        self.max_components = 8
        self.random_state = 42
        self.svd_solver='full'
    

    def pca(self):
        """
        Perform PCA on the input data.
        
        Returns:
        - components: Principal components
        - scores: Projection scores
        - explained_variance: Variance explained by each component
        """
        print("\nApplying PCA...")
        S_m, U = apply_pca(
            emg_data=self.emg_data,
            n_components=self.max_components,
            svd_solver=self.svd_solver,
            random_state=self.random_state
        )
        print("\n\n")
        return S_m, U
    


    def reconstruct_data(self, U, S_m, n_components):
        """
        Reconstruct data using selected number of components.
        
        Returns:
        - reconstructed: Reconstructed data matrix
        """
        print("\nReconstructing data...")
        print("\n\n")

        return pca_reconstruction(U, S_m, n_components)
    