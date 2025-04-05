from config.importer import *
from helper.config_help import *
from helper.nmf_help import *

class NMF:
    """
    A class to perform Non-negative Matrix Factorization (NMF) tasks on EMG data.
    """
    
    def __init__(self, emg_data, min_synergies, max_synergies, init, max_iter, l1_ratio, alpha_W, random_state):
        """
        Initialize the NMF class with parameters and data.
        
        Parameters:
        - emg_data: Input EMG data for NMF analysis.
        - min_synergies: Minimum number of synergies to consider.
        - max_synergies: Maximum number of synergies to consider.
        - init: Initialization method for NMF.
        - max_iter: Maximum number of iterations for NMF.
        - l1_ratio: L1 ratio for sparse NMF.
        - alpha_W: Regularization parameter for W matrix in NMF.
        - random_state: Random seed for reproducibility.
        """
        self.emg_data = emg_data
        self.min_synergies = 1
        self.max_synergies = 8
        self.init = 'nndsvd'
        self.max_iter = 500
        self.l1_ratio = 0
        self.alpha_W = 0
        self.random_state = 42

    def SparseNMF(self):
        """
        Perform sparse NMF on the input data.
        
        Parameters:
        - X: Input data matrix.
        - n_components: Number of components (synergies) for NMF.
        - init: Initialization method for NMF.
        - max_iter: Maximum number of iterations for NMF.
        - l1_ratio: L1 ratio for sparse NMF.
        - alpha_W: Regularization parameter for W matrix in NMF.
        - random_state: Random seed for reproducibility.
        
        Returns:
        - W: Basis matrix (synergies).
        - H: Coefficient matrix (activation).
        """
        model = NMF(n_components=self.max_synergies, init=self.init, max_iter=self.max_iter,
                    l1_ratio=0.8, alpha_W=0.001, random_state=self.random_state)
        W = model.fit_transform(self.emg_data)
        H = model.components_
        
        return W, H
    
    def SparseNMF_reconstruction(self):
        """
        Reconstruct the original data using the NMF components.
        
        Returns:
        - reconstructed_data: Reconstructed data matrix.
        """
        W, H = self.SparseNMF()
        reconstructed_data = np.dot(W, H)
        
        return reconstructed_data
    
    def ClassicNMF(self):
        """
        Perform classic NMF on the input data.
        
        Parameters:
        - X: Input data matrix.
        - n_components: Number of components (synergies) for NMF.
        - init: Initialization method for NMF.
        - max_iter: Maximum number of iterations for NMF.
        - random_state: Random seed for reproducibility.
        
        Returns:
        - W: Basis matrix (synergies).
        - H: Coefficient matrix (activation).
        """
        model = NMF(n_components=self.n_components, init=self.init, max_iter=self.max_iter,
                    random_state=self.random_state)
        W = model.fit_transform(self.emg_data)
        H = model.components_
        
        return W, H

    def ClassicNMF_reconstruction(self):
        """
        Reconstruct the original data using the NMF components.
        
        Returns:
        - reconstructed_data: Reconstructed data matrix.
        """
        W, H = self.ClassicNMF()
        reconstructed_data = np.dot(W, H)
        
        return reconstructed_data
    
    

