from config.importer import *
from helper.config_help import *
from helper.nmf_help import *



class emgNMF:
    """
    A class to perform Non-negative Matrix Factorization (NMF) tasks on EMG data.
    """
    
    def __init__(self, emg_data=None, emg_data_dict=None):
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
        self.emg_data_dict = emg_data_dict

        self.min_synergies = 1
        self.max_synergies = 8
        self.init = 'nndsvd'
        self.max_iter = 500
        self.l1_ratio = 0
        self.sl1_ratio = 0.7
        self.alpha_W = 0
        self.random_state = 42

        self.n_folds = 3
        self.alpha_values = [0.001, 0.005, 0.01, 0.05]

        self.var_weight = 0.5
        self.sparsity_weight = 0.4
        self.cond_weight = 0.1


    def SparseNMF(self, S_alpha_W=0.001):
        """
        Perform sparse NMF on the input data.
        
        Automatic initialization at l1_ratio=0.7 
                                    alpha_W=0.001

        Returns:
        - W: Basis matrix (synergies).
        - H: Coefficient matrix (activation).
        """
        # Check if the data is provided
        if (S_alpha_W == 0):
            print("Warning: Alpha not received, using automatic parameter 0.001.")
        
        W, H  = apply_nmf(emg_data=self.emg_data, n_components=self.max_synergies, init=self.init, max_iter=self.max_iter,
                        l1_ratio=self.sl1_ratio, alpha_W=S_alpha_W, random_state=self.random_state)
        
        return W, H
    


    def ClassicNMF(self):
        """
        Perform classic NMF on the input data.
        
        Returns:
        - W: Basis matrix (synergies).
        - H: Coefficient matrix (activation).
        """
        W, H = apply_nmf(self.emg_data, self.max_synergies, init=self.init, max_iter=self.max_iter,
                        l1_ratio=self.l1_ratio, alpha_W=self.alpha_W, random_state=self.random_state)
        
        return W, H



    def NMF_reconstruction(self, n_synergies, W, H):
        """
        Reconstruct the original data using the NMF components.
        
        Returns:
        - reconstructed_data: Reconstructed data matrix.
        - W_rec: Basis matrix (synergies) for the selected components.
        - H_rec: Coefficient matrix (activation) for the selected components.
        """

        # Select the first n_synergies components
        W_rec = W[:, :n_synergies]
        H_rec = H[:n_synergies, :]

        # Reconstruct the original data from the selected components
        reconstructed = np.dot(W_rec, H_rec)
        
        return reconstructed
    



    def synergy_sparsity_extractor(self):
        """
        Comprehensive NMF analysis with proper data validation.
        """
        print("=== Starting Synergy & Sparsity analysis ===")

        data_dictionary = {k: v[0] for k, v in self.emg_data_dict.items() if v[0] is not None}


        # Initial CV with relaxed sparsity
        optimal_synergies, cv_results = cross_validate_synergies(
            reps_dict=data_dictionary,
            max_synergies=self.max_synergies,
            alpha=self.alpha_W, 
            l1_ratio=self.l1_ratio,
            min_synergies=self.min_synergies
        )
        
        # Sparsity optimization
        sparsity_results = sparsity_evaluation(
            data_dictionary,
            optimal_synergies,
            alpha_values=self.alpha_values,
            l1_ratio=self.l1_ratio,
        )
        
        # Select best alpha
        if not sparsity_results:
            alpha_result  = {'alpha': 0, 'variance': 0, 'sparsity': 0}
            print("Warning: No valid sparsity results, using default alpha = 0")
        else:
            alpha_result  = best_alpha(sparsity_results, self.var_weight, self.sparsity_weight, self.cond_weight)
        

        
        print("\n=== Final Parameters ===")
        print(f"Optimal synergies: {optimal_synergies}")
        print(f"Best alpha: {alpha_result ['alpha']:.3f}")
        print(f"Variance explained: {alpha_result ['variance']:.2%}")
        print(f"Sparsity achieved: {alpha_result ['sparsity']:.1f}%")
        
        return optimal_synergies, alpha_result, cv_results, sparsity_results
