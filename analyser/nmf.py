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

        # for nmf
        self.min_synergies = 1
        self.max_synergies = 8
        self.max_iter = 500
        self.random_state = 42

        # for sparse
        self.s_init = 'nndsvd' # better for sparsity
        self.s_l1_ratio = 0.7
        self.s_alpha_W = 0.01       
        
        # for classical
        self.init = 'nndsvdar' # better if we do not care about sparsity
        self.l1_ratio = 0
        self.alpha_W = 0

        # for cv analysis
        self.n_folds = 3

        # for sparsity evaluation
        self.alpha_values = [0.005, 0.01]
        self.l1_ratio_values = [0.4, 0.5, 0.6, 0.8]

        self.var_weight = 0.5
        self.sparsity_weight = 0.4
        self.cond_weight = 0.1


    def SparseNMF(self):
        """
        Perform sparse NMF on the input data.
        
        Automatic initialization at l1_ratio=0.7 
                                    alpha_W=0.001

        Returns:
        - W: Basis matrix (synergies).
        - H: Coefficient matrix (activation).
        """
        print("\nApplying Sparse NMF...")
        U, S_m  = apply_nmf(emg_data=self.emg_data, n_components=self.max_synergies, init=self.s_init, max_iter=self.max_iter,
                        l1_ratio=self.s_l1_ratio, alpha_W=self.s_alpha_W, random_state=self.random_state)
        print('\n\n')
        
        return U, S_m
    


    def ClassicNMF(self):
        """
        Perform classic NMF on the input data.
        
        Returns:
        - W: Basis matrix (synergies).
        - H: Coefficient matrix (activation).
        """
        print("\nApplying Classical NMF...")
        U, S_m = apply_nmf(self.emg_data, self.max_synergies, init=self.init, max_iter=self.max_iter,
                        l1_ratio=self.l1_ratio, alpha_W=self.alpha_W, random_state=self.random_state)
        print('\n\n')

        return U, S_m



    def NMF_reconstruction(self, n_synergies, U, S_m):
        """
        Reconstruct the original data using the NMF components.
        
        Returns:
        - reconstructed_data: Reconstructed data matrix.
        - W_rec: Basis matrix (synergies) for the selected components.
        - H_rec: Coefficient matrix (activation) for the selected components.
        
        """

        print(f"\nReconstructing the signal with {n_synergies} synergies")
        # Select the first n_synergies components
        U_rec = U[:, :n_synergies]
        S_m_rec = S_m[:n_synergies, :]

        # Reconstruct the original data from the selected components
        reconstructed = np.dot(U_rec, S_m_rec)
        print('\n\n')

        return reconstructed
    



    def synergy_sparsity_extractor(self):
        """
        Comprehensive NMF analysis with proper data validation.
        """
        print("\nStarting Synergy & Sparsity analysis...")

        data_dictionary = {k: v[0] for k, v in self.emg_data_dict.items() if v[0] is not None}


        # Initial CV with relaxed sparsity
        optimal_synergies, cv_results = cross_validate_synergies(
            reps_dict=data_dictionary,
            max_synergies=self.max_synergies,
            alpha_W=self.s_alpha_W, 
            l1_ratio=self.s_l1_ratio,
            min_synergies=self.min_synergies
        )
        
        # Sparsity optimization
        sparsity_results = sparsity_evaluation(
            data_dictionary,
            optimal_synergies,
            alpha_values=self.alpha_values,
            l1_ratio_values=self.l1_ratio_values,
        )
        
        # Select best alpha
        if not sparsity_results:
            alpha_result  = {'alpha': 0, 'variance': 0, 'sparsity': 0}
            print("Warning: No valid sparsity results, using default alpha = 0")
        else:
            alpha, l1  = best_sparsity_param(sparsity_results, self.var_weight, self.sparsity_weight, self.cond_weight)
        

        
        print("\nFinal Parameters...")
        print(f"Optimal synergies: {optimal_synergies}")
        print(f"Best alpha: {alpha}")
        print(f"Best l1_ratio: {l1}")
        
        return optimal_synergies, alpha, l1, cv_results, sparsity_results
