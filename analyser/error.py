from config.importer import *
from helper.config_help import *
from helper.error_help import *
from analyser.nmf import *
from analyser.pca import *


class emgError:
    
    def __init__(self, emg_data=None, emg_data_dict=None):

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


        # for pca
        self.max_components = 8
        self.random_state = 42
        self.svd_solver='full'



    def errors(self, metric_name='rmse', rec_type='snmf'):
        """
        
        Args: 
            metric_name: 'fro', 'rmse', 'vaf' 
            rec_type: 'snmf', 'cnmf', 'pca'
        
        Out: 
            list containing error computation for each component extracted
        """


        print(f"\nErrors computation...")
        print(f"\nMetric name: {metric_name}")
        print(f"Reconstruction type: {rec_type}")
        
        errors_list = []


        for n in range(1, self.max_synergies + 1):

            if rec_type == 'snmf':
                U, S_m = emgNMF.SparseNMF(self)
                reconstruct = emgNMF.NMF_reconstruction(self, n, U, S_m)

            elif rec_type == 'cnmf':
                U, S_m = emgNMF.ClassicNMF(self)
                reconstruct = emgNMF.NMF_reconstruction(self, n, U, S_m)

            elif rec_type == 'pca':
                S_m, U, mean = emgPCA.PCA(self)
                reconstruct = emgPCA.PCA_reconstruction(self, U, S_m, mean, n)

            else:
                print(f'\nInvalid reconstruction type name.')



            if metric_name == 'rmse':
                error = rmse(self.emg_data, reconstruct)
            elif metric_name == 'fro':
                error = frobenius_error(self.emg_data, reconstruct)
            elif metric_name == 'vaf':
                error = vaf(self.emg_data, reconstruct)
            else:
                print(f'\nInvalid metric name.')

            errors_list.append(error)
            print(f"    Error for {n} synergies: {error:.4f}")
        
        
        print("\n\n")
        return errors_list
    

