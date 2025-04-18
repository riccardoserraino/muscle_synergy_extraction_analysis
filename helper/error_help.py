from config.importer import *




#------------------------------------------------------------------------------------------
# function for errors computation from here to below
#------------------------------------------------------------------------------------------


# vaf
def vaf(X, X_estimated):
    """
    Computes the Variance Accounted For (VAF) between original and estimated data.

    Args:
        X: Original data matrix (n_features x n_samples)
        X_estimated: Estimated data matrix (n_features x n_samples)
    
    Output: 
        vaf_percent: Variance accounted for
    """

    SS_error = np.sum((X - X_estimated) ** 2)       # Unexplained variance
    SS_total = np.sum(X ** 2)                       # Total variance (assuming mean = 0)
    vaf = 1 - (SS_error / SS_total)                 # Variance explained percentage
    
    return vaf



# frob_error_percent
def frob_error(X, X_estimated):
    """
    Computes the Frobenius norm error between original and estimated data.

    Args:
        X: Original data matrix (n_features x n_samples)
        X_estimated: Estimated data matrix (n_features x n_samples)

    Outputs:
        error_norm: Frobenius norm of the error
        frob_error_percent: Error percentage

    """

    error_norm = np.linalg.norm(X - X_estimated, 'fro')  # Frobenius norm of error
    data_norm = np.linalg.norm(X, 'fro')                 # Frobenius norm of original data

    frobenius_error_percent = (error_norm / data_norm)*100       # Relative error percentage  

    return error_norm, frobenius_error_percent



# rmse 

def rmse(X, X_estimated):

    """
    Computes the Root Mean Square Error (RMSE) between original and estimated data.

    Args:
        X: Original data matrix (n_features x n_samples)
        X_estimated: Estimated data matrix (n_features x n_samples)

    Outputs:
        rmse_value: RMSE value
    """

    N = X.shape[0]                                          # Number of features
    frob_err, frob_percent = frob_error(X, X_estimated)     # Frobenius error and percentage
    
    rmse_value = np.sqrt(frob_err/N)  # RMSE calculation

    return rmse_value

