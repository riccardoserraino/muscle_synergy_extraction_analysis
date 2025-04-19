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
        vaf_percent: Variance accounted for (accuracy)
    """

    SS_error = np.sum((X - X_estimated) ** 2)       # Unexplained variance
    SS_total = np.sum(X ** 2)                       # Total variance (assuming mean = 0)
    vaf = 1 - (SS_error / SS_total)                 # Variance explained percentage
    
    return vaf



# frob_error_percent
def frobenius_error(X, X_estimated):
    """
    Computes the Frobenius norm error between original and estimated data.

    Args:
        X: Original data matrix (n_features x n_samples)
        X_estimated: Estimated data matrix (n_features x n_samples)

    Outputs:
        frobenius_error: Frobenius norm of the error (accuracy)

    """

    error_norm = np.linalg.norm(X - X_estimated, 'fro')  # Frobenius norm of error
    data_norm = np.linalg.norm(X, 'fro')                 # Frobenius norm of original data

    frobenius_error = 1 - (error_norm / data_norm)      # Relative error  

    return frobenius_error



# rmse 

def rmse(X, X_estimated):

    """
    Computes the Root Mean Square Error (RMSE) between original and estimated data.

    Args:
        X: Original data matrix (n_samples x n_features) 
        X_estimated: Estimated data matrix (n_samples x n_features)

    Outputs:
        rmse_value: RMSE value (accuracy)
    """

    X = np.array(X)
    X_estimated = np.array(X_estimated)

    if X.shape != X_estimated.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    mse = np.mean((X-X_estimated)**2)

    rmse = np.sqrt(mse)

    return 1 - rmse



#--------------------------------------------------------------------------------------------



