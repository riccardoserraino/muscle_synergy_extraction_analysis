from analyser.error import *
from analyser.dataload import *
from helper.visualization_help import *


from analyser.nmf import *
from analyser.dataload import *
from helper.visualization_help import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/int_unibo/all_scripts/muscle_synergy_analysis/config/config.yaml"

loader = emgDataLoader(config_dir)

# Load emg data 

# Pinch x3
pinch_012, pinch_ts_012 = loader.combined_dataset(combination_name='pinch_x3', reps='012')

# Ulnar x3
ulnar_012, ulnar_ts_012 = loader.combined_dataset(combination_name='ulnar_x3', reps='012')

# Power x3
power_012, power_ts_012 = loader.combined_dataset(combination_name='power_x3', reps='012')

# Pinch x3
pinch_ulnar_power, pinch_ulnar_power_ts = loader.combined_dataset(combination_name='pinch_ulnar_power_x3', reps='012012012')



pinch_rmse_sparse = emgError(pinch_012).errors(metric_name='rmse', rec_type='snmf')
pinch_rmse_classic = emgError(pinch_012).errors(metric_name='rmse', rec_type='cnmf')
pinch_rmse_pca = emgError(pinch_012).errors(metric_name='rmse', rec_type='pca')

ulnar_rmse_sparse = emgError(ulnar_012).errors(metric_name='rmse', rec_type='snmf')
ulnar_rmse_classic = emgError(ulnar_012).errors(metric_name='rmse', rec_type='cnmf')
ulnar_rmse_pca = emgError(ulnar_012).errors(metric_name='rmse', rec_type='pca')

power_rmse_sparse = emgError(power_012).errors(metric_name='rmse', rec_type='snmf')
power_rmse_classic = emgError(power_012).errors(metric_name='rmse', rec_type='cnmf')
power_rmse_pca = emgError(power_012).errors(metric_name='rmse', rec_type='pca')

pinch_ulnar_power_rmse_sparse = emgError(pinch_ulnar_power).errors(metric_name='rmse', rec_type='snmf')
pinch_ulnar_power_rmse_classic = emgError(pinch_ulnar_power).errors(metric_name='rmse', rec_type='cnmf')
pinch_ulnar_power_rmse_pca = emgError(pinch_ulnar_power).errors(metric_name='rmse', rec_type='pca')



pinch = [pinch_rmse_sparse, pinch_rmse_classic, pinch_rmse_pca]
ulnar = [ulnar_rmse_sparse, ulnar_rmse_classic, ulnar_rmse_pca]
power = [power_rmse_sparse, power_rmse_classic, power_rmse_pca]
combined = [pinch_ulnar_power_rmse_sparse, pinch_ulnar_power_rmse_classic, pinch_ulnar_power_rmse_pca]

pose = ['Pinch', 'Ulnar', 'Power', 'Combined']
methods = ['Sparse NMF', 'Classic NMF', 'PCA']



bar_chart_errors(pose[0], methods, pinch_rmse_sparse, pinch)
bar_chart_errors(pose[1], methods, ulnar_rmse_sparse, ulnar)
bar_chart_errors(pose[2], methods, power_rmse_sparse, power)
bar_chart_errors(pose[3], methods, pinch_ulnar_power_rmse_sparse, combined)

