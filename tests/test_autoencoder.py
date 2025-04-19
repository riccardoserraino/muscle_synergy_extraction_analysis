from analyser.nmf import *
from analyser.dataload import *
from helper.visualization_help import *
from analyser.autoencoder import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/int_unibo/muscle_synergy_analysis/config/config.yaml"

# Load emg data for all gestures combined
pinch_ulnar_power_000, ts_000 = emgDataLoader(config_dir).combined_dataset(reps='000', combination_name='pinch_ulnar_power')
single_pose, ts_single = emgDataLoader(config_dir, pose_name='pinch').single_dataset()

# Data tensor creation
data_tensor = torch.tensor(pinch_ulnar_power_000, dtype=torch.float32)
# data_tensor = torch.tensor(single_pose, dtype=torch.float32)

# Reset random seed for reproducibility
# This is important for consistent results across different runs
set_seed(42)

# Create model and trainer
latent_space=3
model = emgAutoencoder(latent_size=latent_space)
trainer = emgAutoencoderTrainer(model, data_tensor=data_tensor)

# Train the model
epochs = 50
trainer.train(epochs=epochs)

# Get synergies
recon_data, synergies = trainer.extract_synergies(data_tensor)

# Prepare for plotting
emg_data_np = data_tensor.numpy()
recon_data_np = recon_data.numpy()
synergies_np = synergies.numpy()

plot_synergies_separated(latent_space, synergies_np, emg_data_np, recon_data_np)
plot_reconstruction(emg_data_np, recon_data_np, synergies_np, selected_synergies=synergies.shape[1])


