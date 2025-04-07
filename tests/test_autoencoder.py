from analyser.nmf import *
from analyser.dataload import *
from helper.visualize_help import *
from analyser.autoencoder import *

# Initialization for yaml file directory
config_dir = "C:/Users/ricca/Desktop/th_unibo/muscle_synergy_analysis/config/config.yaml"

loader = emgDataLoader(config_dir)

# Load emg data for all gestures combined
pinch_ulnar_power_000, ts_000 = loader.combined_dataset('0', '0', '0')

# Data tensor creation
data_tensor = torch.tensor(pinch_ulnar_power_000, dtype=torch.float32)

# Reset random seed for reproducibility
# This is important for consistent results across different runs
set_seed(42)

# Create model and trainer
latent_space=2
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



