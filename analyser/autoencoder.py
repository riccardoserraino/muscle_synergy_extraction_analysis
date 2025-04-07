from config.importer import *
from helper.config_help import *
from helper.autoencoder_help import *


class emgAutoencoder(nn.Module):
    def __init__(self, input_size=8, latent_size=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),  
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_size), 
            nn.Softplus()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )
        self._initialize_weights()
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  
                nn.init.zeros_(m.bias)


    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z



class emgAutoencoderTrainer:
    def __init__(self, model, data_tensor, batch_size=128, lr=1e-4, device='cpu'):
        
        set_seed(42)

        self.model = model.to(device)
        self.device = device
        
        # Min-max scaling (better for reconstruction)
        self.data_min = data_tensor.min(0)[0]
        self.data_max = data_tensor.max(0)[0]
        self.data_tensor = (data_tensor - self.data_min) / (self.data_max - self.data_min + 1e-6)
        
        self.dataloader = DataLoader(TensorDataset(self.data_tensor), 
                              batch_size=batch_size, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        

    def train(self, epochs=50):  
        self.model.train()
        print(f'Training Autoencoder...\n')
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.dataloader:
                inputs = batch[0].to(self.device)
                self.optimizer.zero_grad()
                recon, z = self.model(inputs)
                loss = self.criterion(recon, inputs)
                loss += 0.001 * torch.mean(torch.abs(z))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}") 
        print(f'\nTraining Ended.\n')
    

    def extract_synergies(self, data_tensor=None):
        self.model.eval()
        with torch.no_grad():
            inputs = self.data_tensor.to(self.device)
            recon_data, synergies = self.model(inputs)
            # Scale reconstruction back to original range
            recon_data = recon_data * (self.data_max - self.data_min) + self.data_min
        
        print("Synergies shape:", synergies.cpu().shape, '\n')
        
        return recon_data.cpu(), synergies.cpu()
    
    