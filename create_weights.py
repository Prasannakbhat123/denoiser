import torch
from model import DenoisingNetwork

# Create model and save dummy weights for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingNetwork(n_chan=3).to(device)

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')
print("Model weights saved successfully!")
