import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from model import DenoisingNetwork
from utils import add_noise
from training import loss_func
import os

def create_synthetic_data(batch_size=8, height=128, width=128):
    """Create synthetic clean images for training"""
    # Generate random patterns, gradients, and textures
    images = []
    for _ in range(batch_size):
        # Create a combination of patterns
        img = np.zeros((height, width, 3))
        
        # Add gradients
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Random gradients and patterns
        img[:, :, 0] = 0.5 + 0.3 * np.sin(5 * X) * np.cos(5 * Y)
        img[:, :, 1] = 0.5 + 0.3 * np.sin(3 * X + 2) * np.cos(7 * Y + 1)
        img[:, :, 2] = 0.5 + 0.3 * np.sin(7 * X + 1) * np.cos(3 * Y + 2)
        
        # Add some noise patterns
        img += 0.1 * np.random.randn(height, width, 3)
        img = np.clip(img, 0, 1)
        
        images.append(img)
    
    return np.array(images)

def simple_train_model():
    """Train the model with synthetic data for better denoising"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize model
    model = DenoisingNetwork(n_chan=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 50
    batch_size = 4
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 10  # Train on 10 batches per epoch
        
        for batch_idx in range(num_batches):
            # Generate synthetic data
            clean_images = create_synthetic_data(batch_size)
            clean_tensor = torch.from_numpy(clean_images).permute(0, 3, 1, 2).float().to(device)
            
            # Add noise to create training pairs
            noise_levels = [15, 25, 35, 45]
            noise_level = np.random.choice(noise_levels)
            noisy_tensor = add_noise(clean_tensor, noise_level=noise_level)
            
            # Training step
            optimizer.zero_grad()
            loss = loss_func(model, noisy_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Training completed! Model saved to 'model_weights.pth'")
    return model

if __name__ == "__main__":
    # Train and save the model
    trained_model = simple_train_model()
    print("Model training finished. You can now use the denoising app!")
