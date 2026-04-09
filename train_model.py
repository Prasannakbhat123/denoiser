import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model import DenoisingNetwork
from utils import add_noise, set_seed
from training import loss_func
import os
import argparse
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def create_synthetic_data(batch_size=8, height=128, width=128):
    """Create diverse synthetic clean images for training with multiple texture patterns"""
    images = []
    for _ in range(batch_size):
        img = np.zeros((height, width, 3))
        
        # Generate coordinate grids
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Randomly choose pattern type for diversity
        pattern_type = np.random.choice(['sinusoid', 'checkerboard', 'radial', 'combined'])
        
        if pattern_type == 'sinusoid':
            freq_x = np.random.uniform(2, 10)
            freq_y = np.random.uniform(2, 10)
            phase = np.random.uniform(0, 2 * np.pi)
            img[:, :, 0] = 0.5 + 0.3 * np.sin(freq_x * X + phase) * np.cos(freq_y * Y)
            img[:, :, 1] = 0.5 + 0.3 * np.sin(freq_y * X + 1) * np.cos(freq_x * Y + 1)
            img[:, :, 2] = 0.5 + 0.3 * np.sin((freq_x + freq_y) * X + 2) * np.cos((freq_x - freq_y) * Y + 2)
        
        elif pattern_type == 'checkerboard':
            checker_size = np.random.randint(4, 16)
            board = (((X * checker_size).astype(int) + (Y * checker_size).astype(int)) % 2).astype(float)
            img[:, :, 0] = 0.3 + 0.4 * board
            img[:, :, 1] = 0.4 + 0.3 * (1 - board)
            img[:, :, 2] = 0.5 + 0.2 * np.sin(X * checker_size * np.pi)
        
        elif pattern_type == 'radial':
            center_x, center_y = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)
            dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
            img[:, :, 0] = 0.5 + 0.3 * np.sin(10 * dist)
            img[:, :, 1] = 0.5 + 0.3 * np.cos(15 * dist)
            img[:, :, 2] = 0.5 + 0.25 * (dist / dist.max())
        
        else:  # combined
            img[:, :, 0] = 0.5 + 0.25 * np.sin(5 * X) * np.cos(5 * Y) + 0.15 * (((X * 8).astype(int) + (Y * 8).astype(int)) % 2).astype(float) - 0.075
            img[:, :, 1] = 0.5 + 0.25 * np.sin(7 * X + 1) * np.cos(3 * Y + 1) + 0.15 * np.sin(X * 20 * np.pi)
            img[:, :, 2] = 0.5 + 0.25 * np.sin(3 * X + 2) * np.cos(7 * Y + 2) + 0.15 * np.cos(Y * 20 * np.pi)
        
        # Add texture variation layer
        texture_scale = np.random.uniform(0.05, 0.15)
        img += texture_scale * np.random.randn(height, width, 3)
        img = np.clip(img, 0, 1)
        
        images.append(img)
    
    return np.array(images)


def _is_image_file(file_name):
    return file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))


def create_noisy_dataset(clean_dir, noisy_dir, noise_level=25, seed=42):
    """Create noisy copies of clean images and store in a separate directory."""
    set_seed(seed)
    os.makedirs(noisy_dir, exist_ok=True)

    clean_files = sorted([f for f in os.listdir(clean_dir) if _is_image_file(f)])
    if not clean_files:
        raise ValueError(f"No image files found in clean directory: {clean_dir}")

    created = 0
    for file_name in clean_files:
        clean_path = os.path.join(clean_dir, file_name)
        noisy_path = os.path.join(noisy_dir, file_name)

        with Image.open(clean_path) as img:
            img = img.convert("RGB")
            clean_np = np.asarray(img, dtype=np.float32) / 255.0

        clean_tensor = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0)
        noisy_tensor = add_noise(clean_tensor, noise_level=noise_level)
        noisy_np = noisy_tensor.squeeze(0).permute(1, 2, 0).numpy()
        noisy_np = np.clip(noisy_np * 255.0, 0, 255).astype(np.uint8)

        Image.fromarray(noisy_np).save(noisy_path)
        created += 1

    return created


class PairedImageDataset(Dataset):
    """Dataset for aligned clean/noisy image pairs with identical filenames."""

    def __init__(self, clean_dir, noisy_dir, image_size=128):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.image_size = image_size

        clean_files = {f for f in os.listdir(clean_dir) if _is_image_file(f)}
        noisy_files = {f for f in os.listdir(noisy_dir) if _is_image_file(f)}
        self.files = sorted(list(clean_files.intersection(noisy_files)))

        if not self.files:
            raise ValueError("No matching clean/noisy image pairs found. Filenames must match.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_path = os.path.join(self.noisy_dir, file_name)

        with Image.open(clean_path) as clean_img:
            clean_img = clean_img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
            clean_np = np.asarray(clean_img, dtype=np.float32) / 255.0

        with Image.open(noisy_path) as noisy_img:
            noisy_img = noisy_img.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
            noisy_np = np.asarray(noisy_img, dtype=np.float32) / 255.0

        clean_tensor = torch.from_numpy(clean_np).permute(2, 0, 1)
        noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1)
        return noisy_tensor, clean_tensor


def train_on_paired_dataset(clean_dir, noisy_dir, epochs=50, batch_size=8, image_size=128, seed=42, output_dir="artifacts"):
    """Train model using paired noisy-clean supervision."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    dataset = PairedImageDataset(clean_dir=clean_dir, noisy_dir=noisy_dir, image_size=image_size)
    val_size = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = DenoisingNetwork(n_chan=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.L1Loss()

    best_val_loss = float("inf")
    history = []
    patience = 10
    patience_counter = 0

    print(f"Paired training samples: {train_size}, validation samples: {val_size}")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        for noisy_batch, clean_batch in train_loader:
            noisy_batch = noisy_batch.to(device)
            clean_batch = clean_batch.to(device)

            optimizer.zero_grad()
            # Keep inference consistent with existing denoise() pipeline:
            # clean_pred = noisy - model(noisy), so model(noisy) should learn residual noise.
            pred_noise = model(noisy_batch)
            target_noise = noisy_batch - clean_batch
            loss = criterion(pred_noise, target_noise)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / max(1, len(train_loader))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for noisy_batch, clean_batch in val_loader:
                noisy_batch = noisy_batch.to(device)
                clean_batch = clean_batch.to(device)
                pred_noise = model(noisy_batch)
                target_noise = noisy_batch - clean_batch
                val_loss = criterion(pred_noise, target_noise)
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        scheduler.step()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (no validation improvement for {patience} epochs)")
            break

    if os.path.exists(os.path.join(output_dir, "model_best.pth")):
        model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pth"), map_location=device))

    torch.save(model.state_dict(), "model_weights.pth")
    print("\n✓ Paired training completed!")
    print(f"✓ Best model saved to 'model_weights.pth' (Val Loss: {best_val_loss:.6f})")

    training_metrics = {
        "seed": seed,
        "device": str(device),
        "mode": "paired",
        "clean_dir": clean_dir,
        "noisy_dir": noisy_dir,
        "image_size": image_size,
        "epochs_completed": len(history),
        "total_epochs_planned": epochs,
        "batch_size": batch_size,
        "train_samples": train_size,
        "val_samples": val_size,
        "best_val_loss": float(best_val_loss),
        "history": history,
    }

    with open(os.path.join(output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)

    return model

def simple_train_model(epochs=50, batch_size=4, batches_per_epoch=10, seed=42, output_dir='artifacts'):
    """Train the model with diverse synthetic data and validation tracking"""
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize model
    model = DenoisingNetwork(n_chan=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    num_epochs = epochs
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    print(f"Training for {num_epochs} epochs with {batches_per_epoch} batches per epoch...")
    print(f"Batch size: {batch_size}, Learning rate: 0.001")
    
    history = []
    
    # Create validation dataset (clean images + synthetic noise)
    val_clean = create_synthetic_data(batch_size=4, height=128, width=128)
    val_noise_levels = [15, 25, 35, 45]
    val_clean_tensor = torch.from_numpy(val_clean).permute(0, 3, 1, 2).float().to(device)

    for epoch in range(num_epochs):
        # Training loop
        total_train_loss = 0
        
        for batch_idx in range(batches_per_epoch):
            clean_images = create_synthetic_data(batch_size, height=128, width=128)
            clean_tensor = torch.from_numpy(clean_images).permute(0, 3, 1, 2).float().to(device)
            
            noise_level = np.random.choice(val_noise_levels)
            noisy_tensor = add_noise(clean_tensor, noise_level=noise_level)
            
            optimizer.zero_grad()
            loss = loss_func(model, noisy_tensor)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / batches_per_epoch
        
        # Validation loop - compute average loss on validation set
        model.eval()
        with torch.no_grad():
            val_losses = []
            for val_noise_level in val_noise_levels:
                val_noisy = add_noise(val_clean_tensor, noise_level=val_noise_level)
                val_loss = loss_func(model, val_noisy)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
        
        model.train()
        scheduler.step()
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "lr": float(optimizer.param_groups[0]['lr'])
        })
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Checkpointing: save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping (optional)
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Save final model and load best model
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, 'model_best.pth')):
        model.load_state_dict(torch.load(os.path.join(output_dir, 'model_best.pth'), map_location=device))
    
    torch.save(model.state_dict(), 'model_weights.pth')
    print("\n✓ Training completed!")
    print(f"✓ Best model saved to 'model_weights.pth' (Val Loss: {best_val_loss:.6f})")

    training_metrics = {
        "seed": seed,
        "device": str(device),
        "epochs_completed": len(history),
        "total_epochs_planned": num_epochs,
        "batch_size": batch_size,
        "batches_per_epoch": batches_per_epoch,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(history[-1]["train_loss"]) if history else None,
        "history": history,
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(training_metrics, f, indent=2)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train denoising model with proper hyperparameters and validation")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--batches-per-epoch", type=int, default=20, help="Number of batches per epoch (increase for more training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory to save metrics and checkpoints")
    parser.add_argument("--clean-dir", type=str, default="", help="Path to clean image folder for paired training")
    parser.add_argument("--noisy-dir", type=str, default="", help="Path to noisy image folder for paired training")
    parser.add_argument("--image-size", type=int, default=128, help="Resize used for paired training")
    parser.add_argument("--make-noisy-first", action="store_true", help="Generate noisy copies from clean images before training")
    parser.add_argument("--noise-level", type=int, default=25, help="Noise level used when --make-noisy-first is set")
    args = parser.parse_args()

    print("="*60)
    print("DENOISING MODEL TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {args.batches_per_epoch}")
    print(f"  Total training samples: {args.epochs * args.batches_per_epoch * args.batch_size}")
    print("="*60)
    
    if args.clean_dir and args.noisy_dir:
        if args.make_noisy_first:
            print(f"\nCreating noisy dataset from clean images...")
            created = create_noisy_dataset(
                clean_dir=args.clean_dir,
                noisy_dir=args.noisy_dir,
                noise_level=args.noise_level,
                seed=args.seed,
            )
            print(f"✓ Created {created} noisy images in: {args.noisy_dir}")

        trained_model = train_on_paired_dataset(
            clean_dir=args.clean_dir,
            noisy_dir=args.noisy_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        trained_model = simple_train_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            batches_per_epoch=args.batches_per_epoch,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    print("\n✓ Model training finished. You can now use the denoising app!")
    print(f"✓ Metrics saved to {args.output_dir}/training_metrics.json")
