import streamlit as st
import torch
from PIL import Image
import numpy as np
from model import DenoisingNetwork
from utils import add_noise
from training import denoise

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingNetwork(n_chan=3).to(device)
    try:
        model.load_state_dict(torch.load('model_weights.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model weights file not found! Please run train_model.py first.")
        return None, device

model, device = load_model()

st.title("🖼️ AI Image Denoising App")
st.write("Upload an image to remove noise using our trained denoising model!")

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    col1, col2 = st.columns(2)
    with col1:
        noise_level = st.slider("Select Noise Level to Add", 0, 50, 25, help="Higher values add more noise")
    with col2:
        show_comparison = st.checkbox("Show comparison", value=True)

    if uploaded_file is not None:
        # Load and process image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Resize if image is too large
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Add noise
        noisy_img = add_noise(image_tensor, noise_level=noise_level)
        
        # Denoise
        with st.spinner("Denoising image..."):
            denoised_img = denoise(model, noisy_img)

        # Display results
        if show_comparison:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image_np, caption="Original Image", use_container_width=True)
            with col2:
                st.image(noisy_img.cpu().squeeze().permute(1, 2, 0).numpy(), caption="Noisy Image", use_container_width=True)
            with col3:
                st.image(denoised_img.cpu().squeeze().permute(1, 2, 0).numpy(), caption="Denoised Image", use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.image(noisy_img.cpu().squeeze().permute(1, 2, 0).numpy(), caption="Noisy Image", use_container_width=True)
            with col2:
                st.image(denoised_img.cpu().squeeze().permute(1, 2, 0).numpy(), caption="Denoised Image", use_container_width=True)
        
        # Download button for denoised image
        denoised_np = denoised_img.cpu().squeeze().permute(1, 2, 0).numpy()
        denoised_pil = Image.fromarray((denoised_np * 255).astype(np.uint8))
        
        # Convert to bytes for download
        import io
        buf = io.BytesIO()
        denoised_pil.save(buf, format='PNG')
        
        st.download_button(
            label="📥 Download Denoised Image",
            data=buf.getvalue(),
            file_name="denoised_image.png",
            mime="image/png"
        )
        
        # Show image statistics
        with st.expander("📊 Image Statistics"):
            original_std = float(torch.std(image_tensor))
            noisy_std = float(torch.std(noisy_img))
            denoised_std = float(torch.std(denoised_img))
            
            st.write(f"**Original Image Std:** {original_std:.4f}")
            st.write(f"**Noisy Image Std:** {noisy_std:.4f}")
            st.write(f"**Denoised Image Std:** {denoised_std:.4f}")
            st.write(f"**Noise Reduction:** {((noisy_std - denoised_std) / noisy_std * 100):.1f}%")

else:
    st.error("Failed to load the model. Please ensure model_weights.pth exists.")
    st.info("To create the model weights, run: `python train_model.py`")
