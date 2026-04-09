from __future__ import annotations

import io

import pandas as pd
import streamlit as st
from PIL import Image

from frontend_utils import (
    add_noise_to_image,
    benchmark_summary_frame,
    compare_all_methods,
    compute_metrics,
    load_benchmark_summary,
    load_model,
    prepare_image,
    run_denoising,
    tensor_to_rgb_array,
)


st.set_page_config(
    page_title="Denoiser Dashboard",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CUDA-Accelerated Image Denoising Dashboard")
st.caption("Upload a noisy image to denoise it directly, or use the synthetic-noise mode for controlled comparisons.")

model, device, model_ready = load_model()
benchmark_rows = load_benchmark_summary()
benchmark_df = benchmark_summary_frame()

with st.sidebar:
    st.header("Controls")
    input_mode = st.selectbox(
        "Input mode",
        ["Denoise uploaded noisy image", "Upload clean image and add synthetic noise"],
        index=0,
    )
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    reference_file = st.file_uploader(
        "Optional clean reference image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Use this only if you want MSE/PSNR against the clean image.",
    )
    method = st.selectbox("Primary method", ["deep", "mean", "gaussian", "median"], index=0)
    deep_passes = st.slider(
        "Deep passes",
        1,
        3,
        2,
        help="Apply deep denoising multiple times. Higher passes remove more noise but can soften details.",
    )
    noise_level = st.slider("Noise level", 0, 50, 25)
    noise_type = st.selectbox("Noise type", ["gauss", "poiss"], index=0)
    kernel_size = st.slider("Kernel size", 3, 9, 3, step=2)
    sigma = st.slider("Gaussian sigma", 0.5, 3.0, 1.0, step=0.1)
    show_comparison = st.checkbox("Show original/noisy/denoised comparison", value=True)
    compare_all = st.checkbox("Compare all methods on same image", value=True)
    run_button = st.button("Run Denoising", use_container_width=True)

    st.divider()
    st.subheader("Runtime")
    st.write(f"Device: {device.type.upper()}")
    st.write(f"Deep model ready: {'Yes' if model_ready else 'No'}")
    st.write(f"Benchmark records: {len(benchmark_rows)}")

if not model_ready:
    st.warning("Deep model weights were not found. Classical methods still work.")

if not uploaded_file:
    st.info("Upload an image to start. You can also inspect the benchmark tab below.")

image = None
image_np = None
image_tensor = None
reference_tensor = None
noisy_img = None
primary_denoised = None
metrics = None
all_method_outputs = {}
input_label = ""
input_tensor = None

if uploaded_file is not None:
    image, image_np, image_tensor = prepare_image(uploaded_file)

    if reference_file is not None:
        _, _, reference_tensor = prepare_image(reference_file)

    if input_mode == "Upload clean image and add synthetic noise":
        input_label = "Clean input"
        noisy_img = add_noise_to_image(image_tensor, noise_level=noise_level, noise_type=noise_type)
        input_tensor = noisy_img
    else:
        input_label = "Uploaded noisy input"
        noisy_img = image_tensor
        input_tensor = noisy_img

    if run_button:
        status = st.empty()
        progress = st.progress(0)

        status.info("Preparing tensors...")
        progress.progress(15)

        if input_mode == "Upload clean image and add synthetic noise":
            status.info("Added synthetic noise to the image.")
        else:
            status.info("Using the uploaded image as the noisy input.")
        progress.progress(35)

        if method == "deep" and not model_ready:
            status.error("Deep model is unavailable. Switch to a classical method or train model_weights.pth.")
        else:
            method_passes = deep_passes if method == "deep" else 1
            primary_denoised = run_denoising(input_tensor, method, model, kernel_size, sigma, passes=method_passes)
            status.info(f"Applied {method} denoising.")
            progress.progress(70)

            if reference_tensor is not None:
                metrics = compute_metrics(reference_tensor, input_tensor, primary_denoised)
                status.success("Computed image quality metrics against the clean reference.")
            elif input_mode == "Upload clean image and add synthetic noise":
                metrics = compute_metrics(image_tensor, input_tensor, primary_denoised)
                status.success("Computed image quality metrics against the clean input.")
            else:
                metrics = None
                status.success("Denoising complete.")

            progress.progress(100)

            if compare_all:
                all_method_outputs = compare_all_methods(
                    input_tensor,
                    model,
                    kernel_size,
                    sigma,
                    deep_passes=deep_passes,
                )
    else:
        if method == "deep" and not model_ready:
            st.warning("Deep model weights are not available yet. Please train the model or switch to a classical method.")
        else:
            method_passes = deep_passes if method == "deep" else 1
            primary_denoised = run_denoising(input_tensor, method, model, kernel_size, sigma, passes=method_passes)
            if reference_tensor is not None:
                metrics = compute_metrics(reference_tensor, input_tensor, primary_denoised)
            elif input_mode == "Upload clean image and add synthetic noise":
                metrics = compute_metrics(image_tensor, input_tensor, primary_denoised)
            else:
                metrics = None

            if compare_all:
                all_method_outputs = compare_all_methods(
                    input_tensor,
                    model,
                    kernel_size,
                    sigma,
                    deep_passes=deep_passes,
                )

main_tab, benchmark_tab, help_tab = st.tabs(["Denoise Image", "Benchmark Results", "How It Works"])

with main_tab:
    if image is None:
        st.subheader("No image loaded yet")
        st.write("Upload an image from the sidebar to see the denoising pipeline in action.")
    else:
        left, right = st.columns([1.15, 0.85], gap="large")

        with left:
            st.subheader("Preview")
            st.image(image_np, caption=f"Uploaded image ({input_label or 'input'})", use_container_width=True)

            if input_mode == "Upload clean image and add synthetic noise" and noisy_img is not None:
                st.image(tensor_to_rgb_array(noisy_img), caption="Synthetic noisy image", use_container_width=True)
            elif input_mode == "Denoise uploaded noisy image":
                st.image(tensor_to_rgb_array(noisy_img), caption="Uploaded noisy input", use_container_width=True)

            if primary_denoised is not None:
                st.image(tensor_to_rgb_array(primary_denoised), caption=f"Final denoised image ({method})", use_container_width=True)

            if primary_denoised is not None:
                denoised_np = tensor_to_rgb_array(primary_denoised)
                denoised_pil = Image.fromarray((denoised_np * 255).astype("uint8"))
                buffer = io.BytesIO()
                denoised_pil.save(buffer, format="PNG")
                st.download_button(
                    label="Download denoised image",
                    data=buffer.getvalue(),
                    file_name="denoised_image.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with right:
            st.subheader("What’s happening")
            if image_tensor is not None:
                st.metric("Input size", f"{image_tensor.shape[-2]} x {image_tensor.shape[-1]}")
            st.metric("Selected method", method)
            st.metric("Deep passes", deep_passes if method == "deep" else 1)
            st.metric("Input mode", "Noisy upload" if input_mode.startswith("Denoise") else "Synthetic noise")
            st.metric("Noise level", noise_level)
            st.metric("Noise type", noise_type)
            if metrics is not None:
                st.metric("Noisy PSNR", f"{metrics['noisy_psnr']:.2f} dB")
                st.metric("Denoised PSNR", f"{metrics['denoised_psnr']:.2f} dB")
                st.metric("Noisy MSE", f"{metrics['noisy_mse']:.6f}")
                st.metric("Denoised MSE", f"{metrics['denoised_mse']:.6f}")
                if metrics["noisy_std"] > 0:
                    reduction = ((metrics["noisy_std"] - metrics["denoised_std"]) / metrics["noisy_std"]) * 100
                    st.metric("Noise reduction", f"{reduction:.1f}%")
            else:
                st.info("Metrics need a clean reference image. Upload one in the sidebar if you want PSNR and MSE.")

            with st.expander("Processing details", expanded=True):
                if input_mode == "Upload clean image and add synthetic noise":
                    st.write("1. The uploaded clean image is converted to a tensor.")
                    st.write("2. Synthetic noise is added using the chosen noise type and level.")
                    st.write("3. The selected denoising method is applied.")
                    st.write("4. MSE and PSNR are computed against the clean reference.")
                else:
                    st.write("1. The uploaded image is treated as the noisy input.")
                    st.write("2. The selected denoising method is applied directly.")
                    st.write("3. If a clean reference is uploaded, MSE and PSNR are computed.")
                    st.write("4. The final denoised image is available for download.")

        if compare_all and all_method_outputs:
            st.subheader("Method comparison")
            grid_cols = st.columns(len(all_method_outputs))
            for index, (method_name, output_tensor) in enumerate(all_method_outputs.items()):
                with grid_cols[index]:
                    st.image(
                        tensor_to_rgb_array(output_tensor),
                        caption=method_name.capitalize(),
                        use_container_width=True,
                    )

        if compare_all and noisy_img is not None and all_method_outputs:
            comparison_rows = []
            for method_name, output_tensor in all_method_outputs.items():
                if reference_tensor is not None:
                    method_metrics = compute_metrics(reference_tensor, input_tensor, output_tensor)
                    comparison_rows.append(
                        {
                            "method": method_name,
                            "mse": method_metrics["denoised_mse"],
                            "psnr_db": method_metrics["denoised_psnr"],
                        }
                    )
                elif input_mode == "Upload clean image and add synthetic noise":
                    method_metrics = compute_metrics(image_tensor, input_tensor, output_tensor)
                    comparison_rows.append(
                        {
                            "method": method_name,
                            "mse": method_metrics["denoised_mse"],
                            "psnr_db": method_metrics["denoised_psnr"],
                        }
                    )
            if comparison_rows:
                st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

with benchmark_tab:
    st.subheader("Saved benchmark output")
    if benchmark_df.empty:
        st.info("No benchmark artifacts found yet. Run benchmark.py or run_pipeline.py to generate results.")
    else:
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)
        if {"method", "mean_psnr_db"}.issubset(benchmark_df.columns):
            chart_df = benchmark_df[["method", "mean_psnr_db", "latency_ms"]].set_index("method")
            st.bar_chart(chart_df, use_container_width=True)

with help_tab:
    st.subheader("How to use this dashboard")
    st.write("1. Upload a noisy image to denoise it directly, or choose synthetic-noise mode for a controlled demo.")
    st.write("2. Choose a denoising method and optional clean reference image.")
    st.write("3. Click Run Denoising to see the final denoised output and download it.")
    st.write("4. Open the benchmark tab to view saved CPU/GPU comparison results.")
    st.write("5. Use compare-all mode to see all methods on the same noisy input.")

    st.subheader("Next steps")
    st.write("For stronger results, train on a larger and more diverse image set, then rerun the benchmark.")