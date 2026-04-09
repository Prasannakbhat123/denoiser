import argparse
import csv
import json
import os
import time

import torch

from model import DenoisingNetwork
from train_model import create_synthetic_data
from training import denoise_with_method, calculate_mse_psnr
from utils import add_noise, set_seed


def _run_method(method, noisy, model, kernel_size, sigma):
    return denoise_with_method(
        noisy,
        method=method,
        model=model if method == "deep" else None,
        kernel_size=kernel_size,
        sigma=sigma,
    )


def _time_method(method, noisy, model, kernel_size, sigma, repeats, warmup, device):
    for _ in range(warmup):
        _ = _run_method(method, noisy, model, kernel_size, sigma)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(repeats):
        _ = _run_method(method, noisy, model, kernel_size, sigma)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.perf_counter()

    return ((end - start) * 1000.0) / repeats


def evaluate_on_device(args, device_name):
    device = torch.device(device_name)
    set_seed(args.seed)

    clean_images = create_synthetic_data(
        batch_size=args.samples,
        height=args.height,
        width=args.width,
    )
    clean_tensor = torch.from_numpy(clean_images).permute(0, 3, 1, 2).float().to(device)
    noisy_tensor = add_noise(clean_tensor, noise_type=args.noise_type, noise_level=args.noise_level)

    model = None
    methods = ["mean", "gaussian", "median"]

    if os.path.exists(args.weights_path):
        model = DenoisingNetwork(n_chan=3).to(device)
        state = torch.load(args.weights_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        methods.insert(0, "deep")

    per_image_rows = []
    summary_rows = []

    for method in methods:
        denoised = _run_method(method, noisy_tensor, model, args.kernel_size, args.sigma)

        mse_values = []
        psnr_values = []
        for idx in range(clean_tensor.shape[0]):
            m, p = calculate_mse_psnr(
                clean_tensor[idx: idx + 1],
                denoised[idx: idx + 1],
            )
            mse_values.append(m)
            psnr_values.append(p)
            per_image_rows.append(
                {
                    "device": device.type,
                    "image_size": args.height,
                    "method": method,
                    "image_index": idx,
                    "mse": m,
                    "psnr_db": p,
                }
            )

        latency_ms = _time_method(
            method,
            noisy_tensor,
            model,
            args.kernel_size,
            args.sigma,
            args.repeats,
            args.warmup,
            device,
        )

        summary_rows.append(
            {
                "device": device.type,
                "image_size": args.height,
                "method": method,
                "samples": args.samples,
                "noise_type": args.noise_type,
                "noise_level": args.noise_level,
                "mean_mse": float(sum(mse_values) / len(mse_values)),
                "mean_psnr_db": float(sum(psnr_values) / len(psnr_values)),
                "latency_ms": float(latency_ms),
                "images_per_second": float((args.samples * 1000.0) / latency_ms),
            }
        )

    return summary_rows, per_image_rows


def add_speedup(summary_rows):
    by_method = {}
    for row in summary_rows:
        by_method.setdefault(row["method"], {})[row["device"]] = row

    for method, rows in by_method.items():
        cpu_row = rows.get("cpu")
        gpu_row = rows.get("cuda")
        if cpu_row and gpu_row and gpu_row["latency_ms"] > 0:
            speedup = cpu_row["latency_ms"] / gpu_row["latency_ms"]
            cpu_row["speedup_vs_cpu"] = 1.0
            cpu_row["time_savings_pct"] = 0.0
            gpu_row["speedup_vs_cpu"] = float(speedup)
            gpu_row["time_savings_pct"] = float(max(0.0, (1.0 - (gpu_row["latency_ms"] / cpu_row["latency_ms"])) * 100.0))
        else:
            if cpu_row:
                cpu_row["speedup_vs_cpu"] = 1.0
                cpu_row["time_savings_pct"] = 0.0
            if gpu_row:
                gpu_row["speedup_vs_cpu"] = None
                gpu_row["time_savings_pct"] = None


def write_outputs(output_dir, summary_rows, per_image_rows):
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    summary_fields = [
        "device",
        "image_size",
        "method",
        "samples",
        "noise_type",
        "noise_level",
        "mean_mse",
        "mean_psnr_db",
        "latency_ms",
        "images_per_second",
        "speedup_vs_cpu",
        "time_savings_pct",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_path = os.path.join(output_dir, "benchmark_per_image.csv")
    detail_fields = ["device", "image_size", "method", "image_index", "mse", "psnr_db"]
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(per_image_rows)

    payload = {
        "summary": summary_rows,
        "per_image_count": len(per_image_rows),
    }
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return summary_path, detail_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark denoising methods on CPU and GPU")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--sizes", type=str, default="", help="Comma-separated square image sizes to benchmark, e.g. 64,128,256")
    parser.add_argument("--noise-type", type=str, default="gauss", choices=["gauss", "poiss"])
    parser.add_argument("--noise-level", type=int, default=25)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weights-path", type=str, default="model_weights.pth")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def _parse_sizes(args):
    if not args.sizes.strip():
        return [args.height]

    sizes = []
    for item in args.sizes.split(","):
        item = item.strip()
        if not item:
            continue
        sizes.append(int(item))

    if not sizes:
        return [args.height]
    return sizes


def main():
    args = parse_args()
    run_benchmark(args)


def run_benchmark(args):

    all_summary = []
    all_per_image = []

    sizes = _parse_sizes(args)

    for size in sizes:
        sized_args = argparse.Namespace(**vars(args))
        sized_args.height = size
        sized_args.width = size

        summary_cpu, per_image_cpu = evaluate_on_device(sized_args, "cpu")
        all_summary.extend(summary_cpu)
        all_per_image.extend(per_image_cpu)

        if (not sized_args.cpu_only) and torch.cuda.is_available():
            summary_gpu, per_image_gpu = evaluate_on_device(sized_args, "cuda")
            all_summary.extend(summary_gpu)
            all_per_image.extend(per_image_gpu)

    add_speedup(all_summary)

    summary_path, detail_path, json_path = write_outputs(args.output_dir, all_summary, all_per_image)

    print("Benchmark complete")
    print(f"Summary CSV: {summary_path}")
    print(f"Per-image CSV: {detail_path}")
    print(f"JSON: {json_path}")

    return {
        "summary_csv": summary_path,
        "detail_csv": detail_path,
        "json": json_path,
    }


if __name__ == "__main__":
    main()
