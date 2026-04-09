import argparse

from benchmark import run_benchmark
from train_model import simple_train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train the model and run CPU/GPU benchmarks in one command")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50 for report-quality training)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int, default=20, help="Batches per epoch (default: 20, increased for better training)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")

    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--noise-type", type=str, default="gauss", choices=["gauss", "poiss"])
    parser.add_argument("--noise-level", type=int, default=25)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--weights-path", type=str, default="model_weights.pth")
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("CUDA-ACCELERATED IMAGE DENOISING - TRAINING & BENCHMARKING PIPELINE")
    print("="*70)
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}, Batches/Epoch: {args.batches_per_epoch}")
    print(f"  Total Samples: {args.epochs * args.batches_per_epoch * args.batch_size}")
    print(f"Benchmark Configuration:")
    print(f"  Samples: {args.samples}, Image Size: {args.height}x{args.width}")
    print(f"  Noise Level: {args.noise_level}, Repeats: {args.repeats}")
    print("="*70 + "\n")

    print("► Starting TRAINING phase (classical filters + deep model)...")
    simple_train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    print("\n► Starting BENCHMARKING phase (CPU vs GPU comparison)...")
    benchmark_args = argparse.Namespace(
        samples=args.samples,
        height=args.height,
        width=args.width,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
        kernel_size=args.kernel_size,
        sigma=args.sigma,
        repeats=args.repeats,
        warmup=args.warmup,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        seed=args.seed,
        cpu_only=args.cpu_only,
    )
    run_benchmark(benchmark_args)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print(f"✓ Trained model: model_weights.pth")
    print(f"✓ Metrics: {args.output_dir}/training_metrics.json")
    print(f"✓ Benchmarks: {args.output_dir}/benchmark_summary.csv")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
