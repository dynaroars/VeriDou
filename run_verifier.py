
import csv
import sys
import subprocess
import argparse
import multiprocessing as mp
import time
from pathlib import Path


def run_verifier_on_csv(verifier_path, result_dir, benchmark_folder):
    """
    Run venus2 verifier on all instances in the CSV file.

    Args:
        csv_file: Path to the instances.csv file
        verifier_path: Path to verifier directory
        results_dir: Directory to store results (optional)
    """
    BATCH_SIZE = 8  # Process 8 instances at a time
    
    # Known benchmark names
    KNOWN_BENCHMARKS = ["mnist_fc", "oval21", "sri_resnet_a", "cifar100", "tinyimagenet"]
    
    for csv_path in Path(benchmark_folder).rglob("*.csv"):
        benchmark_dir = csv_path.parent.resolve()
        verifier_path = Path(verifier_path).resolve()
        
        # Extract benchmark name from path by checking known benchmarks
        dataname = None
        for benchmark in KNOWN_BENCHMARKS:
            if benchmark in csv_path.parts:
                dataname = benchmark
                break
        results_dir = Path(result_dir) / csv_path.relative_to(benchmark_folder).parent
        results_dir.mkdir(parents=True, exist_ok=True)

        # Count total instances first
        with open(csv_path, "r") as f:
            total_instances = sum(1 for line in f)

        print(f"Found {total_instances} instances to verify")
        print(f"Benchmark directory: {benchmark_dir}")
        print(f"Verifier path: {verifier_path}")

        # Process each instance
        instance_args = []
        with open(csv_path, "r") as f:
            csv_reader = csv.reader(f)

            for i, row in enumerate(csv_reader):
                if len(row) != 3:
                    print(f"Skipping malformed row {i+1}: {row}")
                    continue
                if Path(f"{results_dir}/log_{i+1}.txt").exists():
                    print(f"Skipping already verified instance {i+1}")
                    continue
                onnx_file, vnnlib_file, timeout = row

                # Construct absolute paths
                onnx_path = (benchmark_dir / onnx_file).resolve()
                vnnlib_path = (benchmark_dir / vnnlib_file).resolve()

                # Check if files exist
                if not onnx_path.exists():
                    print(f"ERROR: ONNX file not found: {onnx_path}")
                    continue

                if not vnnlib_path.exists():
                    print(f"ERROR: VNNLib file not found: {vnnlib_path}")
                    continue

                print(f"\n[{i+1}/{total_instances}] Processing: {onnx_file} | {vnnlib_file}")
                # if "motion_blur_0" not in str(Path(csv_path).resolve()):
                    # continue
                log_file = results_dir / f"log_{i+1}.txt"

                if "alpha-beta-CROWN" in str(verifier_path):
                    if "mnist_fc" in str(Path(csv_path).resolve()):
                        if "/5/" in str(Path(csv_path).resolve()):
                            dataname = "mnistfc_fixed_kernel_5"
                        elif "/7/" in str(Path(csv_path).resolve()):
                            dataname = "mnistfc_fixed_kernel_7"
                        elif "/9/" in str(Path(csv_path).resolve()):
                            dataname = "mnistfc_fixed_kernel_9"
                        else:
                            dataname = "mnistfc"
                    
                    cmd = [
                        sys.executable,
                        str(verifier_path / "complete_verifier" / "vnncomp_main.py"),
                        dataname,
                        str(onnx_path),
                        str(vnnlib_path),
                    ]
                    if results_dir:
                        cmd.extend([str(log_file.resolve())])
                    cmd.extend([str(30)])
                elif "neuralsat" in str(verifier_path):
                    cmd = [
                        sys.executable,
                        str(verifier_path / "src/main.py"),
                        "--net",
                        str(onnx_path),
                        "--spec",
                        str(vnnlib_path),
                        "--timeout",
                        str(timeout),
                        "--export_cex"
                    ]
                    if results_dir:
                        cmd.extend(["--result_file", str(log_file.resolve())])
                elif "venus" in str(verifier_path):
                    cmd = [
                        sys.executable,
                        str(verifier_path / "__main__.py"),
                        "--net",
                        str(onnx_path),
                        "--spec",
                        str(vnnlib_path),
                        "--timeout",
                        str(timeout),
                    ]
                    if results_dir:
                        cmd.extend(["--logfile", str(log_file.resolve())])
                else:
                    raise Exception("Unknown verifier")

                print(f"Running command: {cmd}")
                try:
                    # Run venus2 from its own directory
                    start_time = time.perf_counter()
                    result = subprocess.run(
                        cmd,
                        cwd=verifier_path,
                        capture_output=True,
                        text=True
                    )
                    elapsed_s = time.perf_counter() - start_time

                    print(f"Return code: {result.returncode}")
                    if result.stdout:
                        print(f"STDOUT: {result.stdout}")
                    if result.stderr:
                        print(f"STDERR: {result.stderr}")

                    if results_dir:
                        runner_meta = (
                            "\n\n=== runner metadata ===\n"
                            f"RUNNER_RUNTIME_SECONDS: {elapsed_s:.6f}\n"
                            f"RUNNER_RETURN_CODE: {result.returncode}\n"
                        )
                        if log_file.exists():
                            with open(log_file, "a") as f:
                                f.write(runner_meta)
                        else:
                            # Verifier did not create the log file; write a minimal one.
                            with open(log_file, "w") as f:
                                f.write(runner_meta)
                                if result.stdout:
                                    f.write("\n--- captured stdout ---\n")
                                    f.write(result.stdout)
                                    if not result.stdout.endswith("\n"):
                                        f.write("\n")
                                if result.stderr:
                                    f.write("\n--- captured stderr ---\n")
                                    f.write(result.stderr)
                                    if not result.stderr.endswith("\n"):
                                        f.write("\n")

                except subprocess.TimeoutExpired:
                    print(f"Instance {i+1} timed out after {timeout} seconds")

                except Exception as e:
                    print(f"Error running instance {i+1}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run venus2 verifier on CSV instances")
    parser.add_argument(
        "--benchmark_folder",
        default="generated_benchmarks_new",
        help="Path to benchmark folder contains all benchmark to verify, following VNNCOMP format",
    )
    parser.add_argument("--verifier_path", default="/absolute/path/to/verifier/", help="Absolute path to verifier directory")
    parser.add_argument("--result_dir", default="result_crown_new", help="Directory to store results")

    args = parser.parse_args()

    run_verifier_on_csv(args.verifier_path, args.result_dir, args.benchmark_folder)


if __name__ == "__main__":
    main()
