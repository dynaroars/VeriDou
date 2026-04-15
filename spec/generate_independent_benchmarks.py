from beartype import beartype
import onnx
import torch
import os
import argparse

from utils.spec.write_vnnlib import write_vnnlib
from utils.spec.objective import parse_vnnlib

from utils.network.create_onnx_aaai import create_onnx
from utils.network.read_onnx import parse_onnx, inference_onnx


@beartype
def generate_single(
    idx: int,
    output_dir: str,
    onnx_model: onnx.ModelProto,
    image: torch.Tensor,
    strength: float,
    kernel_size: int,
    perturbation_type: str,
    timeout: float,
    benchmark_name: str,
    visualize: bool,
):
    # create onnx model
    onnx_name = f"{perturbation_type}_{kernel_size}_{idx}.onnx"
    output_onnx_path = os.path.join(output_dir, "onnx", onnx_name)
    if not os.path.exists(output_onnx_path):
        create_onnx(
            onnx_model=onnx_model,
            output_path=output_onnx_path,
            image=image,
            kernel_size=kernel_size,
            pertubation=perturbation_type,
            visualize=visualize,
            benchmark_name=benchmark_name,
            spec_id=idx,
        )

    # create vnnlib spec
    spec_name = f"{perturbation_type}_{kernel_size}_{idx}_{strength}.vnnlib"
    output_vnnlib_path = os.path.join(output_dir, "vnnlib", spec_name)
    x_lb = torch.tensor([[0.0]])
    x_ub = torch.tensor([[strength]])
    prediction = torch.from_numpy(inference_onnx(output_onnx_path, x_lb.numpy())[0])
    write_vnnlib(
        spec_path=output_vnnlib_path,
        data_lb=x_lb,
        data_ub=x_ub,
        prediction=prediction,
    )
    return f"onnx/{onnx_name},vnnlib/{spec_name},{timeout}"


@beartype
def generate_benchmarks(
    instances_path: str,
    output_dir: str,
    strength: float,
    kernel_size: int,
    perturbation_type: str,
    benchmark_name: str,
    timeout: float,
    num_instances: int,
    visualize: bool,
):

    benchmark_dir = os.path.dirname(instances_path)
    os.makedirs(os.path.join(output_dir, "onnx"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "vnnlib"), exist_ok=True)
    instances = open(instances_path).readlines()
    with open(os.path.join(output_dir, "instances.csv"), "a") as fp:
        for idx, line in enumerate(instances[:num_instances]):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            onnx_file, vnnlib_file, _ = line.split(",")
            onnx_path = os.path.join(benchmark_dir, onnx_file)
            vnnlib_path = os.path.join(benchmark_dir, vnnlib_file)
            assert os.path.exists(onnx_path)
            assert os.path.exists(vnnlib_path)
            # print(idx, onnx_path, vnnlib_path)

            onnx_model, input_shape, output_shape = parse_onnx(onnx_path)
            vnnlib = parse_vnnlib(vnnlib_path, input_shape).pop(1)
            image = ((vnnlib.lower_bounds + vnnlib.upper_bounds) / 2).view(input_shape)
            line = generate_single(
                idx=idx,
                output_dir=output_dir,
                onnx_model=onnx_model,
                image=image,
                strength=strength,
                kernel_size=kernel_size,
                perturbation_type=perturbation_type,
                timeout=timeout,
                benchmark_name=benchmark_name,
                visualize=visualize,
            )
            print(line, file=fp)
        return os.path.join(output_dir, "instances.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate AAAI Benchmarks")
    parser.add_argument(
        "--perturbation_types",
        nargs="+",
        default=[
            "motion_blur_0",
            "motion_blur_15",
            "motion_blur_30",
            "motion_blur_45",
            "motion_blur_60",
            "motion_blur_75",
            "motion_blur_90",
        ],
        help="List of perturbation types",
    )
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.2, 0.5, 1.0], help="List of strengths")
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[5, 7, 9], help="List of kernel sizes")
    parser.add_argument(
        "--benchmark_names", nargs="+", default=["mnist_fc", "oval21", "sri_resnet_a", "cifar100", "tinyimagenet"], help="List of benchmark names"
    )
    parser.add_argument("--benchmarks_dir", default="./benchmarks", help="Benchmarks directory")
    parser.add_argument("--output_dir", default="./generated_benchmarks", help="Output directory")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout value for each instance")
    parser.add_argument("--num_instances", type=int, default=10, help="Number of instances to process from each CSV")
    parser.add_argument(
        "--visualize", type=int, default=0, choices=[0, 1], help="Enable (1) or disable (0) visualization (default: 1)"
    )
    args = parser.parse_args()

    # Convert visualize to bool
    visualize = bool(args.visualize)

    perturbation_types = args.perturbation_types
    strengths = args.strengths
    kernel_sizes = args.kernel_sizes
    benchmark_names = args.benchmark_names
    benchmarks_dir = args.benchmarks_dir
    output_dir = args.output_dir

    benchmark_paths = []
    from itertools import product
    from multiprocessing import Pool

    # Create all parameter combinations
    param_combinations = list(product(benchmark_names, kernel_sizes, perturbation_types, strengths))

    def process_combination(params):
        benchmark_name, kernel_size, perturbation_type, strength = params
        output_path = f"{output_dir}/{benchmark_name}/{perturbation_type}/{kernel_size}/{strength}"
        os.makedirs(output_path, exist_ok=True)
        return generate_benchmarks(
            instances_path=f"{benchmarks_dir}/{benchmark_name}/instances.csv",
            output_dir=output_path,
            strength=strength,
            kernel_size=kernel_size,
            perturbation_type=perturbation_type,
            benchmark_name=benchmark_name,
            timeout=args.timeout,
            num_instances=args.num_instances,
            visualize=visualize,
        )

    with Pool(processes=8) as pool:
        benchmark_paths = pool.map(process_combination, param_combinations)

    print(benchmark_paths)
