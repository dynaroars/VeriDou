from beartype import beartype
import onnx
import torch
import os
import argparse
from typing import List
import numpy as np

from utils.network.create_onnx_independent import create_motion_blur_kernel
from utils.spec.write_vnnlib import write_vnnlib
from utils.spec.objective import parse_vnnlib

from utils.network.create_onnx_veridou import create_onnx
from utils.network.read_onnx import parse_onnx, inference_onnx


def create_motion_blur_kernel_range(angle_min: float, angle_max: float, kernel_size: int, angle_step: float = 1.0):
    """
    Create a motion blur kernel that represents the union (support) of all line kernels
    for angles in [angle_min, angle_max], normalized to sum to 1.

    Args:
        angle_min (float): Minimum angle in degrees
        angle_max (float): Maximum angle in degrees
        kernel_size (int): Odd kernel size
        angle_step (float): Angular sampling step in degrees (smaller = denser sampling)

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: (kernel, support_mask)
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    if angle_min > angle_max:
        angle_min, angle_max = angle_max, angle_min

    support_mask = np.zeros((kernel_size, kernel_size), dtype=bool)

    theta = angle_min
    # Ensure we include both endpoints
    while theta <= angle_max + 1e-9:
        line_kernel = create_motion_blur_kernel(theta, kernel_size)
        support_mask |= line_kernel > 0
        theta += angle_step

    kernel = support_mask.astype(float)
    kernel = kernel / kernel_size
    return kernel


@beartype
def generate_single(
    spec_id: int,
    output_dir: str,
    onnx_model: onnx.ModelProto,
    image: torch.Tensor,
    kernel_type: str,
    strength: List[float],
    perturb_ratio: float = 0.1,
    robust_interval: float = 0.03,
    timeout: float = 30.0,
    benchmark_name: str = "mnist_fc",
    random_seed: int = 42,
    visualize: bool = False,
    kernel_size: int = 5,
    kernel: str = "0-30",
    mask_probability: float = 0.5,
):
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)

    onnx_name = f"{kernel_type}_{spec_id}.onnx"
    output_onnx_path = os.path.join(output_dir, "onnx", onnx_name)
    spec_name = f"{kernel_type}_{robust_interval}_{perturb_ratio}_{spec_id}.vnnlib"
    output_vnnlib_path = os.path.join(output_dir, "vnnlib", spec_name)
    if os.path.exists(output_onnx_path):
        return "", "", ""
    print(f"{output_onnx_path=}")
    print(f"{output_vnnlib_path=}")
    # create onnx model
    create_onnx(
        spec_id=spec_id,
        onnx_model=onnx_model,
        output_path=output_onnx_path,
        image=image,
        kernel_type=kernel_type,
        strength=strength,
        perturb_ratio=perturb_ratio,
        robust_interval=robust_interval,
        visualize=visualize,
        benchmark_name=benchmark_name,
        random_seed=random_seed,
        kernel_size=kernel_size,
    )
    mask = torch.rand_like(image.flatten()) < perturb_ratio
    noise = torch.zeros_like(image.flatten())
    noise[mask] = torch.ones_like(noise[mask]) * robust_interval
    if kernel_type == "fixed":
        angle_min, angle_max = map(float, kernel.split("-"))
        kernel_arr = create_motion_blur_kernel_range(angle_min, angle_max, kernel_size)
        kernel_arr[kernel_arr != 0] = 1 / kernel_size
        kernel_arr[kernel_size // 2, kernel_size // 2] = 1 / kernel_size - 1

        # Ensure bounds are consistent: if upper bound is negative, swap with lower bound
        kernel_lb = torch.zeros(kernel_size**2)
        kernel_ub = torch.tensor(kernel_arr.flatten())
        kernel_lb[kernel_ub != 0] = 1 / 10

        # Find indices where upper bound is negative
        negative_indices = kernel_ub < 0
        if negative_indices.any():
            # Swap bounds for negative values
            kernel_lb[negative_indices] = kernel_ub[negative_indices]
            kernel_ub[negative_indices] = 0.0
        kernel_ub[kernel_size * kernel_size // 2] = strength[1]
        kernel_lb[kernel_size * kernel_size // 2] = strength[0]

        x_lb = torch.tensor([kernel_lb.tolist() + (-noise).flatten().tolist()])
        x_ub = torch.tensor([kernel_ub.tolist() + noise.flatten().tolist()])
    else:
        new_weight = torch.zeros((kernel_size, kernel_size))
        new_weight = torch.nn.init.kaiming_normal_(new_weight, nonlinearity="conv2d")
        mask = (torch.rand_like(new_weight) < mask_probability).float()  # probability of keeping

        # Apply mask
        new_weight = new_weight * mask
        lower_z = new_weight + strength[0]
        upper_z = new_weight + strength[1]
        x_lb = torch.tensor([lower_z.flatten().tolist() + (-noise).flatten().tolist()])
        x_ub = torch.tensor([upper_z.flatten().tolist() + noise.flatten().tolist()])
    prediction = torch.from_numpy(
        inference_onnx(
            output_onnx_path,
            np.array([[0.0] * (kernel_size**2) + np.zeros(noise.shape[0]).tolist()]).astype(np.float32),
        )[0]
    )
    print(f"{x_lb.shape=}")
    print(f"{x_ub.shape=}")
    print(f"{prediction.shape=}")
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
    kernel_type: str,
    strength: List[float],
    perturb_ratio: float,
    robust_interval: float,
    benchmark_name: str,
    random_seed: int = 42,
    timeout: float = 30.0,
    visualize: bool = False,
    kernel_size: int = 5,
    kernel: str = "0-30",
    num_instances: int = 20,
    mask_probability: float = 0.5,
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
            onnx_name = f"{kernel_type}_{idx}.onnx"
            output_onnx_path = os.path.join(output_dir, "onnx", onnx_name)
            if os.path.exists(output_onnx_path):
                continue
            onnx_model, input_shape, output_shape = parse_onnx(onnx_path)
            vnnlib = parse_vnnlib(vnnlib_path, input_shape).pop(1)
            image = ((vnnlib.lower_bounds + vnnlib.upper_bounds) / 2).view(input_shape)
            line = generate_single(
                spec_id=idx,
                output_dir=output_dir,
                onnx_model=onnx_model,
                image=image,
                kernel_type=kernel_type,
                strength=strength,
                perturb_ratio=perturb_ratio,
                robust_interval=robust_interval,
                timeout=float(timeout),
                benchmark_name=benchmark_name,
                random_seed=random_seed + idx,  # Use different seed for each instance
                visualize=visualize,
                kernel_size=kernel_size,
                kernel=kernel,
                mask_probability=mask_probability,
            )
            if line != ("", "", ""):
                # print(f'{line=}')
                print(line, file=fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new benchmarks")
    parser.add_argument(
        "--kernel_types",
        nargs="+",
        type=str,
        default=["general"],
        # choices=["veridou", "general"],
        help="List of kernel types (veridou or general)",
    )
    parser.add_argument("--kernel_sizes", nargs="+", type=int, default=[5, 7, 9], help="List of kernel sizes")
    parser.add_argument("--kernels", nargs="+", type=str, default=["0-30", "30-60", "60-90"], help="List of kernels")
    parser.add_argument(
        "--strength", nargs=2, type=float, default=[-0.001, 0.001], help="List of 2 lower and upper strengths"
    )
    parser.add_argument(
        "--perturb_ratios", nargs="+", type=float, default=[0.0, 0.5, 1.0], help="List of perturb ratios"
    )
    parser.add_argument("--robust_intervals", nargs="+", type=float, default=[0.005], help="List of robust intervals")
    parser.add_argument(
        "--benchmark_names",
        nargs="+",
        default=["mnist_fc", "oval21", "sri_resnet_a", "cifar100", "tinyimagenet"],
        help="List of benchmark names",
    )
    parser.add_argument("--base_random_seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument("--output_dir", default="./generated_benchmarks_new", help="Output directory")
    parser.add_argument("--benchmarks_dir", default="./benchmarks", help="Benchmarks directory")
    parser.add_argument("--visualize", action="store_true", help="Visualize the generated benchmarks")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout for the generated benchmarks")
    parser.add_argument("--num_instances", type=int, default=20, help="Number of instances to process from each CSV")
    parser.add_argument("--mask_probability", type=float, default=0.5, help="Probability threshold for mask generation")
    args = parser.parse_args()

    kernel_types = args.kernel_types
    strength = args.strength
    perturb_ratios = args.perturb_ratios
    robust_intervals = args.robust_intervals
    benchmark_names = args.benchmark_names
    base_random_seed = args.base_random_seed
    output_dir = args.output_dir
    benchmarks_dir = args.benchmarks_dir
    visualize = args.visualize
    timeout = args.timeout
    kernel_sizes = args.kernel_sizes
    kernels = args.kernels
    mask_probability = args.mask_probability

    seed_offset = 0
    for benchmark_name in benchmark_names:
        for kernel_type in kernel_types:
            for kernel_size in kernel_sizes:
                for kernel in kernels:
                    for perturb_ratio in perturb_ratios:
                        for robust_interval in robust_intervals:
                            os.makedirs(
                                f"{output_dir}/{benchmark_name}/{kernel_type}/{kernel_size}/{kernel}/{perturb_ratio}/{robust_interval}",
                                exist_ok=True,
                            )
                            print(
                                f"{output_dir}/{benchmark_name}/{kernel_type}/{kernel_size}/{kernel}/{perturb_ratio}/{robust_interval}"
                            )
                            generate_benchmarks(
                                instances_path=f"{benchmarks_dir}/{benchmark_name}/instances.csv",
                                output_dir=f"{output_dir}/{benchmark_name}/{kernel_type}/{kernel_size}/{kernel}/{perturb_ratio}/{robust_interval}",
                                kernel_type=kernel_type,
                                strength=strength,
                                perturb_ratio=perturb_ratio,
                                robust_interval=robust_interval,
                                benchmark_name=benchmark_name,
                                random_seed=base_random_seed + seed_offset,
                                timeout=timeout,
                                kernel_size=kernel_size,
                                kernel=kernel,
                                num_instances=args.num_instances,
                                mask_probability=mask_probability,
                            )
                            seed_offset += 1000
