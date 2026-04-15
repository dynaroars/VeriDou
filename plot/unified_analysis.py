#!/usr/bin/env python3
"""
Unified script that combines log parsing, image processing, and perceptual metrics calculation.
This script extends summarize_results.py to include perceptual metrics in the CSV output.
"""

import os
import re
import csv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional, Tuple, List
from utils.network.create_onnx_veridou import create_motion_blur_kernel_range
from collections import defaultdict
import argparse
import warnings

warnings.filterwarnings("ignore")

# # Import from existing visualization script
# try:
#     from utils.network.create_onnx_veridou import create_motion_blur_kernel_range
# except ImportError:
#     print("Warning: Could not import create_motion_blur_kernel_range. Some functionality may be limited.")


def create_default_stats():
    """Factory function to create default statistics dictionary."""
    return {"safe": 0, "unsafe": 0, "undecided": 0, "timeouts": 0}


def defaultdict_to_dict(d):
    """Convert nested defaultdict to regular dict for better printing."""
    if isinstance(d, defaultdict):
        return {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


def load_image_as_tensor(image_path):
    """Load image and convert to tensor for LPIPS calculation."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # LPIPS expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transform(image).unsqueeze(0)


def calculate_ssim_psnr(origin_path, perturbation_path):
    """Calculate SSIM and PSNR between two images."""
    try:
        # Load images as grayscale for SSIM/PSNR
        origin = np.array(Image.open(origin_path).convert("L"))
        perturbation = np.array(Image.open(perturbation_path).convert("L"))

        # Calculate SSIM
        ssim_value = ssim(origin, perturbation, data_range=255)

        # Calculate PSNR
        psnr_value = psnr(origin, perturbation, data_range=255)

        return ssim_value, psnr_value
    except Exception as e:
        print(f"Error calculating SSIM/PSNR for {origin_path}: {e}")
        return None, None


def calculate_lpips(origin_path, perturbation_path, lpips_model):
    """Calculate LPIPS between two images."""
    try:
        origin_tensor = load_image_as_tensor(origin_path)
        perturbation_tensor = load_image_as_tensor(perturbation_path)

        with torch.no_grad():
            lpips_value = lpips_model(origin_tensor, perturbation_tensor)

        return lpips_value.item()
    except Exception as e:
        print(f"Error calculating LPIPS for {origin_path}: {e}")
        return None


def load_image_from_vnnlib(vnnlib_path: str) -> torch.Tensor:
    """
    Load image from vnnlib file by extracting the center point.

    Args:
        vnnlib_path: Path to the vnnlib file

    Returns:
        Image tensor in BCHW format
    """
    try:
        with open(vnnlib_path, "r") as f:
            content = f.read()
        # Extract input bounds using regex
        # Look for patterns like (assert (<= X_0 upper)) and (assert (>= X_0 lower))
        upper_bounds_pattern = r"\(assert\s+\(<=\s+X_(\d+)\s+([\d.-]+)\)\)"
        lower_bounds_pattern = r"\(assert\s+\(>=\s+X_(\d+)\s+([\d.-]+)\)\)"

        upper_matches = re.findall(upper_bounds_pattern, content)
        lower_matches = re.findall(lower_bounds_pattern, content)

        if not upper_matches or not lower_matches:
            raise ValueError("Could not find input bounds in vnnlib file")

        # Create dictionaries for upper and lower bounds
        upper_bounds = {int(idx): float(val) for idx, val in upper_matches}
        lower_bounds = {int(idx): float(val) for idx, val in lower_matches}

        # Combine into bounds dictionary
        bounds = {}
        for idx in upper_bounds:
            if idx in lower_bounds:
                bounds[idx] = (lower_bounds[idx], upper_bounds[idx])

        # Get the center point (average of lower and upper bounds)
        values = []
        for i in range(len(bounds)):
            if i in bounds:
                lower, upper = bounds[i]
                center = (lower + upper) / 2
                values.append(center)

        # Convert to tensor and reshape (assuming 3-channel image)
        values = np.array(values)

        # Try to determine image dimensions
        if len(values) == 3072:  # 32x32x3
            h, w, c = 32, 32, 3
        elif len(values) == 784:  # 28x28x1 (MNIST)
            h, w, c = 28, 28, 1
        elif len(values) == 12288:  # 64x64x3
            h, w, c = 64, 64, 3
        else:
            # Default to square image
            total_pixels = len(values)
            if total_pixels % 3 == 0:
                pixels_per_channel = total_pixels // 3
                side_length = int(np.sqrt(pixels_per_channel))
                h, w, c = side_length, side_length, 3
            else:
                side_length = int(np.sqrt(total_pixels))
                h, w, c = side_length, side_length, 1

        # Reshape and convert to tensor
        image = values.reshape(c, h, w)
        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        return image_tensor

    except Exception as e:
        print(f"Error loading image from vnnlib {vnnlib_path}: {e}")
        return None


def extract_single_z_from_log(log_file_path: str, mode: str = "both") -> tuple:
    """
    Extract single z value from log file for UNSAT and/or SAT results.
    - For UNSAT results: use default z value of 1.0
    - For SAT results: extract first z value from counterexample

    Args:
        log_file_path: Path to the log file
        mode: Processing mode - "sat", "unsat", or "both"

    Returns:
        Tuple of (z_value, result_type) or (None, None) if parsing fails
        result_type is either 'unsat' or 'sat'
    """
    try:
        with open(log_file_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return None, None

        # Check result type
        content = "".join(lines)
        has_unsat = any(line.strip().startswith("unsat") for line in lines)
        has_sat = any(line.strip().startswith("sat") for line in lines)
        has_unsafe = "Result: unsafe" in content
        has_timeout = any(line.strip().startswith("timeout") for line in lines) or "Result: timeout" in content

        # For timeout results, return None
        if has_timeout:
            return None, None

        if has_unsat and mode in ["unsat", "both"]:
            # For UNSAT results, use default z value of 1.0
            return (1.0, "unsat")
        elif (has_sat or has_unsafe) and mode in ["sat", "both"]:
            # For SAT/UNSAFE results, extract first z value from counterexample
            z_value = None

            if has_unsafe:
                # Pattern: Venus unsafe format - extract from Counter-example line
                for line in lines:
                    if "Counter-example:" in line:
                        # Extract first tensor value from the line
                        # Format: Counter-example: [tensor(0.), tensor(0.), tensor(0.2000), ...]
                        tensor_matches = re.findall(r"tensor\(([^)]+)\)", line)
                        if tensor_matches:
                            z_value = float(tensor_matches[0])  # Take first value
                        break
            else:
                # Patterns: SAT formats - take first value found
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue

                    # Pattern 1: CROWN format (X_i value)
                    match = re.search(r"\(X_(\d+)\s+([\d.-]+)\)", line)
                    if match:
                        z_value = float(match.group(2))
                        break

                    # Pattern 2: NeuralSAT format ((X_i value))
                    match = re.search(r"\(\(X_(\d+)\s+([\d.-]+)\)\)", line)
                    if match:
                        z_value = float(match.group(2))
                        break

            if z_value is not None:
                return (z_value, "sat")

        return None, None

    except Exception as e:
        print(f"Error parsing log file {log_file_path}: {e}")
        return None, None


def extract_unsat_values_from_spec(spec_file_path: str, kernel_size: int) -> Tuple[List[List[float]], str, List[float]]:
    """
    Extract kernel and perturbation values from spec file for UNSAT results.
    Reads the vnnlib file to get the ranges and samples values from those ranges.

    Args:
        spec_file_path: Path to the vnnlib spec file
        kernel_size: Size of the kernel (e.g., 5 for 5x5)

    Returns:
        Tuple of (kernel_matrix, 'unsat', perturbation_values)
    """
    with open(spec_file_path, "r") as f:
        content = f.read()

    # Extract input bounds using regex
    # Look for patterns like (assert (<= X_0 upper)) and (assert (>= X_0 lower))
    upper_bounds_pattern = r"\(assert\s+\(<=\s+X_(\d+)\s+([\d.-]+)\)\)"
    lower_bounds_pattern = r"\(assert\s+\(>=\s+X_(\d+)\s+([\d.-]+)\)\)"

    upper_matches = re.findall(upper_bounds_pattern, content)
    lower_matches = re.findall(lower_bounds_pattern, content)

    if not upper_matches or not lower_matches:
        raise ValueError(f"Warning: Could not find input bounds in spec file {spec_file_path}")

    # Create dictionaries for upper and lower bounds
    upper_bounds = {int(idx): float(val) for idx, val in upper_matches}
    lower_bounds = {int(idx): float(val) for idx, val in lower_matches}

    # Combine into bounds dictionary
    bounds = {}
    for idx in upper_bounds:
        if idx in lower_bounds:
            bounds[idx] = (lower_bounds[idx], upper_bounds[idx])

    if not bounds:
        raise ValueError(f"Warning: No valid bounds found in spec file {spec_file_path}")

    # Sample kernel values from the ranges
    import random

    expected_kernel_values = kernel_size * kernel_size
    kernel_values = []

    # Sample first k×k values for kernel from the available bounds
    for i in range(expected_kernel_values):
        if i in bounds:
            lower, upper = bounds[i]
            kernel_values.append(random.uniform(lower, upper))
        else:
            raise ValueError(f"Warning: No valid bounds found in spec file {spec_file_path}")

    # Create kernel matrix
    kernel = []
    for i in range(kernel_size):
        row = kernel_values[i * kernel_size : (i + 1) * kernel_size]
        kernel.append(row)

    # Sample perturbation values from remaining bounds
    perturbation_values = []
    for i in range(expected_kernel_values, len(bounds)):
        if i in bounds:
            lower, upper = bounds[i]
            perturbation_values.append(random.uniform(lower, upper))

    return (kernel, "unsat", perturbation_values)


def extract_kernel_from_log(
    log_file_path: str, spec_file_path: str, kernel_size: int, mode: str = "both"
) -> Optional[Tuple[List[List[float]], str, List[float]]]:
    """
    Extract kernel values from log file for both SAT and UNSAT results.
    - For UNSAT results: sample kernel and perturbation values from spec file ranges
    - For SAT results: extract k×k kernel values + perturbation values from counterexample

    The log file contains:
    - First k×k values: kernel values
    - Remaining values: perturbation values to add on top of perturbed input

    Supports four formats:
    1. Standard: 'sat' followed by '(X_i value)'
    2. NeuralSAT: 'sat,time' followed by '((X_i value))'
    3. CROWN: 'sat' followed by '(X_i  value)' (double space)
    4. Venus unsafe: 'Result: unsafe' followed by 'Counter-example: [tensor(value), ...]'

    Args:
        log_file_path: Path to the log file
        spec_file_path: Path to the spec file (vnnlib)
        kernel_size: Size of the kernel (e.g., 5 for 5x5)
        mode: Processing mode - "sat", "unsat", or "both"

    Returns:
        Tuple of (kernel_matrix, result_type, perturbation_values) or None if parsing fails
        result_type is either 'unsat' or 'sat'
        perturbation_values is empty list for UNSAT, contains remaining values for SAT
    """
    try:
        with open(log_file_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        # Check result type
        content = "".join(lines)
        has_unsat = any(line.strip().startswith("unsat") for line in lines)
        has_sat = any(line.strip().startswith("sat") for line in lines)
        has_unsafe = "Result: unsafe" in content

        if has_unsat and mode in ["unsat", "both"]:
            # For UNSAT results, read from spec file to extract ranges and sample values
            return extract_unsat_values_from_spec(spec_file_path, kernel_size)
        elif (has_sat or has_unsafe) and mode in ["sat", "both"]:
            # For SAT/UNSAFE results, extract k×k kernel values from counterexample
            values = []

            if has_unsafe:
                # Pattern: Venus unsafe format - extract from Counter-example line
                for line in lines:
                    if "Counter-example:" in line:
                        # Extract tensor values from the line
                        # Format: Counter-example: [tensor(0.), tensor(0.), tensor(0.2000), ...]
                        tensor_matches = re.findall(r"tensor\(([^)]+)\)", line)
                        for match in tensor_matches:
                            value = float(match)
                            values.append(value)
                        break
            else:
                # Patterns: SAT formats
                for line in lines[1:]:
                    line = line.strip()
                    if not line:
                        continue

                    # Pattern 1: NeuralSAT format with double parentheses ((X_i value)
                    match = re.search(r"\(\(X_(\d+)\s+([^)]+)\)", line)
                    if match:
                        value = float(match.group(2))
                        values.append(value)
                        continue

                    # Pattern 2: Standard format (X_i value)
                    match = re.search(r"\(X_(\d+)\s+([^)]+)\)", line)
                    if match:
                        value = float(match.group(2))
                        values.append(value)
                        continue

                    # Pattern 3: Venus format tensor(value) - extract all from line
                    matches = re.findall(r"tensor\(([^)]+)\)", line)
                    if matches:
                        for match in matches:
                            value = float(match)
                            values.append(value)
                        continue

            # Check if we have enough values for the kernel
            expected_kernel_values = kernel_size * kernel_size
            if len(values) < expected_kernel_values:
                print(
                    f"Warning: Only found {len(values)} values, expected at least {expected_kernel_values} for kernel"
                )
                return None

            # Take the first k*k values for the kernel
            kernel_values = values[:expected_kernel_values]
            kernel = []
            for i in range(kernel_size):
                row = kernel_values[i * kernel_size : (i + 1) * kernel_size]
                kernel.append(row)

            # Store the remaining values as perturbation values
            perturbation_values = values[expected_kernel_values:] if len(values) > expected_kernel_values else []

            return (kernel, "sat", perturbation_values)

        return None

    except Exception as e:
        print(f"Error parsing log file {log_file_path}: {e}")
        return None


def apply_kernel_convolution(
    image: torch.Tensor, kernel: np.ndarray, perturbation_values: List[float] = None
) -> torch.Tensor:
    """
    Apply convolution with the extracted kernel using R*C perturbation.
    Following the pattern from create_onnx_veridou.py: F(X) = z @ A.T + B
    where A maps z inputs to conv outputs and B is the identity convolution.
    Additional perturbation values are added on top of the perturbed input.

    Args:
        image: Input image tensor in BCHW format
        kernel: Kernel matrix as numpy array (z values)
        perturbation_values: Additional perturbation values to add on top

    Returns:
        Convolved image tensor
    """
    # Ensure image is in BCHW format
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    b, c, h, w = image.shape
    kernel_size = kernel.shape[0]

    # Step 1: Create and apply B convolution filter (identity at center)
    B = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    B[kernel_size // 2, kernel_size // 2] = 1

    conv2d_B = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )
    conv2d_B.weight.data = B.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    if conv2d_B.bias is not None:
        conv2d_B.bias.data = torch.zeros(c)
    conv_result_B = conv2d_B(image)
    B_flat = conv_result_B.flatten().detach().numpy().astype(np.float32)

    # Step 2: Create A matrix (maps z inputs to conv outputs)
    conv2d_A = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )

    # Get the mapping matrix by applying unit kernels at each position
    A_matrix_parts = []
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Create kernel with 1 at position (i,j) and 0 elsewhere
            unit_kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
            unit_kernel[i, j] = 1
            conv2d_A.weight.data = unit_kernel.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
            if conv2d_A.bias is not None:
                conv2d_A.bias.data = torch.zeros(c)
            unit_result = conv2d_A(image)
            A_matrix_parts.append(unit_result.flatten().detach().numpy().astype(np.float32))

    # Stack to create the full A matrix: [output_size, num_z_inputs]
    A_matrix = np.column_stack(A_matrix_parts)

    # Step 3: Apply the formula F(X) = z @ A.T + B
    z_values = kernel.flatten()  # Flatten the kernel to z values

    # Apply the A matrix with z values: F(X) = z @ A.T + B
    fx_result = np.dot(z_values, A_matrix.T) + B_flat
    fx_tensor = torch.from_numpy(fx_result.astype(np.float32))

    # Reshape F(X) back to image dimensions
    result = fx_tensor.view(b, c, h, w)

    # Add additional perturbation values if provided
    if perturbation_values and len(perturbation_values) > 0:
        # Ensure we have enough perturbation values for the image
        expected_perturb_size = b * c * h * w
        if len(perturbation_values) >= expected_perturb_size:
            # Take the first expected_perturb_size values and reshape to match image
            pert_values = perturbation_values[:expected_perturb_size]
            pert_tensor = torch.from_numpy(np.array(pert_values, dtype=np.float32)).view(b, c, h, w)
            result = result + pert_tensor
        else:
            print(f"Warning: Only {len(perturbation_values)} perturbation values, expected {expected_perturb_size}")

    # Clamp to valid range
    result = result.clamp(0, 1)

    return result


def apply_single_z_convolution(
    image: torch.Tensor, motion_blur_value: float, z_value: float, kernel_size: int
) -> torch.Tensor:
    """
    Apply convolution with a single z value.
    Following the pattern from create_onnx_veridou.py: F(X) = z * A + B
    where A is the convolution result of a unit kernel and B is the identity convolution.

    Args:
        image: Input image tensor in BCHW format
        motion_blur_value: Single scalar motion blur value
        z_value: Single scalar z value
        kernel_size: Size of the kernel (e.g., 5 for 5x5)

    Returns:
        Convolved image tensor
    """
    # Ensure image is in BCHW format
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    b, c, h, w = image.shape

    # Step 1: Create and apply B convolution filter (identity at center)
    B = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    B[kernel_size // 2, kernel_size // 2] = 1

    conv2d_B = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )
    conv2d_B.weight.data = B.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    if conv2d_B.bias is not None:
        conv2d_B.bias.data = torch.zeros(c)
    conv_result_B = conv2d_B(image)

    # Step 2: Create A matrix (maps z input to conv outputs)
    conv2d_A = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )

    try:
        kernel = torch.from_numpy(
            create_motion_blur_kernel_range(motion_blur_value, motion_blur_value, kernel_size)
        ).to(dtype=image.dtype)
        kernel[kernel_size // 2, kernel_size // 2] = 1 / kernel_size - 1
        conv2d_A.weight.data = kernel.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
        if conv2d_A.bias is not None:
            conv2d_A.bias.data = torch.zeros(c)
        unit_result = conv2d_A(image)

        # Step 3: Apply the formula F(X) = z * A + B
        A_flat = unit_result.flatten().detach().numpy().astype(np.float32)
        B_flat = conv_result_B.flatten().detach().numpy().astype(np.float32)

        # Apply the formula: F(X) = z * A + B
        fx_result = z_value * A_flat + B_flat
        fx_tensor = torch.from_numpy(fx_result.astype(np.float32))

        # Reshape F(X) back to image dimensions
        result = fx_tensor.view(b, c, h, w)

        # Clamp to valid range
        result = result.clamp(0, 1)

        return result
    except Exception as e:
        print(f"Error in convolution: {e}")
        # Return original image if convolution fails
        return image


def find_spec_file(log_file_path, spec_dir, benchmark_dir):
    """
    Find the corresponding spec file (vnnlib) for a given log file.

    Args:
        log_file_path: Path to the log file
        spec_dir: Base spec directory
        benchmark_dir: Benchmark directory

    Returns:
        Path to the spec file or None if not found
    """
    try:
        # Extract log number from filename
        log_filename = os.path.basename(log_file_path)
        log_number = int(log_filename.split("_")[1].split(".")[0])

        # Extract benchmark type from benchmark_dir
        benchmark_type = os.path.basename(benchmark_dir)

        # Construct spec file path based on log file path structure
        # This mirrors the logic from visualize_log_kernels.py
        log_path_str = str(log_file_path)

        # Extract metadata from log file path (excluding the log filename)
        path_parts = Path(log_file_path).parts
        metadata = ""
        for i, part in enumerate(path_parts):
            if part == benchmark_type:
                # Take everything after the benchmark type, but exclude the log filename
                metadata_parts = path_parts[i + 1 : -1]  # Exclude the last part (log filename)
                metadata = "/".join(metadata_parts)
                break

        # Construct spec file path
        spec_file_path = os.path.join(spec_dir, benchmark_type, metadata, "vnnlib")

        # Find the spec file with matching log number
        if os.path.exists(spec_file_path):
            spec_files = [f for f in os.listdir(spec_file_path) if f.endswith(".vnnlib")]
            spec_files.sort()  # Sort to ensure consistent ordering

            if log_number <= len(spec_files):
                return os.path.join(spec_file_path, spec_files[log_number - 1])

        return None

    except Exception as e:
        print(f"Error finding spec file for {log_file_path}: {e}")
        return None


def generate_image_pairs_from_log(
    log_file_path, instances_csv_path, benchmark_dir, temp_dir, use_full_kernel=False, spec_dir=None
):
    """
    Generate origin and perturbation images from log file and vnnlib file.

    Args:
        log_file_path: Path to the log file
        instances_csv_path: Path to the instances.csv file
        benchmark_dir: Benchmark directory
        temp_dir: Temporary directory to save generated images
        use_full_kernel: If True, use full kernel matrix extraction; if False, use single z value

    Returns:
        Tuple of (origin_path, perturbation_path) or (None, None) if generation fails
    """
    try:
        # Get log file number (e.g., log_1.txt -> 1)
        log_filename = os.path.basename(log_file_path)
        log_number = int(log_filename.split("_")[1].split(".")[0])

        # Read instances.csv to get the corresponding vnnlib file
        with open(instances_csv_path, "r") as f:
            lines = f.readlines()

        if log_number > len(lines):
            print(f"Log number {log_number} exceeds number of instances ({len(lines)})")
            return None, None

        # Get the line corresponding to this log file (0-indexed)
        instance_line = lines[log_number - 1].strip()
        onnx_file, vnnlib_file, timeout = instance_line.split(",")

        # Construct full paths
        benchmark_dir = os.path.dirname(instances_csv_path)
        vnnlib_path = os.path.join(benchmark_dir, vnnlib_file)

        if not os.path.exists(vnnlib_path):
            print(f"VNNLIB file not found: {vnnlib_path}")
            return None, None

        # Load original image from vnnlib
        original_image = load_image_from_vnnlib(vnnlib_path)
        if original_image is None:
            print(f"Failed to load image from {vnnlib_path}")
            return None, None

        # Extract kernel size from log file path
        path_parts = Path(log_file_path).parts
        kernel_size = None
        for part in path_parts:
            if part.isdigit():
                kernel_size = int(part)
                break

        if kernel_size is None:
            print(f"Could not determine kernel size from path: {log_file_path}")
            return None, None

        # Generate perturbed image based on method
        if use_full_kernel:
            # Use full kernel matrix extraction
            if spec_dir is None:
                print(f"Error: spec_dir is required when using --use_full_kernel")
                return None, None

            # Find the corresponding spec file
            spec_file_path = find_spec_file(log_file_path, spec_dir, benchmark_dir)
            if spec_file_path is None:
                print(f"Warning: Could not find spec file for {log_file_path}")
                return None, None

            result = extract_kernel_from_log(log_file_path, spec_file_path, kernel_size, "both")
            if result is None:
                print(f"Warning: Could not extract kernel from {log_file_path}")
                return None, None

            kernel, result_type, perturbation_values = result
            kernel_array = np.array(kernel)
            perturbed_image = apply_kernel_convolution(original_image, kernel_array, perturbation_values)

            # Create filename with kernel info
            base_name = f"log_{log_number}_{result_type}_full_kernel_{kernel_size}x{kernel_size}"
        else:
            # Use single z value extraction
            import re

            match = re.search(r"motion_blur_(\d+\.?\d*)", str(log_file_path))
            if match:
                motion_blur_value = float(match.group(1))
            else:
                print(f"Warning: Could not find motion_blur value in log_file_path: {log_file_path}")
                return None, None

            result = extract_single_z_from_log(log_file_path, "both")
            if result is None or result[0] is None:
                print(f"Warning: Could not extract z value from {log_file_path} (timeout or parsing failed)")
                return None, None

            z_value, result_type = result
            perturbed_image = apply_single_z_convolution(original_image, motion_blur_value, z_value, kernel_size)

            # Create filename with motion blur info
            base_name = (
                f"log_{log_number}_{result_type}_motion_blur_{motion_blur_value}_kernel_{kernel_size}x{kernel_size}"
            )

        # Create temporary directory
        os.makedirs(temp_dir, exist_ok=True)

        # Save images
        origin_path = os.path.join(temp_dir, f"{base_name}_origin.png")
        perturbation_path = os.path.join(temp_dir, f"{base_name}_perturbation.png")

        # Convert tensors to PIL images and save
        # Original image
        orig_np = np.clip(original_image.squeeze(0).detach().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        if orig_np.shape[2] == 1:  # Grayscale
            orig_np = np.repeat(orig_np, 3, axis=2)
        orig_pil = Image.fromarray(orig_np)
        orig_pil.save(origin_path)

        # Perturbed image
        pert_np = np.clip(perturbed_image.squeeze(0).detach().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        if pert_np.shape[2] == 1:  # Grayscale
            pert_np = np.repeat(pert_np, 3, axis=2)
        pert_pil = Image.fromarray(pert_np)
        pert_pil.save(perturbation_path)

        return origin_path, perturbation_path

    except Exception as e:
        print(f"Error generating image pairs for {log_file_path}: {e}")
        return None, None


def find_log_directories(root_dir: str) -> List[str]:
    """
    Recursively find all directories containing log files.

    Args:
        root_dir: Root directory to search

    Returns:
        List of directories containing log files
    """
    log_dirs = []

    for root, dirs, files in os.walk(root_dir):
        # Check if this directory contains log files
        log_files = [f for f in files if f.startswith("log_") and f.endswith(".txt")]
        if log_files:
            log_dirs.append(root)

    return log_dirs


def process_single_log_file(
    result_file,
    benchmark,
    verifier_name,
    benchmark_dir,
    spec_dir,
    temp_dir,
    use_full_kernel,
    lpips_model,
    csv_data,
    results,
    mode="both",
):
    """
    Process a single log file and add results to csv_data and results.
    """
    print(f"Processing: {result_file}")
    is_sat, is_unsat, is_timeout = False, False, False
    runtime = None

    with open(result_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        # Extract runtime if available
        # Preferred: runner timing appended by run_verifier.py
        runner_rt = re.search(r"RUNNER_RUNTIME_SECONDS:\s*([0-9]*\.?[0-9]+)", content, re.IGNORECASE)
        if runner_rt:
            runtime = float(runner_rt.group(1))
        else:
            # Fallback: some tools print "sat,1.23" / "unsat,0.45" / "timeout,30.0"
            runtime_match = re.search(r"(sat|unsat|timeout),\s*([0-9]*\.?[0-9]+)", content, re.IGNORECASE)
            if runtime_match:
                runtime = float(runtime_match.group(2))

        if verifier_name == "venus":
            if "Result: unsafe" in content:
                is_sat = True
            elif "Result: safe" in content:
                is_unsat = True
            elif "Result: timeout" in content or "Result: unverified" in content:
                is_timeout = True
            else:
                is_timeout = True
        elif verifier_name == "neuralsat":
            if "unsat" in content.strip():
                is_unsat = True
            elif "sat" in content.strip():
                is_sat = True
            elif "timeout" in content.strip() or "early_stop" in content.strip() or "unknown" in content.strip():
                is_timeout = True
            else:
                print(content, result_file)
                is_timeout = True
        elif verifier_name == "crown":
            if "unsat" in content.strip():
                is_unsat = True
            elif "sat" in content.strip():
                is_sat = True
            elif "timeout" in content.strip():
                is_timeout = True
            else:
                is_timeout = True
        else:
            raise Exception("Unknown verifier name")

    # Determine status
    if is_sat:
        status = "unsafe"
    elif is_unsat:
        status = "safe"
    elif is_timeout:
        status = "timeout"
    else:
        status = "undecided"

    # Filter by mode if specified
    if mode == "sat" and not is_sat:
        return  # Skip non-SAT results
    elif mode == "unsat" and not is_unsat:
        return  # Skip non-UNSAT results
    # If mode == "both", process all results

    # Parse file path to extract components
    parts = result_file.parts
    if benchmark == "veridou":
        # Path format: benchmark_type/task/perturbation_type/kernel_size/angle_range/perturb_ratio/strength/log_id.txt
        if len(parts) >= 6:
            benchmark_type = parts[-8] if len(parts) > 7 else "unknown"
            task = parts[-7]  # mnist_fc
            perturbation_type = parts[-4]  # 0-30 (angle_range is the actual perturbation type)
            kernel_size = parts[-5]  # 5
            angle_range = parts[-6]  # general (this is just a category name)
            perturb_ratio = parts[-3]  # 0.0
            strength = parts[-2]  # 0.04
            file_idx = int(result_file.parts[-1].split("_")[1].split(".")[0]) - 1
            instances_csv_path = f"benchmarks/{task}/instances.csv"
            if not os.path.exists(instances_csv_path):
                # Try using the benchmark_dir directly
                instances_csv_path = os.path.join(benchmark_dir, "instances.csv")
            with open(instances_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                instances = f.readlines()
                # print(len(instances), file_idx)
                model_name = instances[file_idx].split(",")[0]
                vnnlib_path = instances[file_idx].split(",")[1]

            # Generate images from log file and vnnlib
            origin_path, perturbation_path = generate_image_pairs_from_log(
                result_file, instances_csv_path, benchmark_dir, temp_dir, use_full_kernel, spec_dir
            )

            # Calculate perceptual metrics if images are generated successfully
            ssim_value, psnr_value, lpips_value = None, None, None
            if origin_path and perturbation_path:
                ssim_value, psnr_value = calculate_ssim_psnr(origin_path, perturbation_path)
                if lpips_model:
                    lpips_value = calculate_lpips(origin_path, perturbation_path, lpips_model)

            csv_data.append(
                [
                    "image",
                    task,
                    model_name,
                    vnnlib_path,
                    kernel_size,
                    perturbation_type,
                    perturb_ratio,
                    strength,
                    status,
                    runtime,
                    origin_path or "",
                    perturbation_path or "",
                    ssim_value,
                    psnr_value,
                    lpips_value,
                ]
            )

            if is_sat:
                results[strength][perturbation_type]["unsafe"] += 1
            elif is_unsat:
                results[strength][perturbation_type]["safe"] += 1
            elif is_timeout:
                results[strength][perturbation_type]["timeouts"] += 1
            else:
                raise Exception("Undecided result")

    elif benchmark == "independent":
        # Path format: result_independent_nsat/task/perturbation_type/kernel_size/strength/log_id.txt
        # For independent: kernel_size field = task, perturb_ratio field = actual kernel_size, perturb_ratio = "-"
        if len(parts) >= 6:
            benchmark_type = parts[-6] if len(parts) > 6 else "unknown"
            task = parts[-5]  # This becomes the kernel_size field in CSV
            perturbation_type = parts[-4]  # This is the actual perturbation type
            actual_kernel_size = parts[-3]  # This becomes the perturb_ratio field in CSV
            strength = parts[-2]
            C = "-"  # perturb_ratio is always "-" for independent benchmarks
            file_idx = result_file.parts[-1].split("_")[1].split(".")[0]
            instances_csv_path = f"benchmarks/{task}/instances.csv"
            if not os.path.exists(instances_csv_path):
                # Try using the benchmark_dir directly
                instances_csv_path = os.path.join(benchmark_dir, "instances.csv")
            with open(instances_csv_path, "r", encoding="utf-8", errors="ignore") as f:
                instances = f.readlines()
                model_name = instances[int(file_idx)].split(",")[0]
                vnnlib_path = instances[int(file_idx)].split(",")[1]

            # Generate images from log file and vnnlib
            origin_path, perturbation_path = generate_image_pairs_from_log(
                result_file, instances_csv_path, benchmark_dir, temp_dir, use_full_kernel, spec_dir
            )

            # Calculate perceptual metrics if images are generated successfully
            ssim_value, psnr_value, lpips_value = None, None, None
            if origin_path and perturbation_path:
                ssim_value, psnr_value = calculate_ssim_psnr(origin_path, perturbation_path)
                if lpips_model:
                    lpips_value = calculate_lpips(origin_path, perturbation_path, lpips_model)

            csv_data.append(
                [
                    "image",
                    task,  # This is the task name (mnist_fc, cifar100, etc.)
                    model_name,
                    vnnlib_path,
                    actual_kernel_size,  # kernel_size field gets the actual kernel size
                    perturbation_type,  # perturbation_type field gets the actual perturbation type
                    "-",  # perturb_ratio field is always "-" for independent benchmarks
                    strength,
                    status,
                    runtime,
                    origin_path or "",
                    perturbation_path or "",
                    ssim_value,
                    psnr_value,
                    lpips_value,
                ]
            )

            if is_sat:
                results[strength][perturbation_type]["unsafe"] += 1
            elif is_unsat:
                results[strength][perturbation_type]["safe"] += 1
            elif is_timeout:
                results[strength][perturbation_type]["timeouts"] += 1
            else:
                raise Exception("Undecided result")
    else:
        raise Exception("Unknown benchmark type")


def parse_log_result_dir_with_metrics(
    log_file_path,
    benchmark,
    verifier_name,
    benchmark_dir,
    lpips_model,
    csv_name=None,
    use_full_kernel=False,
    spec_dir=None,
    recursive=False,
    mode="both",
    **kwargs,
):
    """
    Parse log files and extract statistics with perceptual metrics.
    Enhanced version that generates images on-the-fly from log files and vnnlib files.
    """
    # Dictionary to store results
    results = defaultdict(((lambda: defaultdict(create_default_stats))))

    # List to store CSV data
    csv_data = []

    # Create temporary directory for generated images
    temp_dir = os.path.join(os.path.dirname(csv_name) if csv_name else ".", "temp_images")

    if recursive:
        print(f"Searching for log directories in: {log_file_path}")
        log_dirs = find_log_directories(log_file_path)

        if not log_dirs:
            print("No log directories found!")
            return results

        print(f"Found {len(log_dirs)} log directories:")
        for log_dir in log_dirs:
            print(f"  {log_dir}")

        # Group by task (extract from path)
        task_groups = {}
        for log_dir in log_dirs:
            # Extract task from path
            path_parts = Path(log_dir).parts
            task = None
            for part in path_parts:
                if part in ["mnist_fc", "cifar100", "oval21", "sri_resnet_a", "tinyimagenet"]:
                    task = part
                    break

            if task:
                if task not in task_groups:
                    task_groups[task] = []
                task_groups[task].append(log_dir)

        # Process each task group
        for task, task_dirs in task_groups.items():
            print(f"\n=== Processing task: {task} ===")
            task_dir = os.path.join(benchmark_dir, task)

            if not os.path.exists(task_dir):
                print(f"Task directory not found: {task_dir}")
                continue

            for log_dir in task_dirs:
                print(f"\nProcessing: {log_dir}")
                # Process this directory
                for result_file in Path(log_dir).glob("*.txt"):
                    if not result_file.name.startswith("log_"):
                        continue
                    # Process individual file (same logic as below)
                    process_single_log_file(
                        result_file,
                        benchmark,
                        verifier_name,
                        task_dir,
                        spec_dir,
                        temp_dir,
                        use_full_kernel,
                        lpips_model,
                        csv_data,
                        results,
                        mode,
                    )
    else:
        print(f"Processing log files in: {log_file_path}")
        print(f"Using benchmark directory: {benchmark_dir}")
        print(f"Temporary images will be saved to: {temp_dir}")

        for result_file in Path(log_file_path).rglob("*.txt"):
            if not result_file.name.startswith("log_"):
                continue
            process_single_log_file(
                result_file,
                benchmark,
                verifier_name,
                benchmark_dir,
                spec_dir,
                temp_dir,
                use_full_kernel,
                lpips_model,
                csv_data,
                results,
                mode,
            )

    # Save CSV data with perceptual metrics
    if csv_data:
        csv_filename = f"{benchmark}_results_with_metrics.csv" if csv_name is None else csv_name
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "benchmark_type",
                    "task",
                    "model_name",
                    "vnnlib_path",
                    "kernel_size",
                    "perturbation_type",
                    "perturb_ratio",
                    "strength",
                    "status",
                    "runtime",
                    "origin_image_path",
                    "perturbation_image_path",
                    "ssim",
                    "psnr",
                    "lpips",
                ]
            )
            writer.writerows(csv_data)
        print(f"CSV data with perceptual metrics saved to {csv_filename}")

    return results


def print_statistics(results, benchmark):
    """Print formatted statistics."""
    print("\n" + "=" * 80)
    print("UNIFIED ANALYSIS STATISTICS")
    print("=" * 80)

    # Convert defaultdict to regular dict for better printing
    results_dict = defaultdict_to_dict(results)
    print("Results structure:")
    print(results_dict)

    total_safe = 0
    total_unsafe = 0
    total_undecided = 0
    total_timeouts = 0

    for strength in sorted(results.keys()):
        print(f"\nStrength: {strength}" if benchmark == "independent" else f"\nPerturbation Ratio: {strength}")
        print("-" * 50)

        if benchmark == "veridou":
            for perturbation_type in sorted(results[strength].keys()):
                print(f"\n  Perturbation Type: {perturbation_type}")
                print("  " + "-" * 40)

                type_safe = 0
                type_unsafe = 0
                type_undecided = 0
                type_timeouts = 0

                stats = results[strength][perturbation_type]
                safe = stats["safe"]
                unsafe = stats["unsafe"]
                undecided = stats["undecided"]
                timeouts = stats["timeouts"]
                total = safe + unsafe + undecided + timeouts

                type_safe += safe
                type_unsafe += unsafe
                type_undecided += undecided
                type_timeouts += timeouts

                type_total = type_safe + type_unsafe + type_undecided + type_timeouts
                print(f"\n    Type Total:")
                print(
                    f"      SAFE={type_safe}, UNSAFE={type_unsafe}, UNDECIDED={type_undecided}, TIMEOUTS={type_timeouts}"
                )
                print(f"      Total={type_total}")

                total_safe += type_safe
                total_unsafe += type_unsafe
                total_undecided += type_undecided
                total_timeouts += type_timeouts
        else:
            data_safe = 0
            data_unsafe = 0
            data_timeouts = 0

            for pertubation_type in sorted(results[strength].keys(), key=str):
                stats = results[strength][pertubation_type]
                safe = stats["safe"]
                unsafe = stats["unsafe"]
                timeouts = stats["timeouts"]
                total = safe + unsafe + timeouts

                data_safe += safe
                data_unsafe += unsafe
                data_timeouts += timeouts

                print(f"\n    Perturbation Type {pertubation_type}:")
                print(f"      SAFE={safe}, UNSAFE={unsafe}, TIMEOUTS={timeouts}")
                print(f"      Total={total}")

            data_total = data_safe + data_unsafe + data_timeouts
            print(f"\n  Data Total:")
            print(f"    SAFE={data_safe}, UNSAFE={data_unsafe}, TIMEOUTS={data_timeouts}")
            print(f"    Total={data_total}")

            total_safe += data_safe
            total_unsafe += data_unsafe
            total_timeouts += data_timeouts

    print("\n" + "=" * 80)
    grand_total = total_safe + total_unsafe + total_undecided + total_timeouts
    print("OVERALL TOTALS:")
    print(f"  SAFE={total_safe}, UNSAFE={total_unsafe}, UNDECIDED={total_undecided}, TIMEOUTS={total_timeouts}")
    print(f"  Total Cases={grand_total}")
    if grand_total > 0:
        print(f"\nPERCENTAGES:")
        print(f"  SAFE: {(total_safe/grand_total)*100:.1f}%")
        print(f"  UNSAFE: {(total_unsafe/grand_total)*100:.1f}%")
        print(f"  UNDECIDED: {(total_undecided/grand_total)*100:.1f}%")
        print(f"  TIMEOUTS: {(total_timeouts/grand_total)*100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified analysis script with perceptual metrics")
    parser.add_argument("--result_dir", required=True, help="Directory containing log files")
    parser.add_argument("--benchmark", choices=["independent", "veridou"], required=True, help="Benchmark type")
    parser.add_argument("--verifier", required=True, help="Verifier name (venus, neuralsat, crown)")
    parser.add_argument("--benchmark_dir", required=True, help="Benchmark directory containing instances.csv files")
    parser.add_argument(
        "--spec_dir", required=False, help="Spec directory containing vnnlib files (required for --use_full_kernel)"
    )
    parser.add_argument("--csv_name", required=False, help="Output CSV filename")
    parser.add_argument("--skip_lpips", action="store_true", help="Skip LPIPS calculation (faster)")
    parser.add_argument(
        "--use_full_kernel", action="store_true", help="Use full kernel matrix extraction instead of single z value"
    )
    parser.add_argument("--recursive", action="store_true", help="Process all subdirectories recursively")
    parser.add_argument(
        "--mode",
        choices=["sat", "unsat", "both"],
        default="both",
        help="Process mode: sat, unsat, or both (default: both)",
    )

    # Add usage examples
    parser.epilog = """
Examples:
  # Single z value approach (default)
  python unified_analysis.py --result_dir lam_usb/result_independent_crown/mnist_fc --benchmark independent --verifier crown --benchmark_dir benchmarks/mnist_fc

  # Full kernel matrix approach (requires spec_dir)
  python unified_analysis.py --result_dir lam_usb/result_single_angle_independent_nsat/mnist_fc --benchmark independent --verifier crown --benchmark_dir benchmarks/mnist_fc --spec_dir lam_usb/benchmarks_VeriDou_single_angle --use_full_kernel

  # Recursive processing (all benchmarks)
  python unified_analysis.py --result_dir lam_usb/result_single_angle_independent_nsat/ --benchmark independent --verifier crown --benchmark_dir benchmarks/ --spec_dir lam_usb/benchmarks_VeriDou_single_angle/ --use_full_kernel --recursive

  # Skip LPIPS for faster processing
  python unified_analysis.py --result_dir lam_usb/result_independent_crown/mnist_fc --benchmark independent --verifier crown --benchmark_dir benchmarks/mnist_fc --skip_lpips

  # Process only SAT results
  python unified_analysis.py --result_dir lam_usb/result_independent_crown/mnist_fc --benchmark independent --verifier crown --benchmark_dir benchmarks/mnist_fc --mode sat

  # Process only UNSAT results
  python unified_analysis.py --result_dir lam_usb/result_independent_crown/mnist_fc --benchmark independent --verifier crown --benchmark_dir benchmarks/mnist_fc --mode unsat
"""

    args = parser.parse_args()

    # Initialize LPIPS model if not skipping
    lpips_model = None
    if not args.skip_lpips:
        print("Initializing LPIPS model...")
        try:
            lpips_model = lpips.LPIPS(net="alex")
            print("LPIPS model initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS model: {e}")
            print("Continuing without LPIPS calculation...")

    # Run unified analysis
    results = parse_log_result_dir_with_metrics(
        args.result_dir,
        args.benchmark,
        args.verifier,
        args.benchmark_dir,
        lpips_model,
        args.csv_name,
        args.use_full_kernel,
        args.spec_dir,
        args.recursive,
        args.mode,
    )

    print_statistics(results, args.benchmark)

    print("\nUnified analysis complete!")
    print("Check the generated CSV file for verification results with perceptual metrics.")
