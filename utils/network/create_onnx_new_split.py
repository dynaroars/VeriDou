from onnx import helper, numpy_helper, TensorProto
from beartype import beartype
import numpy as np
import torch
import onnx
import os
import cv2
from typing import List

from utils.network.create_onnx_aaai import create_motion_blur_kernel


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
    print(kernel)
    return kernel

@beartype
def visualize_conv2d(
    spec_id: int,
    image: torch.Tensor,
    perturbed_image: torch.Tensor,
    kernel_size: int,
    benchmark_name: str,
    output_dir: str = "conv_visualization_new",
):
    """Visualize the conv2d operations for verification"""
    os.makedirs(output_dir, exist_ok=True)
    b, c, h, w = image.shape
    canvas = np.zeros((h, w * 2, c))
    canvas[:, :w, :] = np.clip(image.squeeze(0).detach().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    canvas[:, w:, :] = np.clip(perturbed_image.squeeze(0).detach().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(
        np.uint8
    )

    # Save the visualization
    filename = f"{benchmark_name}_{spec_id}_{kernel_size}x{kernel_size}.png"
    cv2.imwrite(os.path.join(output_dir, filename), canvas)  # type: ignore
    
    print(f"Visualization saved to {output_dir}/{filename}")
    print(f"Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"Perturbed range: [{perturbed_image.min().item():.3f}, {perturbed_image.max().item():.3f}]")
    print(
        f"Difference range: [{perturbed_image.min().item() - image.min().item():.3f}, {perturbed_image.max().item() - image.max().item():.3f}]"
    )
    # Verify that the images are different (as expected for visualization)
    assert not torch.equal(image, perturbed_image), "Images should be different for visualization"


@beartype
def get_first_layer_type(onnx_model: onnx.ModelProto) -> str:
    """Detect if the first layer is Conv or MatMul/Gemm"""
    graph = onnx_model.graph
    first_node = graph.node[0]
    if first_node.op_type in ["Conv"]:
        return "conv"
    elif first_node.op_type in ["MatMul", "Gemm"]:
        return "fc"
    else:
        # Look for the first computational node
        for node in graph.node:
            if node.op_type in ["Conv"]:
                return "conv"
            elif node.op_type in ["MatMul", "Gemm"]:
                return "fc"
    return "unknown"


@beartype
def create_onnx(
    spec_id: int,
    onnx_model: onnx.ModelProto,
    output_path: str,
    image: torch.Tensor,
    kernel_type: str,
    strength: List[float],
    perturb_ratio: float = 0.1,
    robust_interval: float = 0.1,
    visualize: bool = False,
    benchmark_name: str = "mnist_fc",
    random_seed: int = 42,
    kernel_size: int = 3,
):
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Detect first layer type
    first_layer_type = get_first_layer_type(onnx_model)
    print(f"Detected first layer type: {first_layer_type}")

    # step 2: create appropriate front layer based on first layer type
    if first_layer_type == "fc":
        # For FC networks (like mnist_fc), use GEMM layer
        return _create_fc_front_layer(
            onnx_model,
            output_path,
            image,
            kernel_type,
            strength,
            robust_interval,
            perturb_ratio,
            spec_id,
            benchmark_name,
            visualize,
            random_seed,
            kernel_size,
        )
    elif first_layer_type == "conv":
        # For Conv networks (like oval21), use Conv layer
        return _create_conv_front_layer(
            onnx_model,
            output_path,
            image,
            strength,
            robust_interval,
            perturb_ratio,
            spec_id,
            benchmark_name,
            visualize,
            random_seed,
            kernel_size,
        )
    else:
        raise ValueError(f"Unsupported first layer type: {first_layer_type}")


@beartype
def _create_fc_front_layer(
    onnx_model: onnx.ModelProto,
    output_path: str,
    image: torch.Tensor,
    kernel_type: str,
    strength: List[float],
    robust_interval: float,
    perturb_ratio: float,
    spec_id: int,
    benchmark_name: str,
    visualize: bool,
    random_seed: int,
    kernel_size: int,
):
    """Create GEMM front layer for FC networks"""
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if len(image.shape) == 4:
        b, c, h, w = image.shape
    else:
        b, hw, c = image.shape  # mnist_fc
        h, w = int(np.sqrt(hw)), int(np.sqrt(hw))
        image = image.view(b, 1, h, w)

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

    # For visualization, create a sample perturbation to show the effect
    if visualize:
        # Create example z values to demonstrate the perturbation effect
        if kernel_type == "fixed":
            # example_z_values = np.random.uniform(strength[0], strength[1], kernel_size * kernel_size)  # Small perturbation
            example_z_values = create_motion_blur_kernel_range(0, 0, kernel_size)
            example_z_values[example_z_values!=0] = 0
            example_z_values[kernel_size // 2, kernel_size // 2] = 0
            example_z_values = example_z_values.flatten()
        else:
            example_z_values = np.random.uniform(
                strength[0], strength[1], kernel_size * kernel_size
            )  # Small perturbation
        
        
        # Apply the A matrix with example z values: F(X) = z @ A.T + B
        fx_result = np.dot(example_z_values, A_matrix.T) + B_flat
        fx_tensor = torch.from_numpy(fx_result.astype(np.float32))
        # Reshape F(X) back to image dimensions for visualization
        if len(image.shape) == 4:
            b, c, h, w = image.shape
            fx_image = fx_tensor.view(b, c, h, w)
        else:
            b, hw, c = image.shape
            h, w = int(np.sqrt(hw)), int(np.sqrt(hw))
            fx_image = fx_tensor.view(b, 1, h, w)
        assert 0 <= perturb_ratio <= 1
        # Add random noise to F(X) for the final perturbed image
        noise = torch.zeros_like(fx_image)
        mask = torch.rand_like(fx_image) < perturb_ratio
        noise[mask] = torch.randn_like(noise[mask]) * robust_interval
        noise[~mask] = torch.zeros_like(noise[~mask])
        perturbed_image = fx_image + noise
        perturbed_image = perturbed_image.clamp(image.min(), image.max())
        visualize_conv2d(spec_id, image, perturbed_image, kernel_size, benchmark_name)

    # Step 3: Create GEMM layer
    graph = onnx_model.graph
    num_z_inputs = kernel_size * kernel_size  # z values for each kernel position
    r_size = conv_result_B.flatten().shape[0]  # Size of flattened output
    combined_input_size = num_z_inputs + r_size
    combined_input = helper.make_tensor_value_info("combined_input", TensorProto.FLOAT, [None, combined_input_size])

    A_name, B_name = "front_matrix_A", "front_vec_B"

    graph.initializer.extend([numpy_helper.from_array(A_matrix, A_name), numpy_helper.from_array(B_flat, B_name)])

    # Check opset version to determine Slice format
    opset_version = onnx_model.opset_import[0].version if onnx_model.opset_import else 1
    
    if opset_version >= 10:
        # Use new format for opset 10+
        # Create constant tensors for slice parameters
        z_starts = helper.make_tensor("z_starts", TensorProto.INT64, [1], [0])
        z_ends = helper.make_tensor("z_ends", TensorProto.INT64, [1], [num_z_inputs])
        z_axes = helper.make_tensor("z_axes", TensorProto.INT64, [1], [1])
        
        r_starts = helper.make_tensor("r_starts", TensorProto.INT64, [1], [num_z_inputs])
        r_ends = helper.make_tensor("r_ends", TensorProto.INT64, [1], [combined_input_size])
        r_axes = helper.make_tensor("r_axes", TensorProto.INT64, [1], [1])
        
        graph.initializer.extend([z_starts, z_ends, z_axes, r_starts, r_ends, r_axes])

        # Extract z values (first kernel_size^2 elements) for GEMM
        z_slice_out = "z_slice"
        z_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input", "z_starts", "z_ends", "z_axes"],
            outputs=[z_slice_out],
            name="ZSlice",
        )

        # Extract R part (remaining elements) for Add
        r_slice_out = "r_slice"
        r_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input", "r_starts", "r_ends", "r_axes"],
            outputs=[r_slice_out],
            name="RSlice",
        )
    else:
        # Use old format for opset < 10
        # Extract z values (first kernel_size^2 elements) for GEMM
        z_slice_out = "z_slice"
        z_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input"],
            outputs=[z_slice_out],
            name="ZSlice",
            axes=[1],
            starts=[0],
            ends=[num_z_inputs],
        )

        # Extract R part (remaining elements) for Add
        r_slice_out = "r_slice"
        r_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input"],
            outputs=[r_slice_out],
            name="RSlice",
            axes=[1],
            starts=[num_z_inputs],
            ends=[combined_input_size],
        )

    gemm_out = "gemm_out"
    gemm_node = helper.make_node(
        "Gemm",
        inputs=[z_slice_out, A_name, B_name],
        outputs=[gemm_out],
        name="FrontGemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )

    # Create Add node to compute R + F(X)
    add_out = "add_out"
    add_node = helper.make_node("Add", inputs=[r_slice_out, gemm_out], outputs=[add_out], name="FrontAdd")

    graph.input.insert(0, combined_input)
    graph.node.insert(0, z_slice_node)
    graph.node.insert(1, r_slice_node)
    graph.node.insert(2, gemm_node)
    graph.node.insert(3, add_node)

    # Update connections
    orig_input = graph.input[1]  # Now at index 1 after inserting combined input
    orig_input_name = orig_input.name
    for node in graph.node[4:]:  # Skip the 4 front layer nodes
        for i, inp in enumerate(node.input):
            if inp == orig_input_name:
                node.input[i] = add_out
    graph.input.remove(orig_input)

    onnx.save(onnx_model, output_path)


@beartype
def _create_conv_front_layer(
    onnx_model: onnx.ModelProto,
    output_path: str,
    image: torch.Tensor,
    strength: List[float],
    robust_interval: float,
    perturb_ratio: float,
    spec_id: int,
    benchmark_name: str,
    visualize: bool,
    random_seed: int,
    kernel_size: int,
):
    """Create GEMM + Reshape front layer for Conv networks"""
    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

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

    # For visualization, create a sample perturbation to show the effect
    if visualize:
        if kernel_type == "fixed":
            # Create example z values to demonstrate the perturbation effect
            # example_z_values = np.random.uniform(strength[0], strength[1], kernel_size * kernel_size)  # Small perturbation
            example_z_values = create_motion_blur_kernel_range(0, 0, kernel_size)
            example_z_values[example_z_values!=0] = 0
            example_z_values[kernel_size // 2, kernel_size // 2] = 0
            example_z_values = example_z_values.flatten()
        else:
            example_z_values = np.random.uniform(
                strength[0], strength[1], kernel_size * kernel_size
            )  # Small perturbation
        
        # Apply the A matrix with example z values: F(X) = z @ A.T + B
        fx_result = np.dot(example_z_values, A_matrix.T) + B_flat
        fx_tensor = torch.from_numpy(fx_result.astype(np.float32))

        # Reshape F(X) back to spatial dimensions for visualization
        fx_image = fx_tensor.view(b, c, h, w)
        assert 0 <= perturb_ratio <= 1
        # Add random noise to F(X) for the final perturbed image
        noise = torch.zeros_like(fx_image)
        mask = torch.rand_like(fx_image) < perturb_ratio
        noise[mask] = torch.randn_like(noise[mask]) * robust_interval
        noise[~mask] = torch.zeros_like(noise[~mask])
        perturbed_image = fx_image + noise
        perturbed_image = perturbed_image.clamp(image.min(), image.max())
        visualize_conv2d(spec_id, image, perturbed_image, kernel_size, benchmark_name)

    # Step 3: Create GEMM + Reshape layer for ONNX graph
    graph = onnx_model.graph
    num_z_inputs = kernel_size * kernel_size  # z values for each kernel position
    r_size = c * h * w  # Size of flattened spatial tensor
    combined_input_size = num_z_inputs + r_size
    combined_input = helper.make_tensor_value_info("combined_input", TensorProto.FLOAT, [None, combined_input_size])

    A_name, B_name = "front_matrix_A", "front_vec_B"

    graph.initializer.extend([numpy_helper.from_array(A_matrix, A_name), numpy_helper.from_array(B_flat, B_name)])

    # Check opset version to determine Slice format
    opset_version = onnx_model.opset_import[0].version if onnx_model.opset_import else 1
    
    if opset_version >= 10:
        # Use new format for opset 10+
        # Create constant tensors for slice parameters
        z_starts = helper.make_tensor("z_starts", TensorProto.INT64, [1], [0])
        z_ends = helper.make_tensor("z_ends", TensorProto.INT64, [1], [num_z_inputs])
        z_axes = helper.make_tensor("z_axes", TensorProto.INT64, [1], [1])
        
        r_starts = helper.make_tensor("r_starts", TensorProto.INT64, [1], [num_z_inputs])
        r_ends = helper.make_tensor("r_ends", TensorProto.INT64, [1], [combined_input_size])
        r_axes = helper.make_tensor("r_axes", TensorProto.INT64, [1], [1])
        
        graph.initializer.extend([z_starts, z_ends, z_axes, r_starts, r_ends, r_axes])

        # Extract z values (first kernel_size^2 elements) for GEMM
        z_slice_out = "z_slice"
        z_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input", "z_starts", "z_ends", "z_axes"],
            outputs=[z_slice_out],
            name="ZSlice",
        )

        # Extract R part (remaining elements) for Add
        r_slice_out = "r_slice"
        r_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input", "r_starts", "r_ends", "r_axes"],
            outputs=[r_slice_out],
            name="RSlice",
        )
    else:
        # Use old format for opset < 10
        # Extract z values (first kernel_size^2 elements) for GEMM
        z_slice_out = "z_slice"
        z_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input"],
            outputs=[z_slice_out],
            name="ZSlice",
            axes=[1],
            starts=[0],
            ends=[num_z_inputs],
        )

        # Extract R part (remaining elements) for Add
        r_slice_out = "r_slice"
        r_slice_node = helper.make_node(
            "Slice",
            inputs=["combined_input"],
            outputs=[r_slice_out],
            name="RSlice",
            axes=[1],
            starts=[num_z_inputs],
            ends=[combined_input_size],
        )

    # Create GEMM node
    gemm_out = "gemm_out"
    gemm_node = helper.make_node(
        "Gemm",
        inputs=[z_slice_out, A_name, B_name],
        outputs=[gemm_out],
        name="FrontGemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )

    # Create Reshape node to convert GEMM output back to spatial dimensions
    reshape_out = "reshape_out"
    # Shape should be [batch_size, channels, height, width]
    reshape_shape = np.array([-1, c, h, w], dtype=np.int64)  # -1 for dynamic batch size
    shape_name = "reshape_shape"
    graph.initializer.append(numpy_helper.from_array(reshape_shape, shape_name))

    reshape_node = helper.make_node(
        "Reshape", inputs=[gemm_out, shape_name], outputs=[reshape_out], name="FrontReshape"
    )

    # Reshape R slice to spatial dimensions for element-wise add
    r_reshape_out = "r_reshape_out"
    r_reshape_node = helper.make_node(
        "Reshape", inputs=[r_slice_out, shape_name], outputs=[r_reshape_out], name="RReshape"
    )

    # Create Add node to compute R + F(X)
    add_out = "add_out"
    add_node = helper.make_node("Add", inputs=[r_reshape_out, reshape_out], outputs=[add_out], name="FrontAdd")

    # Insert new input and nodes
    graph.input.insert(0, combined_input)
    graph.node.insert(0, z_slice_node)
    graph.node.insert(1, r_slice_node)
    graph.node.insert(2, gemm_node)
    graph.node.insert(3, reshape_node)
    graph.node.insert(4, r_reshape_node)
    graph.node.insert(5, add_node)

    # Update connections from original input to add output
    orig_input = graph.input[1]  # Now at index 1 after inserting combined input
    orig_input_name = orig_input.name
    for node in graph.node[6:]:  # Skip the 6 front layer nodes
        for i, inp in enumerate(node.input):
            if inp == orig_input_name:
                node.input[i] = add_out
    graph.input.remove(orig_input)

    onnx.save(onnx_model, output_path)
