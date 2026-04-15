import re
from onnx import helper, numpy_helper, TensorProto
from beartype import beartype
import numpy as np
import torch
import onnx
import os
import cv2


def create_motion_blur_kernel(angle_degrees, kernel_size=5):
    """
    Create accurate motion blur kernels based on common patterns.

    Angle convention: angles are interpreted in degrees, measured CLOCKWISE
    from the positive y-axis (upwards). For example:
    - 0°: vertical line (top-bottom)
    - 90°: horizontal line (left-right)
    - 45°: main diagonal (top-left to bottom-right)
    - 135°: anti-diagonal (top-right to bottom-left)

    Args:
        angle_degrees (float): Angle of motion blur in degrees (clockwise)
        kernel_size (int): Size of the kernel (must be odd)

    Returns:
        numpy.ndarray: Motion blur kernel
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")

    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    # Normalize angle to 0-360 range (clockwise from +Y per requirement)
    angle_user = angle_degrees % 360

    # Handle principal directions directly using user angle
    if angle_user % 180 == 0:  # Vertical motion blur (0°, 180°)
        kernel[:, center] = 1.0 / kernel_size

    elif angle_user % 180 == 90:  # Horizontal motion blur (90°, 270°)
        kernel[center, :] = 1.0 / kernel_size

    else:
        # For other angles, use improved line drawing that touches edges
        # Map clockwise-from-vertical to image-space direction vector
        angle_rad = np.radians(angle_user)

        # Calculate direction vector
        # In image coordinates (x right, y down):
        # theta=0 -> up (dx=0, dy=-1), theta=90 -> right (dx=1, dy=0)
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)

        # Find intersection points with kernel boundaries
        points = []

        # Check all four boundaries
        boundaries = [
            # Top edge (y = 0)
            lambda t: (t, 0) if 0 <= t < kernel_size else None,
            # Bottom edge (y = kernel_size - 1)
            lambda t: (t, kernel_size - 1) if 0 <= t < kernel_size else None,
            # Left edge (x = 0)
            lambda t: (0, t) if 0 <= t < kernel_size else None,
            # Right edge (x = kernel_size - 1)
            lambda t: (kernel_size - 1, t) if 0 <= t < kernel_size else None,
        ]

        intersections = []
        for boundary in boundaries:
            for t in range(kernel_size):
                point = boundary(t)
                if point:
                    intersections.append(point)

        # Find the line that passes through center and touches two boundaries
        center_x, center_y = center, center

        # Calculate line parameters: y = mx + b or x = my + b
        if abs(dx) > abs(dy):  # More horizontal
            # Use x as parameter: y = (dy/dx) * (x - center_x) + center_y
            m = dy / dx
            b = center_y - m * center_x

            # Find x values that give y within bounds
            for x in range(kernel_size):
                y = m * x + b
                y_int = int(round(y))
                if 0 <= y_int < kernel_size:
                    points.append((y_int, x))
        else:  # More vertical
            # Use y as parameter: x = (dx/dy) * (y - center_y) + center_x
            m = dx / dy
            b = center_x - m * center_y

            # Find y values that give x within bounds
            for y in range(kernel_size):
                x = m * y + b
                x_int = int(round(x))
                if 0 <= x_int < kernel_size:
                    points.append((y, x_int))

        # Remove duplicates and set kernel values
        unique_points = list(set(points))
        if unique_points:
            for y, x in unique_points:
                kernel[y, x] = 1.0 / len(unique_points)

    return kernel


@beartype
def visualize_conv2d(
    image: torch.Tensor,
    perturbed_image: torch.Tensor,
    perturbation: str,
    kernel_size: int,
    benchmark_name: str,
    spec_id: int,
    output_dir: str = "conv_visualization",
):
    """Visualize the conv2d operations for verification"""
    os.makedirs(output_dir, exist_ok=True)
    b, c, h, w = image.shape
    canvas = np.zeros((h, w * 2, c))
    canvas[:, :w, :] = np.clip(image.squeeze(0).detach().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    canvas[:, w:, :] = np.clip(perturbed_image.squeeze(0).detach().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(
        np.uint8
    )

    # Save the visualizationw
    filename = f"{benchmark_name}_{perturbation}_{kernel_size}x{kernel_size}_{spec_id}.png"
    cv2.imwrite(os.path.join(output_dir, filename), canvas)

    print(f"Visualization saved to {output_dir}/{filename}")
    print(f"Image range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"Perturbed range: [{perturbed_image.min().item():.3f}, {perturbed_image.max().item():.3f}]")
    print(
        f"Difference range: [{perturbed_image.min().item() - image.min().item():.3f}, {perturbed_image.max().item() - image.max().item():.3f}]"
    )


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
    onnx_model: onnx.ModelProto,
    output_path: str,
    image: torch.Tensor,
    kernel_size: int,
    pertubation: str,
    visualize: bool = False,
    benchmark_name: str = "mnist_fc",
    spec_id: int = 0,
):
    # Detect first layer type
    first_layer_type = get_first_layer_type(onnx_model)
    print(f"Detected first layer type: {first_layer_type}")

    # step 1: create convolution filter
    if "motion_blur" in pertubation:
        angle = re.search(r"motion_blur_(\d+)", pertubation).group(1)
        # Ensure A is a torch tensor (downstream expects torch.Tensor)
        A_np = create_motion_blur_kernel(int(angle), kernel_size).astype(np.float32)
        A = torch.from_numpy(A_np)
        B = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        A[kernel_size // 2, kernel_size // 2] = 1 / kernel_size - 1
        B[kernel_size // 2, kernel_size // 2] = 1
    elif pertubation == "box_blur":
        A = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        B = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        A.fill_(1 / kernel_size**2)
        A[kernel_size // 2, kernel_size // 2] = 1 / kernel_size**2 - 1
        B[kernel_size // 2, kernel_size // 2] = 1
    elif pertubation == "sharpen":
        a_negative = -1 / (kernel_size**2 - (kernel_size - 1) * ((kernel_size - 1) / 2 + 1) - 1)
        A = torch.full((kernel_size, kernel_size), a_negative, dtype=torch.float32)
        A[kernel_size // 2, kernel_size // 2] = 1
        for j in range(kernel_size):
            if j <= kernel_size // 2:
                # Top half including center row
                zero_end = (kernel_size - 1) // 2 - j
            else:
                # Bottom half
                zero_end = j - (kernel_size - 1) // 2
            # Set zero entries at both ends of the row
            for i in range(zero_end):
                A[j, i] = 0
                A[j, kernel_size - 1 - i] = 0
        B = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        B[kernel_size // 2, kernel_size // 2] = 1
    else:
        raise ValueError(f"Unknown pertubation: {pertubation}")

    # step 2: create appropriate front layer based on first layer type
    if first_layer_type == "fc":
        # For FC networks (like mnist_fc), use GEMM layer
        return _create_fc_front_layer(
            onnx_model,
            output_path,
            image,
            A,
            B,
            kernel_size,
            pertubation,
            visualize,
            benchmark_name,
            spec_id,
        )
    elif first_layer_type == "conv":
        # For Conv networks (like oval21), use Conv layer
        return _create_conv_front_layer(
            onnx_model,
            output_path,
            image,
            A,
            B,
            kernel_size,
            pertubation,
            visualize,
            benchmark_name,
            spec_id,
        )
    else:
        raise ValueError(f"Unsupported first layer type: {first_layer_type}")


@beartype
def _create_fc_front_layer(
    onnx_model: onnx.ModelProto,
    output_path: str,
    image: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    kernel_size: int,
    pertubation: str,
    visualize: bool,
    benchmark_name: str,
    spec_id: int,
):
    """Create GEMM front layer for FC networks"""
    if len(image.shape) == 4:
        b, c, h, w = image.shape
    else:
        b, hw, c = image.shape  # mnist_fc
        h, w = int(np.sqrt(hw)), int(np.sqrt(hw))
        image = image.view(b, 1, h, w)

    # Create conv2d layers for processing
    conv2d_A = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )
    conv2d_A.weight.data = A.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    conv2d_A.bias.data = torch.zeros(c)

    conv2d_B = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )
    conv2d_B.weight.data = B.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    conv2d_B.bias.data = torch.zeros(c)

    # Apply convolutions
    conv_result_A = conv2d_A(image)
    conv_result_B = conv2d_B(image)

    # Visualize if requested
    if visualize:
        perturbed_image = conv_result_A * 1 + conv_result_B
        visualize_conv2d(image, perturbed_image, pertubation, kernel_size, benchmark_name, spec_id)

    # Create GEMM layer
    graph = onnx_model.graph
    new_input = helper.make_tensor_value_info("new_input", TensorProto.FLOAT, [None, 1])
    A_flat = conv_result_A.flatten().unsqueeze(-1).detach().numpy().astype(np.float32)
    B_flat = conv_result_B.flatten().detach().numpy().astype(np.float32)
    A_name, B_name = "front_vec_A", "front_vec_B"

    graph.initializer.extend([numpy_helper.from_array(A_flat, A_name), numpy_helper.from_array(B_flat, B_name)])

    gemm_out = "gemm_out"
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["new_input", A_name, B_name],
        outputs=[gemm_out],
        name="FrontGemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )

    graph.input.insert(0, new_input)
    graph.node.insert(0, gemm_node)

    # Update connections
    orig_input = graph.input[1]
    orig_input_name = orig_input.name
    for node in graph.node[1:]:
        for i, inp in enumerate(node.input):
            if inp == orig_input_name:
                node.input[i] = gemm_out
    graph.input.remove(orig_input)

    onnx.save(onnx_model, output_path)


@beartype
def _create_conv_front_layer(
    onnx_model: onnx.ModelProto,
    output_path: str,
    image: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    kernel_size: int,
    pertubation: str,
    visualize: bool,
    benchmark_name: str,
    spec_id: int,
):
    """Create GEMM + Reshape front layer for Conv networks"""
    b, c, h, w = image.shape

    # Create conv2d layers for processing
    conv2d_A = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )
    conv2d_A.weight.data = A.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    conv2d_A.bias.data = torch.zeros(c)

    conv2d_B = torch.nn.Conv2d(
        in_channels=c,
        out_channels=c,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        groups=c,
    )
    conv2d_B.weight.data = B.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    conv2d_B.bias.data = torch.zeros(c)

    # Apply convolutions
    conv_result_A = conv2d_A(image)
    conv_result_B = conv2d_B(image)

    # Visualize if requested
    if visualize:
        perturbed_image = conv_result_A * 1 + conv_result_B
        visualize_conv2d(image, perturbed_image, pertubation, kernel_size, benchmark_name, spec_id)

    # Create GEMM + Reshape layer for ONNX graph
    graph = onnx_model.graph
    new_input = helper.make_tensor_value_info("scalar_input", TensorProto.FLOAT, [None, 1])

    # Flatten the conv results for GEMM
    A_flat = conv_result_A.flatten().unsqueeze(-1).detach().numpy().astype(np.float32)
    B_flat = conv_result_B.flatten().detach().numpy().astype(np.float32)
    A_name, B_name = "front_vec_A", "front_vec_B"

    graph.initializer.extend([numpy_helper.from_array(A_flat, A_name), numpy_helper.from_array(B_flat, B_name)])

    # Create GEMM node
    gemm_out = "gemm_out"
    gemm_node = helper.make_node(
        "Gemm",
        inputs=["scalar_input", A_name, B_name],
        outputs=[gemm_out],
        name="FrontGemm",
        alpha=1.0,
        beta=1.0,
        transA=0,
        transB=1,
    )

    # Create Reshape node to convert back to spatial dimensions
    reshape_out = "reshape_out"
    # Shape should be [batch_size, channels, height, width]
    reshape_shape = np.array([1, c, h, w], dtype=np.int64)  # Assuming batch size 1
    shape_name = "reshape_shape"
    graph.initializer.append(numpy_helper.from_array(reshape_shape, shape_name))

    reshape_node = helper.make_node(
        "Reshape", inputs=[gemm_out, shape_name], outputs=[reshape_out], name="FrontReshape"
    )

    # Insert new input and nodes
    graph.input.insert(0, new_input)
    graph.node.insert(0, gemm_node)
    graph.node.insert(1, reshape_node)

    # Update connections from original input to reshape output
    orig_input = graph.input[1]  # Now at index 1 after inserting scalar_input
    orig_input_name = orig_input.name
    for node in graph.node[2:]:  # Skip the 2 front layer nodes
        for i, inp in enumerate(node.input):
            if inp == orig_input_name:
                node.input[i] = reshape_out
    graph.input.remove(orig_input)

    onnx.save(onnx_model, output_path)
