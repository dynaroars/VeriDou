from beartype import beartype
import onnxruntime as ort
import numpy as np
import onnx
import io


@beartype
def _load_onnx(path: str | io.BytesIO):
    if isinstance(path, str):
        onnx_model = onnx.load(path)
    else:
        onnx_model = onnx.load_model_from_string(path.getvalue())
    # print(onnx_model)
    return onnx_model

@beartype
def inference_onnx(path: str | io.BytesIO, *inputs: np.ndarray):
    sess = ort.InferenceSession(_load_onnx(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, dict(zip(names, inputs)))


@beartype
def add_batch(shape: tuple) -> tuple:
    if len(shape) == 1:
        return (1, shape[0])
    
    if shape[0] not in [-1, 1]:
        return (1, *shape)
    
    return shape
        

@beartype
def parse_onnx(path: str | io.BytesIO, input_shape: None | list = None, output_shape: None | list = None) -> tuple:
    # load model
    onnx_model = _load_onnx(path)
    
    # extract shapes
    onnx_inputs = [node.name for node in onnx_model.graph.input]
    initializers = [node.name for node in onnx_model.graph.initializer]
    inputs = list(set(onnx_inputs) - set(initializers))
    inputs = [node for node in onnx_model.graph.input if node.name in inputs]
    
    if input_shape is None:
        onnx_input_dims = inputs[0].type.tensor_type.shape.dim
        orig_input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_input_dims)
        batched_input_shape = add_batch(orig_input_shape)
    else:
        orig_input_shape = batched_input_shape = tuple(input_shape)
        
    if output_shape is None:
        onnx_output_dims = onnx_model.graph.output[0].type.tensor_type.shape.dim
        orig_output_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in onnx_output_dims) if len(onnx_output_dims) else (1,)
        batched_output_shape = add_batch(orig_output_shape)
    else:
        batched_output_shape = tuple(output_shape)
        
    return onnx_model, batched_input_shape, batched_output_shape
