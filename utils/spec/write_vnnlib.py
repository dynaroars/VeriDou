from beartype import beartype
import torch

@beartype
def write_vnnlib(spec_path: str, data_lb: torch.Tensor, data_ub: torch.Tensor, prediction: torch.Tensor):
    # input bounds
    x_lb = data_lb.flatten()
    x_ub = data_ub.flatten()
    
    # outputs
    n_class = prediction.numel()
    y = prediction.argmax(-1).item()
    
    with open(spec_path, "w") as f:
        f.write(f"; Specification for class {int(y)}\n")

        f.write(f"\n; Definition of input variables\n")
        for i in range(len(x_ub)):
            f.write(f"(declare-const X_{i} Real)\n")

        f.write(f"\n; Definition of output variables\n")
        for i in range(n_class):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write(f"\n; Definition of input constraints\n")
        for i in range(len(x_ub)):
            f.write(f"(assert (<= X_{i} {x_ub[i]:.8f}))\n")
            f.write(f"(assert (>= X_{i} {x_lb[i]:.8f}))\n\n")

        f.write(f"\n; Definition of output constraints\n")
        f.write(f"(assert (or\n")
        for i in range(n_class):
            if i == y:
                continue
            f.write(f"\t(and (>= Y_{i} Y_{y}))\n")
        f.write(f"))\n")
