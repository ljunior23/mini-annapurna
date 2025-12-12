from typing import Tuple, Optional

try:
    import torch
    import torch.fx as fx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def capture_matmul(model) -> Tuple[int, int, int]:
    """
    Capture FX graph and extract matrix dimensions from matmul operation.
    Returns (M, K, N) where C[M×N] = A[M×K] @ B[K×N]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for FX graph capture. Install with: pip install torch")
    
    traced = fx.symbolic_trace(model)
    
    for node in traced.graph.nodes:
        if node.op == 'call_function' and node.target == torch.matmul:
            # Extract shapes from tensor metadata
            a_shape = node.args[0].meta.get('tensor_meta', None)
            b_shape = node.args[1].meta.get('tensor_meta', None)
            
            if a_shape and b_shape:
                m, k = a_shape.shape
                k2, n = b_shape.shape
                assert k == k2, f"Inner dimensions must match: {k} != {k2}"
                return int(m), int(k), int(n)
    
    raise ValueError("No matmul operation found in model")


def extract_from_shapes(m: int, k: int, n: int) -> Tuple[int, int, int]:
    """
    Direct extraction when shapes are known.
    Returns (M, K, N) tuple.
    """
    return m, k, n


class SimpleMatMul:
    """Simple matmul wrapper for FX tracing"""
    def __init__(self, m: int, k: int, n: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        self.A = torch.nn.Parameter(torch.randn(m, k))
        self.B = torch.nn.Parameter(torch.randn(k, n))
    
    def forward(self):
        return torch.matmul(self.A, self.B)
