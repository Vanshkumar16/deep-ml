import torch
import torch.nn as nn

def calculate_parameters(layers: list[dict]) -> int:
    """
    Calculate the total number of trainable parameters in a neural network
    using PyTorch's built-in nn modules.

    Args:
        layers: List of dictionaries, each describing a layer.
                Supported types: 'dense' (uses nn.Linear), 'conv2d' (uses nn.Conv2d).

    Returns:
        Total number of trainable parameters (int).
    """
    # Your code here
    total = 0

    for layer in layers:
        if layer["type"] == "dense":
            params = layer["input_size"] * layer["output_size"]

            if layer.get("bias", True):
                params += layer["output_size"]

        elif layer["type"] == "conv2d":
            params = (
                layer["in_channels"]
                * layer["out_channels"]
                * layer["kernel_size"]
                * layer["kernel_size"]
            )

            if layer.get("bias", True):
                params += layer["out_channels"]

        total += params

    return total