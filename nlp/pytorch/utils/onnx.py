# -------------------------------------------------------------------------
#   ONNX Converter
# -------------------------------------------------------------------------
# Imports
from typing import Tuple, Optional, List, Dict
import torch


# -------------------------------------------------------------------------
#   Globals
# -------------------------------------------------------------------------
Tensor = torch.Tensor
OptStringList = Optional[List[str]]


# -------------------------------------------------------------------------
#   PyTorch -> ONNX
# -------------------------------------------------------------------------
def convert_model_to_onnx(
        net:torch.nn.Module,
        input_tensors:Tuple[Tensor, ...],
        path:str,
        dynamic_axes:Dict[str, Dict[int, str]],
        input_names:OptStringList=None,
        output_names:OptStringList=None) -> None:
    """
    """
    # Export Model
    torch.onnx.export(
            net,
            input_tensors,
            path,
            export_params=True,
            opset_version=10,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes)
