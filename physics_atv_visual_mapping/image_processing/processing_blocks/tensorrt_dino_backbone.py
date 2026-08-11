"""TensorRT adapter for Talk2DINO's fixed-shape DINOv2 backbone."""

from pathlib import Path

import numpy as np
import torch


class TensorRTDinoBackbone(torch.nn.Module):
    """Expose a TensorRT engine through DINOv2's ``forward_features`` API."""

    def __init__(self, engine_path, feature_store, device="cuda"):
        super().__init__()

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT is required when tensorrt_engine is configured"
            ) from exc

        path = Path(engine_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {path}")

        self.trt = trt
        self.feature_store = feature_store
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Could not deserialize TensorRT engine {path}; "
                "engines must match the installed TensorRT version and Jetson"
            )
        self.context = self.engine.create_execution_context()

        input_names = []
        output_names = []
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                input_names.append(name)
            else:
                output_names.append(name)

        if input_names != ["image"]:
            raise ValueError(f"Expected TensorRT input ['image'], got {input_names}")
        required_outputs = {"patch_tokens", "attention_qkv"}
        if set(output_names) != required_outputs:
            raise ValueError(
                f"Expected TensorRT outputs {sorted(required_outputs)}, "
                f"got {sorted(output_names)}"
            )

        self.input_name = input_names[0]
        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.input_dtype = self._torch_dtype(
            self.engine.get_tensor_dtype(self.input_name)
        )
        self.outputs = {}
        for name in output_names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dimension < 0 for dimension in shape):
                raise ValueError(
                    f"TensorRT engine must use static output shapes; {name}={shape}"
                )
            self.outputs[name] = torch.empty(
                shape,
                dtype=self._torch_dtype(self.engine.get_tensor_dtype(name)),
                device=self.device,
            )
            self.context.set_tensor_address(name, self.outputs[name].data_ptr())

    def _torch_dtype(self, dtype):
        return torch.from_numpy(
            np.empty((), dtype=self.trt.nptype(dtype))
        ).dtype

    def forward_features(self, image):
        if tuple(image.shape) != self.input_shape:
            raise ValueError(
                f"TensorRT engine expects image shape {self.input_shape}, "
                f"got {tuple(image.shape)}"
            )

        image = image.to(device=self.device, dtype=self.input_dtype).contiguous()
        self.context.set_tensor_address(self.input_name, image.data_ptr())
        if not self.context.execute_async_v3(
            torch.cuda.current_stream(self.device).cuda_stream
        ):
            raise RuntimeError("TensorRT DINOv2 inference failed")

        self.feature_store["self_attn"] = self.outputs["attention_qkv"]
        return {"x_norm_patchtokens": self.outputs["patch_tokens"]}
