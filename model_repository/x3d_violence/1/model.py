"""
Development-only fallback Triton Python backend for X3D.

Prefer the exported ONNX model in normal deployments. Keep this file only for
local development and fallback debugging.
"""

import os

import torch
import triton_python_backend_utils as pb_utils  # type: ignore
import x3d_net as x3d


class TritonPythonModel:
    def initialize(self, args):
        weights = torch.load(
            os.path.join(os.path.dirname(__file__), "pre_trained_x3d_model.pt"),
            map_location="cpu",
        )

        self.model = x3d.generate_model(
            x3d_version="M",
            n_classes=2,
            n_input_channels=3,
            dropout=0,
            base_bn_splits=1,
        )

        clean_state = {}

        for k, v in weights.items():
            k = k.replace("module.", "")

            if "split_bn" in k:
                continue

            clean_state[k] = v

        self.model.load_state_dict(clean_state, strict=False)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "video")

            video = input_tensor.as_numpy()

            video = torch.from_numpy(video).float()

            if torch.cuda.is_available():
                video = video.cuda()

            with torch.no_grad():
                logits = self.model(video)

            logits = logits.cpu().numpy()

            out_tensor = pb_utils.Tensor("logits", logits)

            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
