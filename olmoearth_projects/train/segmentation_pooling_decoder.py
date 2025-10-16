"""SegmentationPoolingDecoder class."""

from typing import Any

import torch
from rslearn.models.pooling_decoder import PoolingDecoder


class SegmentationPoolingDecoder(PoolingDecoder):
    """Like rslearn.models.pooling_decoder, but copy output to all pixels.

    This allows for global output with compatibility for SegmentationTask.
    """

    def forward(
        self, features: list[torch.Tensor], inputs: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Extend PoolingDecoder forward to upsample the output to a segmentation mask.

        This only works when all of the pixels have the same segmentation target.
        """
        output_probs = super().forward(features, inputs)
        # BC -> BCHW
        h, w = inputs[0]["sentinel2_l2a"].shape[1:3]
        return output_probs[:, :, None, None].repeat([1, 1, h, w])
