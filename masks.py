# Code from https://github.com/BadCafeCode/masquerade-nodes-comfyui?tab=readme-ov-file

import os
import torch
import numpy as np
import math
from torchvision import transforms
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as torchfn
import subprocess
import sys


class SR_MaskMorphologyNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "distance": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1}),
                "op": (["dilate", "erode", "open", "close"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "morph"

    CATEGORY = "Masquerade Nodes"

    def morph(self, mask, distance, op):
        if op == "dilate":
            mask = self.dilate(mask, distance)
        elif op == "erode":
            mask = self.erode(mask, distance)
        elif op == "open":
            mask = self.erode(mask, distance)
            mask = self.dilate(mask, distance)
        elif op == "close":
            mask = self.dilate(mask, distance)
            mask = self.erode(mask, distance)
        return (mask,)

    def erode(self, mask, distance):
        return 1. - self.dilate(1. - mask, distance)

    def dilate(self, mask, distance):
        kernel_size = 1 + distance * 2
        # Add the channels dimension
        mask = mask.unsqueeze(1)
        out = torchfn.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2).squeeze(1)
        return out
