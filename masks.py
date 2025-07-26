# Code from https://github.com/BadCafeCode/masquerade-nodes-comfyui?tab=readme-ov-file

import torch
import numpy as np
import torch.nn.functional as torchfn
import torchvision.transforms.functional as TF
import cv2


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

    CATEGORY = "Showrunner Nodes"

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


class SR_OutlineMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "distance": ("INT", {"default": 5, "min": 0, "max": 128, "step": 1})
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "outline"

    CATEGORY = "Showrunner Nodes"

    def outline(self, mask, distance):
        mask1 = tensor2mask(self.dilate(mask, distance))
        mask2 = tensor2mask(self.erode(mask, distance))
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255

        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            return mask1
        
    def erode(self, mask, distance):
        return 1. - self.dilate(1. - mask, distance)

    def dilate(self, mask, distance):
        kernel_size = 1 + distance * 2
        # Add the channels dimension
        mask = mask.unsqueeze(1)
        out = torchfn.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2).squeeze(1)
        return out

    def subtract_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255

        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            return mask1
    
def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t

def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t