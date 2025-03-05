import torch
import numpy as np

from PIL import ImageFont, ImageDraw, Image
from torchvision.transforms.functional import to_pil_image
import matplotlib.font_manager as fm

if TYPE_CHECKING:
    from mypy.typeshed.stdlib._typeshed import SupportsDunderGT, SupportsDunderLT

from PIL import Image

#Code from https://github.com/dzqdzq/ComfyUI-crop-alpha
class SR_AlphaCropAndPositionImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "maintain_aspect": (["True", "False"], {"default": "True"}),
                "left_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "top_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "right_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
                "bottom_padding": ("INT", {"default": 0, "min": 0, "max": 1024, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")

    FUNCTION = "crop"
    CATEGORY = "image/processing"

    def crop(self, image, maintain_aspect, left_padding: int = 0, right_padding: int = 0, top_padding: int = 0, bottom_padding: int = 0):
        cropped_images = []
        cropped_masks = []
        alpha_images = []

        for img in image:
            alpha = img[..., 3]

            height = img.shape[0]
            width = img.shape[1]
            mask = (alpha > 0.01)

            rows = torch.any(mask, dim=1)
            cols = torch.any(mask, dim=0)

            ymin, ymax = self._find_boundary(rows)
            xmin, xmax = self._find_boundary(cols)

            if ymin is None or xmin is None:
                cropped_images.append(img)
                cropped_masks.append(torch.zeros_like(alpha))
                alpha_images.append(img)
                continue

            ymin = max(0, ymin - top_padding)
            ymax = min(height, ymax + bottom_padding)
            xmin = max(0, xmin - left_padding)
            xmax = min(width, xmax + right_padding)

            cropped = img[ymin:ymax, xmin:xmax, :4]
            cropped_mask = alpha[ymin:ymax, xmin:xmax]

            # Apply padding to the cropped image
            padded_height = ymax - ymin + top_padding + bottom_padding
            padded_width = xmax - xmin + left_padding + right_padding

            if maintain_aspect == "True":
                if padded_height > padded_width:
                    pad = (padded_height - padded_width) // 2
                    left_padding += pad
                    right_padding += pad
                    padded_width = padded_height
                else:
                    pad = (padded_width - padded_height) // 2
                    top_padding += pad
                    bottom_padding += pad
                    padded_height = padded_width

            padded_image = torch.zeros((padded_height, padded_width, 4), dtype=img.dtype)
            padded_image[top_padding:top_padding + (ymax - ymin), left_padding:left_padding + (xmax - xmin), :] = cropped

            padded_mask = torch.zeros((padded_height, padded_width), dtype=alpha.dtype)
            padded_mask[top_padding:top_padding + (ymax - ymin), left_padding:left_padding + (xmax - xmin)] = cropped_mask

            cropped_masks.append(padded_mask)
            alpha_images.append(padded_image)

        return alpha_images, cropped_masks, xmax - xmin + left_padding + right_padding, ymax - ymin + top_padding + bottom_padding
    
    def _find_boundary(self, arr):
        nz = torch.nonzero(arr)
        if nz.numel() == 0:
            return (None, None)
        return (nz[0].item(), nz[-1].item() + 1)


class SR_ShrinkImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["scale", "pixels"], {"default": "scale"}),
                "resize_algorithm": (list(resize_algorithms.keys()), {"default": "LANCZOS"}),
                "maintain_aspect": (["True", "False"], {"default": "True"})
            },
            "optional": {
                "scale": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "width": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1}),
                "height": ("FLOAT", {"default": 100, "min": 2, "max": 10000, "step": 1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shrink_image"
    CATEGORY = "image/processing"

    def calculate_scale(self, img, mode, maintain_aspect, scale=None, width=None, height=None):
        if mode == "scale":
            return scale
        else:
            img_width, img_height = img.size
            if maintain_aspect == "True":
                aspect_ratio = img_width / img_height
                if width / height > aspect_ratio:
                    width = height * aspect_ratio
                else:
                    height = width / aspect_ratio
            scale_x = width / img_width
            scale_y = height / img_height
            return min(scale_x, scale_y)

    def shrink_image_with_scale(self, img, scale, algorithm):
        width, height = img.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        return img.resize((new_width, new_height), algorithm)

    def shrink_image(self, image, mode, resize_algorithm, maintain_aspect, scale=None, width=None, height=None):
        resize_algorithms = {
            "NEAREST": Image.NEAREST,
            "BILINEAR": Image.BILINEAR,
            "BICUBIC": Image.BICUBIC,
            "LANCZOS": Image.LANCZOS
        }
        algorithm = resize_algorithms[resize_algorithm]

        output_images = []
        for img in image:
            img = to_pil_image(img.permute(2, 0, 1))
            scale = self.calculate_scale(img, mode, maintain_aspect, scale, width, height)
            resized_img = self.shrink_image_with_scale(img, scale, algorithm)
            resized_img_np = np.array(resized_img).astype(np.float32) / 255.0
            resized_img_np = torch.from_numpy(resized_img_np)
            output_images.append(resized_img_np)

        return (output_images,)
