from .align_face_node import AlignFace
from .generate_timestamp import GenerateTimestamp
from .most_common_colors import GetMostCommonColors
from .images import SR_AlphaCropAndPositionImage, SR_ShrinkImage, SR_PadMask, SR_LoadImageFromUrl
from .openai import SR_Image2Text
from .image_composite import SR_ImageCompositeAbsoluteByContainer
from .images import SR_AdjustBottomAlphaDistance
from .masks import SR_MaskMorphologyNode, SR_OutlineMask
from .textutils import SR_ShowrunnerFilename
from .seed import SR_Seed

NODE_CLASS_MAPPINGS = {
    "AlignFace": AlignFace,
    "GenerateTimestamp": GenerateTimestamp,
    "GetMostCommonColors": GetMostCommonColors,
    "Alpha Crop and Position Image": SR_AlphaCropAndPositionImage,
    "Shrink Image": SR_ShrinkImage,    # "ReadImage": ReadImage,
    "PadMask": SR_PadMask,
    "OpenAI Image 2 Text": SR_Image2Text,
    "ImageCompositeAbsoluteByContainer": SR_ImageCompositeAbsoluteByContainer,
    "LoadImageFromUrl": SR_LoadImageFromUrl,
    "AdjustBottomAlphaDistance": SR_AdjustBottomAlphaDistance,
    "MaskMorphologyNode": SR_MaskMorphologyNode,
    "ShowrunnerFilename": SR_ShowrunnerFilename,
    "ShowrunnerSeed": SR_Seed,
    "CreateOutlineMask": SR_OutlineMask,

}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "AlignFace": "Align Face",
    "GenerateTimestamp": "Generate Timestamp",
    "GetMostCommonColors": "Get Most Common Image Colors",
    "Alpha Crop and Position Image": "Alpha Crop and Position Image",
    "Shrink Image": "Shrink Image",
    "PadMask": "Pad Mask",
    "OpenAI Image 2 Text": "OpenAI Image 2 Text",
    "ImageCompositeAbsoluteByContainer": "Image Composite Absolute By Container",
    "LoadImageFromUrl": "Load Image From URL",
    "AdjustBottomAlphaDistance": "Adjust Bottom Alpha Distance",
    "MaskMorphologyNode": "Mask Morphology Node",
    "ShowrunnerFilename": "Showrunner Filename",
    "ShowrunnerSeed": "Seed",
    "CreateOutlineMask": "Create Outline Mask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
