from .align_face_node import AlignFace
from .generate_timestamp import GenerateTimestamp
from .most_common_colors import GetMostCommonColors
from .images import SR_AlphaCropAndPositionImage, SR_ShrinkImage, SR_PadMask
from .openai import SR_Image2Text

# from .read_image_node import ReadImage
# from .render_open_street_view_tile import RenderOpenStreetMapTile


NODE_CLASS_MAPPINGS = {
    "AlignFace": AlignFace,
    "GenerateTimestamp": GenerateTimestamp,
    "GetMostCommonColors": GetMostCommonColors,
    "Alpha Crop and Position Image": SR_AlphaCropAndPositionImage,
    "Shrink Image": SR_ShrinkImage,    # "ReadImage": ReadImage,
    "PadMask": SR_PadMask,
    "OpenAI Image 2 Text": SR_Image2Text,
    # "RenderOpenStreetMapTile": RenderOpenStreetMapTile,
}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "AlignFace": "Align Face",
    "GenerateTimestamp": "Generate Timestamp",
    "GetMostCommonColors": "Get Most Common Image Colors",
    "Alpha Crop and Position Image": "Alpha Crop and Position Image",
    "Shrink Image": "Shrink Image",
    "PadMask": "Pad Mask",
    "OpenAI Image 2 Text": "OpenAI Image 2 Text",
    # "ReadImageTest": "Read Image",
    # "RenderOpenStreetMapTile": "Render OpenStreetMap Tile",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
