from .align_face_node import AlignFace
from .generate_timestamp import GenerateTimestamp
from .most_common_colors import GetMostCommonColors

# from .read_image_node import ReadImage
# from .render_open_street_view_tile import RenderOpenStreetMapTile


NODE_CLASS_MAPPINGS = {
    "AlignFace": AlignFace,
    "GenerateTimestamp": GenerateTimestamp,
    "GetMostCommonColors": GetMostCommonColors,
    # "ReadImage": ReadImage,
    # "RenderOpenStreetMapTile": RenderOpenStreetMapTile,
}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "AlignFace": "Align Face",
    "GenerateTimestamp": "Generate Timestamp",
    "GetMostCommonColors": "Get Most Common Image Colors",
    # "ReadImageTest": "Read Image",
    # "RenderOpenStreetMapTile": "Render OpenStreetMap Tile",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
