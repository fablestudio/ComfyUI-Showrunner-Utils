from .align_face_node import AlignFace
from .generate_timestamp import GenerateTimestamp
from .render_open_street_view_tile import RenderOpenStreetMapTile

# from .read_image_node import ReadImage


NODE_CLASS_MAPPINGS = {
    "AlignFace": AlignFace,
    "GenerateTimestamp": GenerateTimestamp,
    # "ReadImage": ReadImage,
    "RenderOpenStreetMapTile": RenderOpenStreetMapTile,
}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "AlignFace": "Align Face",
    "GenerateTimestamp": "Generate Timestamp",
    # "ReadImageTest": "Read Image",
    "RenderOpenStreetMapTile": "Render OpenStreetMap Tile",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
