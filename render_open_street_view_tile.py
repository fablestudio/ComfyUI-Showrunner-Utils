import os
import math
import requests
from io import BytesIO
from PIL import Image
import logging
import numpy as np
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RenderOpenStreetMapTile:
    """
    A ComfyUI custom node that fetches and outputs OpenStreetMap tiles based on a center coordinate and specified zoom level.
    """

    # Define the cache directory relative to the script's location
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "osm_tile_cache")

    TILE_SIZE = 256  # OSM tiles are 256x256 pixels

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "center_lat*1000": (
                    "INT",
                    {"default": 37774, "min": -90000, "max": 90000},
                ),  # Latitude multiplied by 1000
                "center_lon*1000": (
                    "INT",
                    {"default": -122419, "min": -180000, "max": 180000},
                ),  # Longitude multiplied by 1000
                "zoom": ("INT", {"default": 12, "min": 0, "max": 19}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_osm_tile"
    CATEGORY = "map/tile"

    def __init__(self):
        # Ensure the cache directory exists
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        self.logger = logger

    def render_osm_tile(self, center_lat, center_lon, zoom):
        """
        Fetches OSM tiles based on center coordinates and outputs the image.
        """
        # Convert center_lat and center_lon from INT to float
        center_lat = center_lat / 1000.0
        center_lon = center_lon / 1000.0

        # Get fractional tile coordinates
        x_tile, y_tile = self.latlon_to_tile(center_lat, center_lon, zoom)

        # Compute integer tile indices
        x0 = int(math.floor(x_tile))
        y0 = int(math.floor(y_tile))

        # Fetch tiles from x0 -1 to x0 +1 (inclusive)
        tiles = {}
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x = x0 + dx
                y = y0 + dy
                # Handle wrapping of tiles at zoom level (for x)
                max_tile = 2**zoom
                x = x % max_tile
                if y < 0 or y >= max_tile:
                    # Tiles out of bounds in y; create blank tile
                    tile_image = Image.new(
                        "RGB", (self.TILE_SIZE, self.TILE_SIZE), (255, 255, 255)
                    )
                else:
                    # Fetch the tile
                    tile_path = self.fetch_tile(zoom, x, y)
                    if tile_path:
                        tile_image = Image.open(tile_path)
                    else:
                        # If tile fetch fails, create a blank tile
                        tile_image = Image.new(
                            "RGB", (self.TILE_SIZE, self.TILE_SIZE), (255, 255, 255)
                        )
                tiles[(dx, dy)] = tile_image

        # Create a new image to hold the combined tiles
        combined_image = Image.new("RGB", (self.TILE_SIZE * 3, self.TILE_SIZE * 3))

        # Paste tiles into the combined image
        for (dx, dy), tile_image in tiles.items():
            x_offset = (dx + 1) * self.TILE_SIZE
            y_offset = (dy + 1) * self.TILE_SIZE
            combined_image.paste(tile_image, (x_offset, y_offset))

        # Compute the pixel coordinates of the center point
        x_pixel = (x_tile - (x0 - 1)) * self.TILE_SIZE
        y_pixel = (y_tile - (y0 - 1)) * self.TILE_SIZE

        # Crop a 512x512 px area centered at (x_pixel, y_pixel)
        left = int(x_pixel - 256)
        upper = int(y_pixel - 256)
        right = left + 512
        lower = upper + 512

        cropped_image = combined_image.crop((left, upper, right, lower))

        image = cropped_image.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        if "A" in cropped_image.getbands():
            mask = np.array(cropped_image.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return (image, mask)

    def fetch_tile(self, zoom, x, y):
        """
        Fetches a single OSM tile, using cache if available.
        """
        cache_filename = f"{zoom}_{x}_{y}.png"
        cache_path = os.path.join(self.CACHE_DIR, cache_filename)

        # Check if tile exists in cache
        if os.path.exists(cache_path):
            self.logger.info(f"Tile found in cache: {cache_path}")
            return cache_path

        # Else, fetch the tile from the server
        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
        self.logger.info(f"Fetching tile from URL: {url}")

        try:
            response = requests.get(
                url,
                timeout=10,
                headers={
                    "User-Agent": "ComfyUI-OSMTileNode/1.0 (your_email@example.com)"
                },
            )
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            # Save to cache
            image.save(cache_path)
            self.logger.info(f"Tile fetched and cached at: {cache_path}")
            return cache_path
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch tile {zoom}/{x}/{y}: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error while fetching tile {zoom}/{x}/{y}: {e}"
            )
            return None

    @staticmethod
    def latlon_to_tile(lat, lon, zoom):
        """
        Converts latitude and longitude to OSM tile coordinates.
        """
        lat_rad = math.radians(lat)
        n = 2.0**zoom
        x_tile = (lon + 180.0) / 360.0 * n
        y_tile = (
            (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
            / 2.0
            * n
        )
        return x_tile, y_tile
