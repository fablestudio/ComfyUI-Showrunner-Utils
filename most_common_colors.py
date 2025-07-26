import torch


class GetMostCommonColors:
    """
    Extracts the most common colors from an image or batch of images.

    For single images: Returns the most common colors from that image.
    For batched images: Returns the most common colors for each image individually,
    with RGB results separated by newlines (one line per image) and hex colors
    concatenated using the specified delimiter.

    The hex_delimiter parameter allows customization of how hex colors are joined:
    - "-" for dash separation (default)
    - "_" for underscore separation
    - "\\n" for newline separation
    - " " for space separation
    - Any other string as needed

    Supports both single images and batched images from nodes like CR_BatchProcessSwitch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "num_colors": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Number of colors to detect",
                    },
                ),
                "exclude_colors": (
                    "STRING",
                    {
                        "default": "#000000,#FFFFFF",
                        "tooltip": "Comma-separated list of colors to exclude from the output",
                    },
                ),
                "hex_delimiter": (
                    "STRING",
                    {
                        "default": "-",
                        "tooltip": "Delimiter to separate hex colors (e.g., '-', '_', '\\n', ' ', etc.)",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = ("rgb_colors", "hex_colors")
    FUNCTION = "main"
    CATEGORY = "Showrunner Nodes"

    def main(
        self,
        input_image: torch.Tensor,
        num_colors: int = 5,
        exclude_colors: str = "",
        hex_delimiter: str = "-",
    ) -> tuple[str, ...]:
        # Process exclude colors
        if exclude_colors.strip():
            exclude_list = [
                color.strip().lower() for color in exclude_colors.strip().split(",")
            ]
        else:
            exclude_list = []

        # Handle escape sequences in delimiter
        delimiter = hex_delimiter.replace("\\n", "\n").replace("\\t", "\t")

        # Handle both single images and batched images
        # input_image shape: [batch, height, width, channels] or [height, width, channels]
        if len(input_image.shape) == 4:  # Batched images
            batch_size = input_image.shape[0]
            all_rgb_colors = []
            all_hex_colors = []

            # Process each image in the batch individually
            for i in range(batch_size):
                single_image = input_image[i]  # Get single image from batch
                rgb_colors, hex_colors = self._process_single_image(
                    single_image, num_colors, exclude_list, delimiter
                )
                all_rgb_colors.append(rgb_colors)
                all_hex_colors.append(hex_colors)

            # Join results with newlines to separate each image's results
            return (
                "\n".join(all_rgb_colors),
                delimiter.join(
                    [color for colors in all_hex_colors for color in colors.split(", ")]
                ),
            )
        else:  # Single image
            return self._process_single_image(
                input_image, num_colors, exclude_list, delimiter
            )

    def _process_single_image(
        self, image: torch.Tensor, num_colors: int, exclude_list: list, delimiter: str
    ) -> tuple[str, str]:
        """Process a single image and return its most common colors."""
        # Convert image to pixels
        pixels = image.reshape(-1, image.shape[-1]).numpy()
        pixels = (pixels * 255).astype(int)  # Scale to 0-255 and convert to integers

        # Create color strings and count them
        color_counts = {}
        for pixel in pixels:
            if pixel.shape[0] == 3:  # RGB image
                r, g, b = pixel
            else:  # RGBA image
                r, g, b, _ = pixel  # Ignore alpha channel
            rgb_str = f"rgb({r}, {g}, {b})"
            hex_str = f"#{r:02x}{g:02x}{b:02x}"

            # Skip if this color should be excluded
            if hex_str.lower() in exclude_list or rgb_str.lower() in exclude_list:
                continue

            if (rgb_str, hex_str) in color_counts:
                color_counts[(rgb_str, hex_str)] += 1
            else:
                color_counts[(rgb_str, hex_str)] = 1

        # Sort by frequency and take top num_colors
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
        top_colors = sorted_colors[:num_colors]

        # Separate RGB and hex colors
        rgb_colors = []
        hex_colors = []
        for (rgb, hex_color), _ in top_colors:
            rgb_colors.append(rgb)
            hex_colors.append(hex_color)

        return (", ".join(rgb_colors), delimiter.join(hex_colors))
