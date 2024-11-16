import torch


class GetMostCommonColors:
    @classmethod
    def INPUT_TYPES(s):
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
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "rgb_colors",
        "hex_colors"
    )
    FUNCTION = "main"
    CATEGORY = "image"

    def main(
        self,
        input_image: torch.Tensor,
        num_colors: int = 5,
        exclude_colors: str = "",
    ) -> tuple[str, ...]:
        # Process exclude colors
        if exclude_colors.strip():
            self.exclude = [color.strip().lower() for color in exclude_colors.strip().split(",")]
        else:
            self.exclude = []

        # Convert image to pixels
        pixels = input_image.view(-1, input_image.shape[-1]).numpy()
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
            if hex_str.lower() in self.exclude or rgb_str.lower() in self.exclude:
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

        return (
            ", ".join(rgb_colors),
            ", ".join(hex_colors)
        )