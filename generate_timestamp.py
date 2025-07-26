# custom_comfyui_nodes/nodes/generate_timestamp.py

from datetime import datetime


class GenerateTimestamp:
    """
    A node that generates the current system timestamp.

    You can optionally provide a custom format string. If no format is provided,
    the default format 'YYYYMMDD-HHMMSS' is used.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "format_str": (
                    "STRING",
                    {
                        "default": "%Y%m%d-%H%M%S",
                        "help": "Custom datetime format string. Defaults to '%Y%m%d-%H%M%S'.",
                    },
                )
            }
        }

    CATEGORY = "Showrunner Nodes"

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_timestamp"

    def generate_timestamp(self, format_str="%Y%m%d-%H%M%S"):
        """
        Generates the current system timestamp using the provided format string.
        If no format is provided, uses the default format 'YYYYMMDD-HHMMSS'.

        Args:
            format_str (str): The format string for the timestamp.

        Returns:
            tuple: A tuple containing the formatted timestamp string.
        """
        try:
            # Get the current system time
            current_time = datetime.now()
            # Format the timestamp as a string based on the provided format
            timestamp_str = current_time.strftime(format_str)
            # Send the timestamp to the output
            return (timestamp_str,)
        except Exception as e:
            print(f"Error generating timestamp: {e}")
            return ("",)  # Return an empty string in case of error
