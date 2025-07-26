import datetime

class SR_ShowrunnerFilename:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
                "string4": ("STRING", {"default": ""}),
                "string5": ("STRING", {"default": ""}),
            },
            "required": {
                "delimiter": ("STRING", {"default": "_"}),
                "clean_whitespace": (["True", "False"], {"default": "True"}),
                "prefix_timestamp": (["True", "False"], {"default": "True"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "concat"
    CATEGORY = "Showrunner Nodes"

    def concat(self, delimiter, clean_whitespace, prefix_timestamp, string1="", string2="", string3="", string4="", string5=""):
        strings = [string1, string2, string3, string4, string5]
        # Remove empty strings
        strings = [s for s in strings if s]
        # Optionally clean whitespace
        if clean_whitespace == "True":
            strings = [s.strip() for s in strings]
        result = delimiter.join(strings)
        # Optionally prefix timestamp
        if prefix_timestamp == "True":
            ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            result = f"{ts}{delimiter}{result}" if result else ts
        return (result,)
