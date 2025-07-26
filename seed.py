#Code from https://github.com/WASasquatch/was-node-suite-comfyui

class SR_Seed:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"seed": ("INT", {"default": 0, "min": 0,
                          "max": 0xffffffffffffffff})}
                }

    RETURN_TYPES = ("SEED", "NUMBER", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("seed", "number", "float", "int", "string")
    FUNCTION = "seed"

    CATEGORY = "Showrunner Nodes"

    def seed(self, seed):
        return ({"seed": seed, }, seed, float(seed), int(seed), str(seed))

