"""
Multiply Sigmas node adapted from comfyui-detail-daemon.
"""

def _clamp_percent(value):
    return max(0.0, min(1.0, float(value)))


def _normalize_percent_window(start, end):
    start = _clamp_percent(start)
    end = _clamp_percent(end)
    return min(start, end), max(start, end)


class MultiplySigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "factor": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 100,
                        "step": 0.001,
                        "tooltip": (
                            "Multiplies sampling-step sigmas by this factor. "
                            "Values below 1.0 generally increase detail, but can "
                            "alter composition or add noisy grain."
                        ),
                    },
                ),
                "start": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001}),
                "end": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 1,
                        "step": 0.001,
                        "tooltip": "Percent window over sampling steps. The final zero sigma is preserved.",
                    },
                ),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    def simple_output(self, sigmas, factor, start, end):
        sigmas = sigmas.clone()

        steps = max(0, len(sigmas) - 1)
        if steps == 0:
            return (sigmas,)

        start, end = _normalize_percent_window(start, end)
        start_idx = int(start * steps)
        end_idx = int(end * steps)

        sigmas[start_idx:end_idx] *= factor

        return (sigmas,)
