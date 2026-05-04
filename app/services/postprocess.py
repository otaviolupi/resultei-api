import numpy as np
from PIL import Image
import cv2


def blend_result(original: Image.Image, result: Image.Image,
                 mask_np: np.ndarray) -> Image.Image:
    # Blending já foi feito no inpainting — só redimensiona de volta
    if result.size != original.size:
        result = result.resize(original.size, Image.LANCZOS)
    return result


def create_side_by_side(original: Image.Image, result: Image.Image) -> Image.Image:
    w, h = original.size
    canvas = Image.new("RGB", (w * 2 + 4, h), (200, 200, 200))
    canvas.paste(original, (0, 0))
    canvas.paste(result, (w + 4, 0))
    return canvas


def _mask_center(mask_binary: np.ndarray):
    moments = cv2.moments(mask_binary)
    if moments["m00"] == 0:
        h, w = mask_binary.shape
        return (w // 2, h // 2)
    return (int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]))