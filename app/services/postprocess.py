"""
postprocess.py
--------------
Pos-processamento final:
  - blend_result: integra resultado com original (delega para color_match)
  - create_side_by_side: monta compara??o antes/depois
  - add_labels: opcional, adiciona texto "ANTES" / "DEPOIS"
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

from app.services.color_match import match_and_blend


def blend_result(
    original: Image.Image,
    result: Image.Image,
    mask_np: np.ndarray,
    procedure: str = "",
) -> Image.Image:
    """
    Integra result no original usando color matching + seamless clone.
    Se o procedimento n?o usa warp (eye_bags, skin_smooth), o result
    j? vem blendado corretamente ? s?o redimensiona se necess?rio.
    """
    if result.size != original.size:
        result = result.resize(original.size, Image.LANCZOS)

    # Se a m?scara for vazia ou procedimento j? processou internamente
    if mask_np is None or not np.any(mask_np > 10):
        return result

    return match_and_blend(original, result, mask_np, use_seamless=True, procedure=procedure)


def create_side_by_side(
    original: Image.Image,
    result: Image.Image,
    label: bool = True,
) -> Image.Image:
    """
    Cria imagem de compara??o lado a lado.
    Adiciona labels "ANTES" / "DEPOIS" se label=True.
    """
    w, h = original.size

    # Separador branco de 3px
    separator = Image.new("RGB", (3, h), (240, 240, 240))
    canvas = Image.new("RGB", (w * 2 + 3, h), (240, 240, 240))
    canvas.paste(original, (0, 0))
    canvas.paste(separator, (w, 0))
    canvas.paste(result, (w + 3, 0))

    if label:
        canvas = _add_labels(canvas, w, h)

    return canvas


def _add_labels(
    canvas: Image.Image,
    panel_width: int,
    panel_height: int,
) -> Image.Image:
    """Adiciona labels ANTES/DEPOIS com fundo semitransparente."""
    draw = ImageDraw.Draw(canvas)

    label_h = max(28, panel_height // 18)
    font_size = max(14, label_h - 8)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for text, x_start in [("ANTES", 0), ("DEPOIS", panel_width + 3)]:
        # Fundo semitransparente
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [x_start, panel_height - label_h, x_start + panel_width, panel_height],
            fill=(0, 0, 0, 120),
        )
        canvas = canvas.convert("RGBA")
        canvas = Image.alpha_composite(canvas, overlay).convert("RGB")

        # Texto
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_start + (panel_width - text_w) // 2
        text_y = panel_height - label_h + (label_h - (bbox[3] - bbox[1])) // 2
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return canvas