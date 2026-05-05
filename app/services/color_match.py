"""
color_match.py
--------------
Corrige disparidades de cor após warp geométrico.

Técnicas:
  1. Reinhard color transfer — combina estatísticas de cor da região modificada
     com a pele original adjacente, evitando artefatos de saturação/brilho.
  2. Seamless clone (Poisson blending) — integra a região warped de volta à
     imagem original com bordas invisíveis.
  3. Histogram matching local — garante que a textura de pele na região
     modificada mantenha a mesma distribuição tonal da original.
"""

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Reinhard color transfer (LAB space)
# ---------------------------------------------------------------------------

def _reinhard_transfer(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Transfere as estatísticas de cor (mean/std) do target para o source
    apenas na região definida por mask.

    source, target: RGB uint8 (H, W, 3)
    mask: uint8 (H, W), 0-255
    Retorna source corrigido.
    """
    mask_bool = mask > 10

    if not np.any(mask_bool):
        return source

    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

    result_lab = source_lab.copy()

    for ch in range(3):
        src_ch = source_lab[:, :, ch]
        tgt_ch = target_lab[:, :, ch]

        # Estatísticas apenas na região de interesse
        src_mean = np.mean(src_ch[mask_bool])
        src_std  = np.std(src_ch[mask_bool])  + 1e-6
        tgt_mean = np.mean(tgt_ch[mask_bool])
        tgt_std  = np.std(tgt_ch[mask_bool])  + 1e-6

        # Normaliza source e reescala para estatísticas do target
        corrected = (src_ch - src_mean) * (tgt_std / src_std) + tgt_mean

        # Aplica apenas dentro da máscara (blend gradiente)
        alpha = mask.astype(np.float32) / 255.0
        result_lab[:, :, ch] = src_ch * (1 - alpha) + corrected * alpha

    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Poisson / Seamless cloning
# ---------------------------------------------------------------------------

def _seamless_clone(
    source: np.ndarray,
    destination: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Usa cv2.seamlessClone para integrar source em destination dentro de mask.
    Retorna imagem clonada.
    """
    mask_bin = (mask > 30).astype(np.uint8) * 255

    if not np.any(mask_bin):
        return destination

    # Centro da máscara
    moments = cv2.moments(mask_bin)
    if moments["m00"] == 0:
        h, w = mask_bin.shape
        center = (w // 2, h // 2)
    else:
        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )

    try:
        src_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(destination, cv2.COLOR_RGB2BGR)
        result_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask_bin, center, cv2.NORMAL_CLONE)
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    except cv2.error:
        # Fallback: alpha blend simples
        alpha = mask.astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha] * 3, axis=-1)
        blended = source.astype(np.float32) * alpha_3ch + destination.astype(np.float32) * (1 - alpha_3ch)
        return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def _alpha_blend_soft(
    source: np.ndarray,
    destination: np.ndarray,
    mask: np.ndarray,
    feather: int = 15,
) -> np.ndarray:
    if feather > 0:
        kernel_size = feather * 2 + 1
        mask_soft = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    else:
        mask_soft = mask
    alpha = mask_soft.astype(np.float32) / 255.0
    alpha_3ch = np.stack([alpha] * 3, axis=-1)
    blended = (
        source.astype(np.float32) * alpha_3ch
        + destination.astype(np.float32) * (1 - alpha_3ch)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def match_and_blend(
    original: Image.Image,
    warped: Image.Image,
    mask_np: np.ndarray,
    use_seamless: bool = True,
    procedure: str = "",
) -> Image.Image:
    """
    Combina warped com original usando Reinhard + seamless clone.
    Para lip_filler: alpha blend feathered para preservar cor dos labios.
    """
    orig_np = np.array(original.convert("RGB"))
    warp_np = np.array(warped.convert("RGB"))

    if procedure == "lip_filler":
        result_np = _alpha_blend_soft(warp_np, orig_np, mask_np, feather=18)
        return Image.fromarray(result_np)

    color_corrected = _reinhard_transfer(warp_np, orig_np, mask_np)

    if use_seamless:
        result_np = _seamless_clone(color_corrected, orig_np, mask_np)
    else:
        result_np = _alpha_blend_soft(color_corrected, orig_np, mask_np)

    return Image.fromarray(result_np)