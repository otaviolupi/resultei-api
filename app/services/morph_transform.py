"""
morph_transform.py
------------------
Aplica transformações geométricas (warp) para simular procedimentos estéticos.
Cada procedimento tem sua própria função de warp baseada em landmarks.

Técnica central: Thin Plate Spline (TPS) via cv2.createThinPlateSplineShapeTransformer
para deformações suaves e naturais a partir de correspondências de pontos.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple

from app.services.face_detection import landmarks_to_pixels, PROCEDURE_LANDMARK_MAP


# ---------------------------------------------------------------------------
# TPS Warp engine
# ---------------------------------------------------------------------------

def _tps_warp(
    image_np: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Aplica Thin Plate Spline warp.
    src_pts e dst_pts: arrays (N, 2) float32.
    Retorna imagem warped com mesmo shape.
    """
    h, w = image_np.shape[:2]

    src = src_pts.reshape(1, -1, 2).astype(np.float32)
    dst = dst_pts.reshape(1, -1, 2).astype(np.float32)

    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps.estimateTransformation(dst, src, matches)

    warped = tps.warpImage(image_np)
    return warped


def _build_anchor_grid(h: int, w: int, step: int = 64) -> np.ndarray:
    """
    Cria grade de pontos de âncora nas bordas da imagem para evitar
    deformações indesejadas fora da região de interesse.
    """
    pts = []
    for y in range(0, h + 1, step):
        for x in [0, w]:
            pts.append([x, y])
    for x in range(0, w + 1, step):
        for y in [0, h]:
            pts.append([x, y])
    return np.array(pts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Utilidades de deslocamento
# ---------------------------------------------------------------------------

def _expand_points_outward(pts: np.ndarray, center: np.ndarray, factor: float) -> np.ndarray:
    """Move pontos para fora do centro por um fator multiplicativo."""
    vectors = pts - center
    return pts + vectors * factor


def _contract_points_inward(pts: np.ndarray, center: np.ndarray, factor: float) -> np.ndarray:
    """Move pontos em direção ao centro por um fator multiplicativo."""
    vectors = pts - center
    return pts - vectors * factor


def _translate_points(pts: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translada pontos por (dx, dy)."""
    return pts + np.array([dx, dy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Procedimentos individuais
# ---------------------------------------------------------------------------

def _lip_bell(x_norm: float) -> float:
    """Curva gaussiana: concentra volume no centro, comissuras fixas."""
    return float(np.exp(-5.0 * x_norm ** 2))


def _build_contour_map(pts_crop: np.ndarray, cw: int, kind: str) -> np.ndarray:
    """
    Interpola Y do contorno labial coluna a coluna.
    kind='upper' → Y mínimo (contorno de cima)
    kind='lower' → Y máximo (contorno de baixo)
    Preserva assimetrias e curvas individuais de cada boca.
    """
    from scipy.interpolate import interp1d

    xs = pts_crop[:, 0].astype(float)
    ys = pts_crop[:, 1].astype(float)

    # Ordenar por X para interpolação monotônica
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]

    # Remover duplicatas de X
    _, unique_idx = np.unique(xs.round(1), return_index=True)
    xs, ys = xs[unique_idx], ys[unique_idx]

    if len(xs) < 2:
        scalar_y = ys.min() if kind == 'upper' else ys.max()
        return np.full(cw, scalar_y, dtype=np.float32)

    f = interp1d(xs, ys, kind='cubic' if len(xs) >= 4 else 'linear',
                 bounds_error=False,
                 fill_value=(ys[0], ys[-1]))
    return f(np.arange(cw)).astype(np.float32)


def _add_lip_gloss(
    crop: np.ndarray,
    lip_pts_crop: np.ndarray,
    geo_mask_f: np.ndarray,
    intensity: float,
    lip_h: int,
    cw: int,
    ch: int,
) -> np.ndarray:
    """
    Adiciona brilho/gloss ao lábio preenchido simulando pele esticada.

    Três camadas:
      1. Saturação aumentada (lábio fica mais rosado/vermelho)
      2. Highlight especular no centro do lábio superior (ponto de luz)
      3. Suavização leve da textura (pele esticada fica mais lisa)
    """
    result = crop.astype(np.float32).copy()

    # ------------------------------------------------------------------
    # 1. Saturação — lábio fica mais vivo
    # ------------------------------------------------------------------
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat_boost = 1.0 + intensity * 0.35
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    # Aplicar saturação só dentro da máscara geométrica
    mask_3ch = np.stack([geo_mask_f] * 3, axis=-1)
    result = result * (1 - mask_3ch * intensity * 0.5) + saturated * (mask_3ch * intensity * 0.5)

    # ------------------------------------------------------------------
    # 2. Highlight especular — centro do lábio superior
    # Ponto de luz alongado horizontalmente, no terço superior do lábio
    # ------------------------------------------------------------------
    highlight_map = np.zeros((ch, cw), dtype=np.float32)

    # Centro do lábio superior = média dos pontos superiores
    upper_center_x = float(np.mean(lip_pts_crop[:, 0]))
    upper_center_y = float(np.min(lip_pts_crop[:, 1]))  # Y mais alto = contorno ext sup
    highlight_cy   = upper_center_y + lip_h * 0.28      # um pouco abaixo do topo

    # Gaussiana elíptica: larga em X, estreita em Y
    grid_x_h, grid_y_h = np.meshgrid(np.arange(cw, dtype=np.float32),
                                      np.arange(ch, dtype=np.float32))
    sigma_x = lip_h * 0.55
    sigma_y = lip_h * 0.09
    highlight_map = np.exp(
        -((grid_x_h - upper_center_x) ** 2 / (2 * sigma_x ** 2) +
          (grid_y_h - highlight_cy)   ** 2 / (2 * sigma_y ** 2))
    ).astype(np.float32)

    # Recortar pelo contorno geométrico
    highlight_map *= geo_mask_f

    # Intensidade do brilho: sutil a moderado
    highlight_strength = intensity * 55.0  # 0..55 em valor de pixel
    highlight_3ch = np.stack([highlight_map * highlight_strength] * 3, axis=-1)

    result = np.clip(result + highlight_3ch, 0, 255)

    # ------------------------------------------------------------------
    # 3. Suavização leve — pele esticada tem menos rugas
    # ------------------------------------------------------------------
    smooth = cv2.bilateralFilter(result.astype(np.uint8), d=5,
                                  sigmaColor=20, sigmaSpace=5).astype(np.float32)
    smooth_alpha = geo_mask_f * intensity * 0.40
    smooth_3ch   = np.stack([smooth_alpha] * 3, axis=-1)
    result = result * (1 - smooth_3ch) + smooth * smooth_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


def _warp_lip_filler(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Lip filler v6: flow adaptativo por contorno + gloss pós-processamento.

    Melhorias vs v5:
      - upper_profile usa interpolação cúbica (preserva arco de cupido real)
      - Amplitude do flow superior proporcional à distância LOCAL ao contorno
        (pontos altos do arco de cupido sobem menos, bordas sobem mais)
      - Gloss: saturação + highlight especular + suavização de textura
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w)

    # ------------------------------------------------------------------
    # 1. Índices e bounding box
    # ------------------------------------------------------------------
    upper_outer_idx = [185, 40, 39, 37, 0, 267, 269, 270, 409]
    lower_outer_idx = [324, 318, 402, 317, 14, 87, 178, 88, 95]
    seam_upper_idx  = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
    seam_lower_idx  = [375, 321, 405, 314, 17, 84, 181, 91, 146]
    all_lip_idx = upper_outer_idx + lower_outer_idx + seam_upper_idx + seam_lower_idx + [61, 291]

    lip_pts = all_pts[all_lip_idx]
    x_min, x_max = int(np.min(lip_pts[:, 0])), int(np.max(lip_pts[:, 0]))
    y_min, y_max = int(np.min(lip_pts[:, 1])), int(np.max(lip_pts[:, 1]))

    lip_h = max(y_max - y_min, 1)
    lip_w = max(x_max - x_min, 1)

    pad_x     = int(lip_w * 0.55)
    pad_y_top = int(lip_h * 0.90)
    pad_y_bot = int(lip_h * 0.70)

    cx0 = max(0, x_min - pad_x)
    cx1 = min(w, x_max + pad_x)
    cy0 = max(0, y_min - pad_y_top)
    cy1 = min(h, y_max + pad_y_bot)

    crop = image_np[cy0:cy1, cx0:cx1].copy()
    ch, cw = crop.shape[:2]

    if ch < 10 or cw < 10:
        return image_np

    # ------------------------------------------------------------------
    # 2. Perfis de contorno coluna a coluna (coordenadas do crop)
    # ------------------------------------------------------------------
    offset = np.array([cx0, cy0], dtype=np.float32)

    upper_pts_crop = (all_pts[upper_outer_idx] - offset).astype(np.float32)
    lower_pts_crop = (all_pts[lower_outer_idx] - offset).astype(np.float32)
    seam_u_crop    = (all_pts[seam_upper_idx]  - offset).astype(np.float32)
    seam_l_crop    = (all_pts[seam_lower_idx]  - offset).astype(np.float32)
    seam_pts_crop  = np.vstack([seam_u_crop, seam_l_crop])

    upper_profile = np.clip(_build_contour_map(upper_pts_crop, cw, 'upper'), 0, ch - 1)
    lower_profile = np.clip(_build_contour_map(lower_pts_crop, cw, 'lower'), 0, ch - 1)
    seam_profile  = np.clip(_build_contour_map(seam_pts_crop,  cw, 'upper'), 0, ch - 1)

    # ------------------------------------------------------------------
    # 3. Grade + bell curve horizontal
    # ------------------------------------------------------------------
    grid_x, grid_y = np.meshgrid(
        np.arange(cw, dtype=np.float32),
        np.arange(ch, dtype=np.float32),
    )

    corner_left_x  = float(all_pts[61][0]  - cx0)
    corner_right_x = float(all_pts[291][0] - cx0)
    mouth_cx = (corner_left_x + corner_right_x) / 2.0
    mouth_w  = float(corner_right_x - corner_left_x) + 1e-6

    x_norm = np.clip((grid_x - mouth_cx) / (mouth_w / 2.0), -1.0, 1.0)
    bell_h = np.exp(-5.0 * x_norm ** 2).astype(np.float32)

    upper_2d = upper_profile[np.newaxis, :]
    lower_2d = lower_profile[np.newaxis, :]
    seam_2d  = seam_profile[np.newaxis, :]

    max_up   = lip_h * (0.06 + intensity * 0.16)
    max_down = lip_h * (0.04 + intensity * 0.10)

    # ------------------------------------------------------------------
    # 4. Dead zone no seam (zero exatamente na linha de contato)
    # ------------------------------------------------------------------
    dead_zone_sigma = lip_h * 0.08
    dist_to_seam = grid_y - seam_2d
    dead_zone = (1.0 - np.exp(-0.5 * (dist_to_seam / dead_zone_sigma) ** 2)).astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Flow SUPERIOR: amplitude LOCAL proporcional à espessura do lábio
    #    Espessura = seam_y - upper_y por coluna
    #    Colunas com lábio mais fino (arco de cupido) sobem MENOS
    #    Colunas com lábio mais grosso sobem MAIS
    #    → preserva a curva do arco de cupido
    # ------------------------------------------------------------------
    local_thickness_upper = np.clip(seam_2d - upper_2d, 1, None)  # (1, cw)
    max_thickness_upper   = float(local_thickness_upper.max()) + 1e-6
    thickness_ratio_upper = (local_thickness_upper / max_thickness_upper).astype(np.float32)

    dist_upper = grid_y - upper_2d
    sigma_upper = lip_h * 0.38
    bell_upper  = np.exp(-0.5 * (dist_upper / sigma_upper) ** 2).astype(np.float32)
    bell_upper *= dead_zone
    bell_upper *= (grid_y < seam_2d).astype(np.float32)
    bell_upper *= (grid_y > (upper_2d - lip_h * 0.80)).astype(np.float32)

    # Modular pela espessura local: arco de cupido (lábio mais fino) sobe menos
    flow_y_upper = +max_up * bell_h * bell_upper * thickness_ratio_upper

    # ------------------------------------------------------------------
    # 6. Flow INFERIOR
    # ------------------------------------------------------------------
    local_thickness_lower = np.clip(lower_2d - seam_2d, 1, None)
    max_thickness_lower   = float(local_thickness_lower.max()) + 1e-6
    thickness_ratio_lower = (local_thickness_lower / max_thickness_lower).astype(np.float32)

    dist_lower = grid_y - lower_2d
    sigma_lower = lip_h * 0.30
    bell_lower  = np.exp(-0.5 * (dist_lower / sigma_lower) ** 2).astype(np.float32)
    bell_lower *= dead_zone
    bell_lower *= (grid_y > seam_2d).astype(np.float32)
    bell_lower *= (grid_y < (lower_2d + lip_h * 0.55)).astype(np.float32)

    flow_y_lower = -max_down * bell_h * bell_lower * thickness_ratio_lower

    flow_y = (flow_y_upper + flow_y_lower).astype(np.float32)

    # ------------------------------------------------------------------
    # 7. Remap
    # ------------------------------------------------------------------
    map_x = np.clip(grid_x, 0, cw - 1).astype(np.float32)
    map_y = np.clip(grid_y + flow_y, 0, ch - 1).astype(np.float32)

    crop_warped = cv2.remap(
        crop, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # ------------------------------------------------------------------
    # 8. Máscara de blend
    # ------------------------------------------------------------------
    flow_mag = np.abs(flow_y)
    flow_max = flow_mag.max()
    if flow_max < 1e-6:
        return image_np

    alpha_flow = (flow_mag / flow_max).astype(np.float32)

    geo_mask = np.zeros((ch, cw), dtype=np.uint8)
    lip_pts_crop = (all_pts[all_lip_idx] - np.array([cx0, cy0])).astype(np.int32)
    lip_pts_crop[:, 0] = np.clip(lip_pts_crop[:, 0], 0, cw - 1)
    lip_pts_crop[:, 1] = np.clip(lip_pts_crop[:, 1], 0, ch - 1)
    cv2.fillConvexPoly(geo_mask, cv2.convexHull(lip_pts_crop), 255)
    geo_mask = cv2.dilate(geo_mask, np.ones((5, 5), np.uint8), iterations=4)
    geo_mask_f = geo_mask.astype(np.float32) / 255.0

    alpha = alpha_flow * geo_mask_f
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(2.0, lip_h * 0.10))
    alpha = np.clip(alpha, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 9. Alpha blend do warp
    # ------------------------------------------------------------------
    alpha_3ch = np.stack([alpha] * 3, axis=-1)
    blended_crop = (
        crop_warped.astype(np.float32) * alpha_3ch
        + crop.astype(np.float32) * (1.0 - alpha_3ch)
    ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 10. Gloss: saturação + highlight especular + suavização
    # ------------------------------------------------------------------
    blended_crop = _add_lip_gloss(
        blended_crop, lip_pts_crop, geo_mask_f, intensity, lip_h, cw, ch
    )

    # ------------------------------------------------------------------
    # 11. Recomposição
    # ------------------------------------------------------------------
    result = image_np.copy()
    result[cy0:cy1, cx0:cx1] = blended_crop
    return result



def _warp_nose_reshape(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Rhinoplasty: afina a asa do nariz, refina a ponta.
    - Asas contraem em direção à linha central
    - Ponta levemente levantada
    - Bridge afunilado
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w)

    scale = 0.01 + intensity * 0.045

    src_pts_list = []
    dst_pts_list = []

    # Centro vertical do nariz (linha de simetria)
    nose_center_x = all_pts[1][0]  # ponta do nariz X
    nose_top_y = all_pts[168][1]   # bridge Y

    # Asas do nariz — contraem horizontalmente em direção ao centro
    wing_left = all_pts[[98, 97, 99, 100, 45, 44]]
    wing_right = all_pts[[327, 326, 328, 329, 275, 274]]

    for pt in wing_left:
        dx = (nose_center_x - pt[0]) * scale * 0.6
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([dx, 0]))

    for pt in wing_right:
        dx = (nose_center_x - pt[0]) * scale * 0.6
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([dx, 0]))

    # Ponta do nariz: sobe levemente
    tip = all_pts[[1, 2, 5, 4]]
    for pt in tip:
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([0, -scale * h * 0.008]))

    # Bridge: afunila levemente
    bridge = all_pts[[168, 6, 197, 195]]
    for pt in bridge:
        dx = (nose_center_x - pt[0]) * scale * 0.3
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([dx, 0]))

    # Âncoras
    for idx in [33, 263, 61, 291, 152, 234, 454, 70, 300]:
        pt = all_pts[idx]
        src_pts_list.append(pt)
        dst_pts_list.append(pt)

    src_pts = np.array(src_pts_list, dtype=np.float32)
    dst_pts = np.array(dst_pts_list, dtype=np.float32)

    return _tps_warp(image_np, src_pts, dst_pts)


def _warp_jaw_slim(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Jaw slimming: contrai lateralmente a mandíbula.
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w)

    scale = 0.008 + intensity * 0.04

    face_center_x = all_pts[1][0]  # nariz como referência central

    src_pts_list = []
    dst_pts_list = []

    # Mandíbula esquerda
    jaw_left = all_pts[[172, 136, 150, 149, 176, 132, 93, 58]]
    for pt in jaw_left:
        dx = (face_center_x - pt[0]) * scale * 0.5
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([dx, 0]))

    # Mandíbula direita
    jaw_right = all_pts[[397, 365, 379, 378, 400, 361, 323, 288]]
    for pt in jaw_right:
        dx = (face_center_x - pt[0]) * scale * 0.5
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([dx, 0]))

    # Queixo central — mantém posição
    chin = all_pts[[152, 148, 377, 175]]
    for pt in chin:
        src_pts_list.append(pt)
        dst_pts_list.append(pt)

    # Âncoras superiores
    for idx in [10, 338, 297, 332, 284, 251, 389,
                 10, 109,  67,  103, 54, 21, 162]:
        pt = all_pts[idx]
        src_pts_list.append(pt)
        dst_pts_list.append(pt)

    src_pts = np.array(src_pts_list, dtype=np.float32)
    dst_pts = np.array(dst_pts_list, dtype=np.float32)

    return _tps_warp(image_np, src_pts, dst_pts)


def _warp_chin_augment(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Chin augmentation: projeta o queixo para baixo e levemente para frente.
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w)

    scale = 0.008 + intensity * 0.035

    src_pts_list = []
    dst_pts_list = []

    # Queixo central — desce
    chin_center = all_pts[[152, 175, 199, 200]]
    for pt in chin_center:
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([0, scale * h * 0.015]))

    # Queixo lateral — segue parcialmente
    chin_sides = all_pts[[148, 377, 176, 400]]
    for pt in chin_sides:
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([0, scale * h * 0.008]))

    # Mandíbula — âncoras, não mexe
    for idx in [136, 150, 149, 379, 365, 378, 172, 397]:
        pt = all_pts[idx]
        src_pts_list.append(pt)
        dst_pts_list.append(pt)

    # Âncoras faciais
    for idx in [1, 33, 263, 61, 291, 234, 454]:
        pt = all_pts[idx]
        src_pts_list.append(pt)
        dst_pts_list.append(pt)

    src_pts = np.array(src_pts_list, dtype=np.float32)
    dst_pts = np.array(dst_pts_list, dtype=np.float32)

    return _tps_warp(image_np, src_pts, dst_pts)


def _warp_brow_lift(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Brow lift: eleva as sobrancelhas uniformemente.
    Região da testa estica sutilmente para acompanhar.
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w)

    scale = 0.006 + intensity * 0.025

    src_pts_list = []
    dst_pts_list = []

    # Sobrancelha esquerda
    brow_left = all_pts[[70, 63, 105, 66, 107, 55, 65, 52, 53, 46]]
    for pt in brow_left:
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([0, -scale * h * 0.012]))

    # Sobrancelha direita
    brow_right = all_pts[[296, 334, 293, 300, 276, 283, 282, 295, 285]]
    for pt in brow_right:
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([0, -scale * h * 0.012]))

    # Arcada supraorbital (âncora leve — acompanha um pouco)
    orbit_left = all_pts[[33, 246, 161, 160, 159]]
    orbit_right = all_pts[[362, 398, 384, 385, 386]]
    for pt in list(orbit_left) + list(orbit_right):
        src_pts_list.append(pt)
        dst_pts_list.append(pt + np.array([0, -scale * h * 0.004]))

    # Âncoras fixas: nariz, boca, mandíbula
    for idx in [1, 2, 5, 61, 291, 152, 234, 454, 168]:
        pt = all_pts[idx]
        src_pts_list.append(pt)
        dst_pts_list.append(pt)

    src_pts = np.array(src_pts_list, dtype=np.float32)
    dst_pts = np.array(dst_pts_list, dtype=np.float32)

    return _tps_warp(image_np, src_pts, dst_pts)


def _warp_cheek_filler(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Cheek filler: expande região malar/zigomática para fora.
    Cria volume nas maçãs do rosto.
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w)

    scale = 0.008 + intensity * 0.04

    face_center = np.array([all_pts[1][0], all_pts[1][1]])  # nariz como centro

    src_pts_list = []
    dst_pts_list = []

    # Bochecha esquerda — expande para fora (esquerda e levemente para cima)
    cheek_left = all_pts[[116, 123, 147, 213, 192, 214, 210, 50, 118, 117]]
    cheek_left_center = np.mean(cheek_left, axis=0)
    for pt in cheek_left:
        vec = pt - cheek_left_center
        src_pts_list.append(pt)
        dst_pts_list.append(pt + vec * scale * 0.8)

    # Bochecha direita — expande para fora
    cheek_right = all_pts[[345, 352, 376, 433, 416, 434, 430, 280, 347, 346]]
    cheek_right_center = np.mean(cheek_right, axis=0)
    for pt in cheek_right:
        vec = pt - cheek_right_center
        src_pts_list.append(pt)
        dst_pts_list.append(pt + vec * scale * 0.8)

    # Âncoras: olhos, nariz, boca, testa
    for idx in [33, 263, 1, 61, 291, 152, 10, 234, 454, 70, 300]:
        pt = all_pts[idx]
        src_pts_list.append(pt)
        dst_pts_list.append(pt)

    src_pts = np.array(src_pts_list, dtype=np.float32)
    dst_pts = np.array(dst_pts_list, dtype=np.float32)

    return _tps_warp(image_np, src_pts, dst_pts)


# ---------------------------------------------------------------------------
# Procedimentos sem warp (color/texture based)
# ---------------------------------------------------------------------------

def _process_eye_bags(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Eye bag removal: clareia e suaviza região sob os olhos.
    Combina:
      - Clarear pixels escuros (olheiras)
      - Suavização seletiva de textura
      - Blend gradiente com original
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w).astype(np.int32)

    result = image_np.copy().astype(np.float32)

    # Construir máscaras para cada olho
    for eye_indices in [
        [33, 7, 163, 144, 145, 153, 154, 155, 133, 119, 120, 121, 128],
        [362, 382, 381, 380, 374, 373, 390, 249, 263, 348, 349, 350, 357],
    ]:
        pts = all_pts[eye_indices]
        # Deslocar máscara levemente para baixo para pegar região da olheira
        pts_shifted = pts.copy()
        pts_shifted[:, 1] += int(h * 0.02)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts_shifted), 255)
        # Dilatar para cobrir bem a olheira
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
        mask_blur = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0

        # LAB colorspace para clarear
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]

        # Calcula valor médio da pele ao redor para referência
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_pts = all_pts[[234, 454, 10, 152]]
        cv2.fillConvexPoly(face_mask, cv2.convexHull(face_pts), 255)
        face_mean_l = np.mean(l_channel[face_mask > 0]) if np.any(face_mask > 0) else 128.0

        # Clarear: puxa L em direção à média da pele
        brighten_amount = intensity * 18.0
        l_brightened = l_channel + (face_mean_l - l_channel) * (mask_blur * intensity * 0.6)
        l_brightened = np.clip(l_brightened, 0, 255)

        lab_result = lab.copy()
        lab_result[:, :, 0] = l_brightened

        brightened = cv2.cvtColor(lab_result.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

        # Suavizar textura na região
        smooth = cv2.bilateralFilter(image_np, d=9, sigmaColor=75, sigmaSpace=75).astype(np.float32)

        # Blend: brightened para cor, smooth para textura
        texture_blend = 0.3 * intensity
        combined = brightened * (1 - texture_blend) + smooth * texture_blend

        # Aplicar com máscara gradiente
        mask_3ch = np.stack([mask_blur] * 3, axis=-1)
        result = result * (1 - mask_3ch) + combined * mask_3ch

    return np.clip(result, 0, 255).astype(np.uint8)


def _process_skin_smooth(
    image_np: np.ndarray,
    landmarks,
    intensity: float,
) -> np.ndarray:
    """
    Skin smoothing: suavização seletiva preservando bordas (olhos, lábios, sobrancelhas).
    Usa bilateral filter em múltiplas iterações + blend com original.
    """
    h, w = image_np.shape[:2]
    all_pts = landmarks_to_pixels(landmarks, h, w).astype(np.int32)

    # Máscara do rosto inteiro
    face_pts = all_pts[[
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    ]]
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, cv2.convexHull(face_pts), 255)

    # Máscara de exclusão: olhos, sobrancelhas, lábios (preservar detalhes)
    exclude_mask = np.zeros((h, w), dtype=np.uint8)
    for region_indices in [
        [33, 133, 159, 145],           # olho esquerdo
        [362, 263, 386, 374],          # olho direito
        [70, 107, 55, 46],             # sobrancelha esq
        [300, 276, 285, 296],          # sobrancelha dir
        [61, 291, 0, 17],              # lábios
    ]:
        region_pts = all_pts[region_indices]
        cv2.fillConvexPoly(exclude_mask, cv2.convexHull(region_pts), 255)

    # Dilatar exclusão
    exclude_mask = cv2.dilate(exclude_mask, np.ones((7, 7), np.uint8), iterations=2)

    # Máscara final = rosto - exclusões
    smooth_mask = np.clip(face_mask.astype(int) - exclude_mask.astype(int), 0, 255).astype(np.uint8)
    smooth_mask = cv2.GaussianBlur(smooth_mask, (21, 21), 0).astype(np.float32) / 255.0

    # Bilateral filter iterativo
    sigma_color = 30 + int(intensity * 50)
    sigma_space = 15 + int(intensity * 25)
    n_iters = 1 + int(intensity * 3)

    smoothed = image_np.copy()
    for _ in range(n_iters):
        smoothed = cv2.bilateralFilter(smoothed, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Blend
    mask_3ch = np.stack([smooth_mask] * 3, axis=-1)
    result = (image_np.astype(np.float32) * (1 - mask_3ch) +
              smoothed.astype(np.float32) * mask_3ch)

    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dispatcher principal
# ---------------------------------------------------------------------------

WARP_PROCEDURES = {
    "lip_filler",
    "nose_reshape",
    "jaw_slim",
    "chin_augment",
    "brow_lift",
    "cheek_filler",
}

TEXTURE_PROCEDURES = {
    "eye_bags",
    "skin_smooth",
}


def apply_procedure(
    image_pil: Image.Image,
    landmarks,
    procedure: str,
    intensity: float,
) -> Image.Image:
    """
    Entry point: aplica o procedimento correto à imagem.
    Retorna PIL Image com o resultado.
    """
    image_np = np.array(image_pil.convert("RGB"))

    if procedure == "lip_filler":
        result_np = _warp_lip_filler(image_np, landmarks, intensity)
    elif procedure == "nose_reshape":
        result_np = _warp_nose_reshape(image_np, landmarks, intensity)
    elif procedure == "jaw_slim":
        result_np = _warp_jaw_slim(image_np, landmarks, intensity)
    elif procedure == "chin_augment":
        result_np = _warp_chin_augment(image_np, landmarks, intensity)
    elif procedure == "brow_lift":
        result_np = _warp_brow_lift(image_np, landmarks, intensity)
    elif procedure == "cheek_filler":
        result_np = _warp_cheek_filler(image_np, landmarks, intensity)
    elif procedure == "eye_bags":
        result_np = _process_eye_bags(image_np, landmarks, intensity)
    elif procedure == "skin_smooth":
        result_np = _process_skin_smooth(image_np, landmarks, intensity)
    else:
        raise ValueError(f"Procedimento desconhecido: {procedure}")

    return Image.fromarray(result_np.astype(np.uint8))