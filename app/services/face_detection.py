"""
face_detection.py
-----------------
Detecta landmarks faciais via MediaPipe e retorna:
  - get_landmarks()      → lista de landmarks normalizados
  - landmarks_to_pixels() → array (N,2) em pixels
  - get_procedure_points() → pontos do procedimento em pixels
  - generate_mask()      → máscara binária suavizada
"""

import mediapipe as mp
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Mapeamento de landmarks por procedimento
# ---------------------------------------------------------------------------

PROCEDURE_LANDMARK_MAP = {
    "lip_filler": [
        # Lábio superior — borda externa
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        # Lábio inferior — borda externa
        308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
        # Interior
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        375, 321, 405, 314, 17, 84, 181, 91, 146,
    ],
    "nose_reshape": [
        1, 2, 5, 4, 6,
        98, 97, 99, 100, 101, 102, 45, 51, 44,
        327, 326, 328, 329, 330, 331, 275, 281, 274,
        168, 6, 197, 195,
        220, 115, 48, 64,
        440, 344, 278, 294,
    ],
    "jaw_slim": [
        172, 136, 150, 149, 176, 148, 152,
        377, 400, 378, 379, 365, 397, 288,
        132, 93, 58,
        361, 323,
    ],
    "chin_augment": [
        152, 148, 176, 149, 150, 136, 172,
        377, 400, 378, 379, 365, 397, 288,
        175, 199, 200, 201, 202,
    ],
    "brow_lift": [
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        296, 334, 293, 300, 276, 283, 282, 295, 285,
        33, 246, 161, 160, 159, 158, 157, 173,
        362, 398, 384, 385, 386, 387, 388, 466,
    ],
    "eye_bags": [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        362, 382, 381, 380, 374, 373, 390, 249, 263,
        119, 120, 121, 128, 245,
        348, 349, 350, 357, 465,
    ],
    "cheek_filler": [
        116, 123, 147, 213, 192, 214, 210, 211, 32, 36,
        345, 352, 376, 433, 416, 434, 430, 431, 262, 266,
        50, 101, 118, 117, 111,
        280, 330, 347, 346, 340,
    ],
    "skin_smooth": None,
}

VALID_PROCEDURES = list(PROCEDURE_LANDMARK_MAP.keys())


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------

def _run_mediapipe(image_np: np.ndarray):
    """Executa MediaPipe FaceMesh e retorna landmarks brutos."""
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0].landmark
    except AttributeError:
        pass

    # Fallback: nova API mediapipe >= 0.10
    import urllib.request, os
    model_path = "./mediapipe_face_landmarker.task"
    if not os.path.exists(model_path):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        print("Baixando modelo MediaPipe FaceLandmarker...")
        urllib.request.urlretrieve(url, model_path)

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect(mp_image)

    if result.face_landmarks:
        return result.face_landmarks[0]
    return None


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def get_landmarks(image_np: np.ndarray) -> list:
    """Retorna lista de landmarks MediaPipe normalizados. Lança ValueError se não detectar rosto."""
    landmarks = _run_mediapipe(image_np)
    if landmarks is None:
        raise ValueError(
            "Nenhum rosto detectado. Use uma foto frontal com boa iluminação."
        )
    return landmarks


def landmarks_to_pixels(landmarks, h: int, w: int) -> np.ndarray:
    """Converte landmarks normalizados para array (N, 2) em pixels (x, y)."""
    return np.array(
        [[lm.x * w, lm.y * h] for lm in landmarks],
        dtype=np.float32,
    )


def get_procedure_points(landmarks, procedure: str, h: int, w: int) -> np.ndarray:
    """Retorna array (N, 2) dos pontos relevantes ao procedimento em pixels."""
    all_pts = landmarks_to_pixels(landmarks, h, w)
    indices = PROCEDURE_LANDMARK_MAP.get(procedure)
    if indices is None:
        return all_pts
    return all_pts[indices]


def generate_mask(
    image_np: np.ndarray,
    procedure: str,
    landmarks=None,
    dilate_px: int = 15,
    blur_px: int = 21,
) -> np.ndarray:
    """Gera máscara binária suavizada para a região do procedimento."""
    if landmarks is None:
        landmarks = get_landmarks(image_np)

    h, w = image_np.shape[:2]
    indices = PROCEDURE_LANDMARK_MAP.get(procedure)
    mask = np.zeros((h, w), dtype=np.uint8)

    all_pts = landmarks_to_pixels(landmarks, h, w).astype(np.int32)
    pts = all_pts if indices is None else all_pts[indices]

    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
    if blur_px > 0:
        mask = cv2.GaussianBlur(mask, (blur_px, blur_px), 0)

    return mask