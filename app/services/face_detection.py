import mediapipe as mp
import numpy as np
import cv2

PROCEDURE_LANDMARK_MAP = {
    "lip_filler": [
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
    ],
    "nose_reshape": [
        1, 2, 98, 327, 168, 6, 122, 351, 196, 419,
        3, 51, 281, 45, 275, 44, 274, 220, 440,
    ],
    "jaw_slim": [
        172, 136, 150, 149, 176, 148, 152, 377, 400,
        378, 379, 365, 397, 288, 361, 323,
    ],
    "chin_augment": [
        152, 148, 377, 176, 149, 150, 136, 172,
        58, 132, 93, 234, 454, 323, 361, 288,
    ],
    "brow_lift": [
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        296, 334, 293, 300, 276, 283, 282, 295, 285,
    ],
    "eye_bags": [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        362, 382, 381, 380, 374, 373, 390, 249, 263,
    ],
    "cheek_filler": [
        116, 123, 147, 213, 192, 214, 210, 211, 32,
        345, 352, 376, 433, 416, 434, 430, 431, 262,
    ],
    "skin_smooth": None,  # máscara = rosto inteiro
}

VALID_PROCEDURES = list(PROCEDURE_LANDMARK_MAP.keys())


def generate_mask(image_np: np.ndarray, procedure: str) -> np.ndarray:
    """
    Detecta landmarks faciais e retorna máscara binária (uint8, 0 ou 255)
    cobrindo a região relevante ao procedimento.
    """
    # API nova do mediapipe (>= 0.10.x no Windows)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    # Tenta primeiro a API legada, que pode funcionar dependendo da versão
    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_np)
            return _build_mask(results, image_np, procedure)
    except AttributeError:
        # Fallback: API nova do mediapipe
        return _generate_mask_new_api(image_np, procedure)


def _build_mask(results, image_np: np.ndarray, procedure: str) -> np.ndarray:
    if not results.multi_face_landmarks:
        raise ValueError(
            "Nenhum rosto detectado. Use uma foto frontal com boa iluminação."
        )

    h, w = image_np.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark
    indices = PROCEDURE_LANDMARK_MAP.get(procedure)
    mask = np.zeros((h, w), dtype=np.uint8)

    if indices is None:
        all_pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, cv2.convexHull(all_pts), 255)
    else:
        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def _generate_mask_new_api(image_np: np.ndarray, procedure: str) -> np.ndarray:
    """Fallback usando a API de tasks do mediapipe >= 0.10."""
    import mediapipe as mp
    import urllib.request
    import os

    model_path = "./mediapipe_face_landmarker.task"
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
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
        detection_result = landmarker.detect(mp_image)

    if not detection_result.face_landmarks:
        raise ValueError(
            "Nenhum rosto detectado. Use uma foto frontal com boa iluminação."
        )

    h, w = image_np.shape[:2]
    landmarks = detection_result.face_landmarks[0]
    indices = PROCEDURE_LANDMARK_MAP.get(procedure)
    mask = np.zeros((h, w), dtype=np.uint8)

    if indices is None:
        all_pts = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in landmarks],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, cv2.convexHull(all_pts), 255)
    else:
        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    return mask