import time
import uuid
import numpy as np
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from PIL import Image

from app.services import face_detection, postprocess, storage
from app.services.morph_transform import apply_procedure, WARP_PROCEDURES, TEXTURE_PROCEDURES
from app.models.schemas import SimulateResponse, ImageResult
from app.db import crud
from app.utils.image_utils import (
    validate_image_format, validate_file_size,
    validate_dimensions, resize_for_processing,
)

router = APIRouter()

VALID_PROCEDURES = face_detection.VALID_PROCEDURES


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    procedure: str = Form(...),
    intensity: float = Form(0.5, ge=0.0, le=1.0),
    mode: str = Form("fast"),
):
    start = time.time()

    # --- Validações ---
    validate_image_format(image.content_type)
    contents = await image.read()
    validate_file_size(len(contents))

    if procedure not in VALID_PROCEDURES:
        raise HTTPException(
            status_code=422,
            detail=f"Procedimento inválido: '{procedure}'. "
                   f"Opções: {', '.join(VALID_PROCEDURES)}",
        )

    try:
        image_pil = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Não foi possível abrir a imagem.")

    validate_dimensions(image_pil)
    image_pil = resize_for_processing(image_pil)

    job_id = str(uuid.uuid4())
    image_np = np.array(image_pil)

    # --- Detecção facial (uma só vez) ---
    try:
        landmarks = face_detection.get_landmarks(image_np)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # --- Aplicar procedimento ---
    try:
        result_pil = apply_procedure(image_pil, landmarks, procedure, intensity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

    # --- Gerar máscara para blend final ---
    # Procedimentos de textura já fazem blend internamente — máscara None pula seamless clone
    if procedure in WARP_PROCEDURES:
        if procedure == 'lip_filler':
            # lip_filler faz blend interno (warp localizado) — mask None pula blend externo
            mask_np = None
        else:
            mask_np = face_detection.generate_mask(
                image_np, procedure,
                landmarks=landmarks,
                dilate_px=12,
                blur_px=19,
            )
    else:
        mask_np = None  # eye_bags e skin_smooth ja retornam imagem blendada

    # --- Pós-processamento ---
    final_pil = postprocess.blend_result(image_pil, result_pil, mask_np, procedure=procedure)
    comparison_pil = postprocess.create_side_by_side(image_pil, final_pil, label=True)

    # --- Storage ---
    original_url   = storage.upload_image(image_pil,     "original",    job_id, "orig")
    result_url     = storage.upload_image(final_pil,     "result",      job_id, "result")
    comparison_url = storage.upload_image(comparison_pil,"comparison",  job_id, "compare")

    ms = int((time.time() - start) * 1000)

    background_tasks.add_task(
        crud.save_job, job_id, procedure, intensity, mode,
        original_url, result_url, comparison_url, ms,
    )

    return SimulateResponse(
        job_id=job_id,
        status="completed",
        procedure=procedure,
        intensity=intensity,
        mode=mode,
        images=ImageResult(
            original_url=original_url,
            result_url=result_url,
            side_by_side_url=comparison_url,
        ),
        processing_time_ms=ms,
    )


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await crud.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado.")
    return {
        "job_id": job.id,
        "status": job.status,
        "procedure": job.procedure,
        "images": {
            "original_url": job.original_url,
            "result_url": job.result_url,
            "side_by_side_url": job.comparison_url,
        },
        "processing_time_ms": job.processing_time_ms,
        "created_at": job.created_at,
    }