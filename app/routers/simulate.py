import time
import uuid
import numpy as np
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from PIL import Image

from app.services import face_detection, local_generation, postprocess, storage
from app.services.local_generation import get_pipeline
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
    mode: str = Form("fast"),  # default fast — sem Replicate por enquanto
):
    start = time.time()

    # Validações
    validate_image_format(image.content_type)

    contents = await image.read()
    validate_file_size(len(contents))

    if procedure not in VALID_PROCEDURES:
        raise HTTPException(
            status_code=422,
            detail=f"Procedimento inválido: '{procedure}'. "
                   f"Opções: {', '.join(VALID_PROCEDURES)}",
        )

    # Decodifica
    try:
        image_pil = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Não foi possível abrir a imagem.")

    validate_dimensions(image_pil)
    image_pil = resize_for_processing(image_pil)

    job_id = str(uuid.uuid4())

    # Detecção facial
    image_np = np.array(image_pil)
    # MediaPipe espera BGR
    try:
        mask_np = face_detection.generate_mask(image_np, procedure)

        Image.fromarray(mask_np).save("debug_01_mask.png")

        from PIL import Image as PILImage
        import os
        os.makedirs("static/debug", exist_ok=True)
        PILImage.fromarray(mask_np).save("static/debug/mask_debug.png")
        PILImage.fromarray(image_np).save("static/debug/image_debug.png")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Geração
    pipe = get_pipeline()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado ainda.")

    result_pil = local_generation.run_inpainting(pipe, image_pil, mask_np, procedure, intensity)

    result_pil.save("debug_02_result_before_blend.png")

    # Pós-processamento
    final_pil = postprocess.blend_result(image_pil, result_pil, mask_np)
    final_pil.save("debug_03_final_after_blend.png")
    comparison_pil = postprocess.create_side_by_side(image_pil, final_pil)

    # Storage local
    original_url = storage.upload_image(image_pil, "original", job_id, "orig")
    result_url = storage.upload_image(final_pil, "result", job_id, "result")
    comparison_url = storage.upload_image(comparison_pil, "comparison", job_id, "compare")

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