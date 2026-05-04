import os
from pathlib import Path
from PIL import Image
from app.config import settings

_storage_dir = Path(settings.LOCAL_STORAGE_DIR)


def upload_image(image_pil: Image.Image, folder: str,
                 job_id: str, suffix: str) -> str:
    folder_path = _storage_dir / folder
    folder_path.mkdir(parents=True, exist_ok=True)

    filename = f"{job_id}_{suffix}.jpg"
    file_path = folder_path / filename
    image_pil.save(str(file_path), format="JPEG", quality=92)

    return f"{settings.LOCAL_BASE_URL}/{folder}/{filename}"