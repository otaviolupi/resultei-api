from PIL import Image, ImageOps

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_MB = 10
MIN_DIMENSION = 256
WORK_RESOLUTION = 512  # SD 1.5 — GTX 1660 Super 6GB


def validate_image_format(content_type: str):
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Formato não suportado: '{content_type}'. "
            "Envie a foto em JPG ou PNG."
        )


def validate_file_size(size_bytes: int):
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"Arquivo muito grande ({size_mb:.1f}MB). Máximo: {MAX_FILE_SIZE_MB}MB."
        )


def validate_dimensions(image: Image.Image):
    w, h = image.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise ValueError(
            f"Imagem muito pequena ({w}x{h}px). Mínimo: {MIN_DIMENSION}x{MIN_DIMENSION}px."
        )


def resize_for_processing(image: Image.Image, target: int = WORK_RESOLUTION) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")

    w, h = image.size
    min_side = min(w, h)

    left = (w - min_side) // 2
    top = (h - min_side) // 2
    right = left + min_side
    bottom = top + min_side

    image = image.crop((left, top, right, bottom))
    image = image.resize((target, target), Image.LANCZOS)

    return image