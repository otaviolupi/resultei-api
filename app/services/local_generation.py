import torch
from PIL import Image, ImageOps
import numpy as np

PROCEDURE_PROMPTS = {
    "lip_filler": {
        "positive": "natural human lips, realistic mouth, detailed lips texture, skin texture, photorealistic, detailed face, professional portrait, same person",
        "negative": "black area, empty space, missing mouth, distorted face, deformed lips, cartoon, unrealistic, blur, low quality",
    },
    "nose_reshape": {
        "positive": "subtle rhinoplasty result, refined nose bridge, natural proportions, realistic skin, photorealistic",
        "negative": "cartoon, dramatic change, pinched, unnatural, fake, distorted, black, dark",
    },
    "jaw_slim": {
        "positive": "slimmer jawline, natural facial contouring, realistic skin texture, subtle change, photorealistic",
        "negative": "cartoon, extreme, unnatural, distorted, black, dark",
    },
    "chin_augment": {
        "positive": "subtle chin augmentation, natural chin projection, realistic skin, photorealistic",
        "negative": "cartoon, extreme, unnatural, dramatic change, fake, black, dark",
    },
    "brow_lift": {
        "positive": "natural eyebrow lift, refreshed appearance, realistic skin, subtle change, photorealistic",
        "negative": "cartoon, surprised look, frozen, unnatural, extreme, black, dark",
    },
    "eye_bags": {
        "positive": "under-eye rejuvenation, refreshed look, natural skin, subtle improvement, photorealistic",
        "negative": "cartoon, extreme, hollow, sunken, unnatural, black, dark",
    },
    "cheek_filler": {
        "positive": "subtle cheek filler result, gentle volume enhancement, natural skin, photorealistic",
        "negative": "cartoon, chipmunk cheeks, overdone, puffy, unnatural, black, dark",
    },
    "skin_smooth": {
        "positive": "smooth even skin texture, natural skin, subtle improvement, photorealistic",
        "negative": "cartoon, plastic skin, airbrushed, blurred, waxy, unnatural, black, dark",
    },
}

_pipeline = None


def load_pipeline(device="cuda", precision="fp16"):
    global _pipeline

    from diffusers import StableDiffusionInpaintPipeline

    torch_dtype = torch.float16 if precision == "fp16" else torch.float32

    print("Carregando runwayml/stable-diffusion-inpainting...")

    _pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if device == "cuda":
        _pipeline = _pipeline.to("cuda")
        _pipeline.enable_attention_slicing()

    print("Modelo carregado com sucesso.")
    return _pipeline


def get_pipeline():
    return _pipeline


def run_inpainting(
    pipe,
    image_pil: Image.Image,
    mask_np: np.ndarray,
    procedure: str,
    intensity: float = 0.5,
) -> Image.Image:
    prompts = PROCEDURE_PROMPTS[procedure]

    image_512 = image_pil.resize((512, 512), Image.LANCZOS)
    mask_512 = Image.fromarray(mask_np).resize((512, 512), Image.LANCZOS).convert("L")

    # Expande o contexto — passa região maior para o modelo ter referência de pele
    import numpy as np
    mask_array = np.array(mask_512)
    # Dilata a máscara para dar mais contexto ao modelo
    import cv2
    mask_512 = Image.fromarray(mask_array).convert("L")

    strength = 0.08 + (intensity * 0.10)

    with torch.inference_mode():
        result = pipe(
            prompt=prompts["positive"],
            negative_prompt=prompts["negative"],
            image=image_512,
            mask_image=mask_512,
            inpaint_full_res=True,
            height=512,
            width=512,
            strength=strength,
            num_inference_steps=50,
            guidance_scale=6.5,
            generator=torch.Generator("cuda").manual_seed(42),
        ).images[0]

    # Recorta só a região modificada e cola na original
    orig_array = np.array(image_512.convert("RGB"))
    result_array = np.array(result.convert("RGB"))
    
    # Usa máscara ORIGINAL (não expandida) para o blending final
    mask_orig = np.array(Image.fromarray(mask_np).resize((512, 512), Image.LANCZOS).convert("L"))
    mask_norm = mask_orig.astype(float) / 255.0
    mask_3ch = np.stack([mask_norm] * 3, axis=-1)
    
    blended = (result_array * mask_3ch + orig_array * (1 - mask_3ch)).astype(np.uint8)
    
    return Image.fromarray(blended)