"""Local passport image preprocessing. Images remain in memory."""

import cv2
import numpy as np
from PIL import Image, ImageOps


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def resize_image(image: np.ndarray, target_width: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    if w == target_width:
        return image.copy()
    scale = target_width / max(w, 1)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, max(1, int(h * scale))), interpolation=interpolation)


def estimate_skew(gray: np.ndarray, max_angle: float = 8.0, step: float = 0.5) -> float:
    """Estimate a small rotation that makes text rows more horizontal."""
    h, w = gray.shape[:2]
    if w < 200:
        return 0.0

    small_w = 700
    small = cv2.resize(gray, (small_w, max(1, int(h * small_w / w))), interpolation=cv2.INTER_AREA)
    binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    sh, sw = binary.shape
    best_angle, best_score = 0.0, -1.0

    for angle in np.arange(-max_angle, max_angle + step, step):
        matrix = cv2.getRotationMatrix2D((sw / 2, sh / 2), float(angle), 1.0)
        rotated = cv2.warpAffine(binary, matrix, (sw, sh), flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score

    return best_angle


def rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.25:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    border = int(np.median(gray))
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=border)


def prepare_mrz_images(gray: np.ndarray) -> list[tuple[str, Image.Image]]:
    """Create several conservative MRZ crops. The bottom part of the page is used."""
    h, w = gray.shape[:2]
    crops = [
        ("bottom30", gray[int(h * 0.70):, :]),
        ("bottom38", gray[int(h * 0.62):, :]),
        ("bottom45", gray[int(h * 0.55):, :]),
    ]

    attempts = []
    for name, crop in crops:
        if crop.size == 0:
            continue
        scale = 2400 / max(crop.shape[1], 1)
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)

        blur = cv2.GaussianBlur(enlarged, (3, 3), 0)
        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 11
        )

        attempts.append((f"{name}-gray", cv2_to_pil(enlarged)))
        attempts.append((f"{name}-otsu", cv2_to_pil(otsu)))
        attempts.append((f"{name}-adaptive", cv2_to_pil(adaptive)))

    return attempts


def preprocess_image(pil_image: Image.Image, upside_down: bool = False) -> dict:
    """Return visual OCR regions and multiple MRZ OCR inputs."""
    image = ImageOps.exif_transpose(pil_image).convert("RGB")
    original = pil_to_cv2(image)

    if original.shape[0] < 100 or original.shape[1] < 150:
        raise ValueError("Image is too small to contain a passport page.")

    if upside_down:
        original = cv2.rotate(original, cv2.ROTATE_180)

    resized = resize_image(original, 1600)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = rotate_gray(gray, estimate_skew(gray))

    h = gray.shape[0]

    # Keep the identity/visual area but exclude the very bottom MRZ when possible.
    visual_top = gray[:int(h * 0.86), :]

    visual_contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(visual_top)

    return {
        "original": cv2_to_pil(resized),
        "visual_region": cv2_to_pil(visual_contrast),
        "mrz_attempts": [
            {"name": name, "kind": "block", "images": [image]}
            for name, image in prepare_mrz_images(gray)
        ],
        "mrz_located": False,
    }
