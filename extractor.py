"""Passport extraction pipeline: local Tesseract OCR + MRZ + targeted visual OCR."""

import os
import shutil

import pytesseract
from PIL import Image, ImageOps

from preprocess import preprocess_image
from mrz_parser import (
    parse_mrz,
    line1_is_good,
    line1_plausibility,
    with_line1,
)
from validator import build_result, failure_result
from visual_ocr import extract_visual_fields


# ---------------- Tesseract ----------------

def _configure_tesseract():
    if shutil.which("tesseract"):
        return

    for path in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"
        ),
    ):
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


_configure_tesseract()


# ---------------- MRZ OCR configuration ----------------

MRZ_WHITELIST = (
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
)

MRZ_BLOCK_CONFIG = (
    f"--oem 3 --psm 6 {MRZ_WHITELIST} "
    "-c load_system_dawg=0 "
    "-c load_freq_dawg=0 "
    "-c user_defined_dpi=300"
)

MRZ_LINE_CONFIG = (
    f"--oem 3 --psm 7 {MRZ_WHITELIST} "
    "-c load_system_dawg=0 "
    "-c load_freq_dawg=0 "
    "-c user_defined_dpi=300"
)

VISUAL_PSM6 = "--oem 3 --psm 6 -c user_defined_dpi=300"


# ---------------- Basic OCR ----------------

def run_ocr(
    image: Image.Image,
    config: str = VISUAL_PSM6,
) -> str:
    if image is None:
        return ""

    return pytesseract.image_to_string(
        image,
        config=config,
    ).strip()


# ---------------- MRZ scoring ----------------

def _critical_check_count(parsed: dict) -> int:
    """Count the four important TD3 line-2 check digits."""
    validation = parsed.get("validation", {})

    return sum(
        bool(validation.get(key))
        for key in (
            "document_number_valid",
            "birth_date_valid",
            "expiry_date_valid",
            "composite_valid",
        )
    )


def _mrz_quality(parsed: dict) -> int:
    """Score MRZ quality for choosing both an OCR attempt and orientation."""
    validation = parsed.get("validation", {})

    checks = _critical_check_count(parsed)

    score = checks * 100

    if validation.get("overall_valid"):
        score += 1000

    if line1_is_good(parsed):
        score += 100

    line1 = parsed.get("line1", "")
    nationality = parsed.get("data", {}).get("nationality", "")

    score += max(
        0,
        line1_plausibility(
            line1,
            nationality,
        ),
    )

    return score


def _mrz_is_orientation_candidate(parsed: dict) -> bool:
    """
    Decide whether an MRZ parse is strong enough to identify orientation.

    Parser success alone is deliberately NOT enough because parse_mrz can
    produce a structured candidate even when some check digits failed.
    """
    if not parsed.get("success"):
        return False

    if _critical_check_count(parsed) < 2:
        return False

    validation = parsed.get("validation", {})
    line1 = parsed.get("line1", "")
    nationality = parsed.get("data", {}).get("nationality", "")

    if not line1.startswith("P"):
        return False

    if not validation.get("country_format_valid"):
        return False

    if not validation.get("nationality_format_valid"):
        return False

    # A good line-1 structure is preferred, but we don't require it here;
    # the validator will still decide whether names are trustworthy.
    if len(line1) != 44:
        return False

    if not nationality:
        return False

    return True


# ---------------- MRZ reading ----------------

def _read_mrz(attempts: list) -> tuple:
    """Run the existing MRZ attempts and select the strongest validated result."""

    texts = {}
    parses = []
    all_parses = []

    for attempt in attempts:

        config = (
            MRZ_LINE_CONFIG
            if attempt["kind"] == "lines"
            else MRZ_BLOCK_CONFIG
        )

        outputs = []

        for image in attempt["images"]:
            try:
                text = run_ocr(
                    image,
                    config,
                )
            except pytesseract.TesseractError:
                text = ""

            if text:
                outputs.append(text)

        combined = "\n".join(outputs)
        texts[attempt["name"]] = combined

        parsed = parse_mrz(combined)

        if parsed.get("success"):
            all_parses.append(
                (
                    attempt["name"],
                    parsed,
                )
            )

            if _mrz_is_orientation_candidate(parsed):
                parses.append(
                    (
                        attempt["name"],
                        parsed,
                    )
                )

    # Only trust strong MRZ candidates here. Weak parser successes are not
    # allowed to drive orientation selection.
    if not parses:
        return None, None, texts

    best_name, best = max(
        parses,
        key=lambda item: _mrz_quality(item[1]),
    )

    nationality = (
        best
        .get("data", {})
        .get("nationality", "")
    )

    # Improve line 1 independently using the same nationality anchor.
    line1_name, line1_best = max(
        parses,
        key=lambda item: line1_plausibility(
            item[1]["line1"],
            nationality,
        ),
    )

    if (
        line1_best is not best
        and line1_plausibility(
            line1_best["line1"],
            nationality,
        )
        > line1_plausibility(
            best["line1"],
            nationality,
        )
    ):
        best = with_line1(
            best,
            line1_best,
        )
        best_name = (
            f"{best_name} (line 2) + "
            f"{line1_name} (line 1)"
        )

    return best, best_name, texts


# ---------------- Orientation ----------------

ORIENTATION_ORDER = (
    0,
    180,
    90,
    270,
)


def _rotate_orientation(
    image: Image.Image,
    angle: int,
) -> Image.Image:
    """Rotate directly from the original image."""

    if angle == 0:
        return image.copy()

    # Quarter-turns are exact pixel rearrangements. Do not use bicubic
    # interpolation here because Stage 2 may need the original-quality pixels.
    if angle == 90:
        return image.transpose(Image.Transpose.ROTATE_90)

    if angle == 180:
        return image.transpose(Image.Transpose.ROTATE_180)

    if angle == 270:
        return image.transpose(Image.Transpose.ROTATE_270)

    raise ValueError(
        f"Unsupported orientation angle: {angle}"
    )


def _find_readable_orientation(
    pil_image: Image.Image,
) -> tuple:
    """
    Test the passport in four orientations and use MRZ quality to select it.

    The normal and upside-down orientations are tested first because they are
    the common cases. Sideways orientations are tested when necessary, but a
    stronger result in a later orientation is still allowed to win.
    """

    base_image = ImageOps.exif_transpose(
        pil_image
    )

    orientation_results = []

    for angle in ORIENTATION_ORDER:

        try:
            oriented = _rotate_orientation(
                base_image,
                angle,
            )

            steps = preprocess_image(
                oriented,
                upside_down=False,
            )

            best, best_name, mrz_texts = _read_mrz(
                steps["mrz_attempts"]
            )

            if best is None:
                orientation_results.append(
                    {
                        "angle": angle,
                        "success": False,
                        "score": 0,
                        "critical_checks": 0,
                        "overall_valid": False,
                        "line1_good": False,
                        "mrz_variant": None,
                    }
                )
                continue

            validation = best.get(
                "validation",
                {},
            )

            orientation_results.append(
                {
                    "angle": angle,
                    "success": True,
                    "score": _mrz_quality(best),
                    "critical_checks": _critical_check_count(best),
                    "overall_valid": bool(
                        validation.get(
                            "overall_valid"
                        )
                    ),
                    "line1_good": line1_is_good(best),
                    "mrz_variant": best_name,
                    "validation": validation,
                    "image": oriented,
                    "mrz": best,
                    "mrz_texts": mrz_texts,
                }
            )

            # A fully valid MRZ with good line-1 structure is sufficient.
            # We can stop early on this orientation because orientation is no
            # longer ambiguous for a passport image that produced this result.
            if (
                validation.get("overall_valid")
                and line1_is_good(best)
            ):
                break

        except pytesseract.TesseractNotFoundError:
            raise

        except Exception as exc:
            orientation_results.append(
                {
                    "angle": angle,
                    "success": False,
                    "score": 0,
                    "critical_checks": 0,
                    "overall_valid": False,
                    "line1_good": False,
                    "mrz_variant": None,
                    "error": str(exc),
                }
            )

    successful = [
        item
        for item in orientation_results
        if item.get("success")
    ]

    if not successful:
        return (
            None,
            None,
            None,
            None,
            {},
            orientation_results,
        )

    # Prefer complete MRZ validation. Otherwise use the highest quality
    # strong candidate.
    complete = [
        item
        for item in successful
        if item.get("overall_valid")
    ]

    candidates = (
        complete
        if complete
        else successful
    )

    selected = max(
        candidates,
        key=lambda item: item["score"],
    )

    return (
        selected["image"],
        selected["angle"],
        selected["mrz"],
        selected["mrz_variant"],
        selected["mrz_texts"],
        orientation_results,
    )


# ---------------- Visual OCR ----------------

def _read_visual_text(
    image: Image.Image,
) -> tuple[str, dict, dict]:
    """Run targeted visual OCR on the selected readable orientation."""

    fields = extract_visual_fields(image)

    full_text = (
        fields
        .get("_meta", {})
        .get("full_text", "")
    )

    passes = {
        "identity-data-psm11": full_text
    }

    return (
        full_text,
        {
            "selected": "identity-data-psm11",
            "score": None,
            "passes": passes,
        },
        fields,
    )


# ---------------- Pipeline ----------------

def _extract_once(
    pil_image: Image.Image,
) -> tuple:
    """Determine orientation from MRZ, then run visual OCR once."""

    try:
        (
            image,
            selected_angle,
            best,
            best_name,
            mrz_texts,
            orientation_debug,
        ) = _find_readable_orientation(
            pil_image
        )

    except pytesseract.TesseractNotFoundError:
        return (
            failure_result(
                "Tesseract is not installed or could not be found."
            ),
            None,
            True,
        )

    except Exception as exc:
        return (
            failure_result(
                f"The image could not be processed ({exc})."
            ),
            None,
            True,
        )

    clean_orientation_debug = []

    for item in orientation_debug:
        clean_orientation_debug.append(
            {
                key: value
                for key, value in item.items()
                if key not in {
                    "image",
                    "mrz",
                    "mrz_texts",
                }
            }
        )

    if best is None:

        debug = {
            "mrz_variant_used": None,
            "mrz_ocr_texts": {},
            "visual_ocr_text": "",
            "visual_ocr_passes": {},
            "visual_selected_pass": None,
            "visual_score": None,
            "visual_fields": {},
            "mrz_lines": [None, None],
            "mrz_validation": None,
            "selected_orientation": None,
            "orientation_scores": clean_orientation_debug,
        }

        return (
            failure_result(
                "No sufficiently readable machine-readable zone "
                "could be found in any orientation. Please upload "
                "a clear passport data-page image."
            ),
            debug,
            False,
        )

    debug = {
        "mrz_variant_used": best_name,
        "mrz_ocr_texts": mrz_texts,
        "visual_ocr_text": "",
        "visual_ocr_passes": {},
        "visual_selected_pass": None,
        "visual_score": None,
        "visual_fields": {},
        "mrz_lines": [
            best.get("line1"),
            best.get("line2"),
        ],
        "mrz_validation": best.get("validation"),
        "selected_orientation": selected_angle,
        "orientation_scores": clean_orientation_debug,

        # Keep the orientation-corrected image available to the Streamlit UI.
        # It is stored only in debug output; the structured API result remains JSON-safe.
        "corrected_image": image,
    }

    try:

        # IMPORTANT: use the selected orientation for the visual OCR too.
        visual_text, visual_debug, visual_fields = (
            _read_visual_text(image)
        )

        debug["visual_ocr_text"] = visual_text
        debug["visual_ocr_passes"] = visual_debug["passes"]
        debug["visual_selected_pass"] = visual_debug["selected"]
        debug["visual_score"] = visual_debug["score"]
        debug["visual_fields"] = {
            key: value
            for key, value in visual_fields.items()
            if key != "_meta"
        }

        result = build_result(
            best,
            visual_text,
            visual_fields=visual_fields,
        )

    except pytesseract.TesseractNotFoundError:
        return (
            failure_result(
                "Tesseract is not installed or could not be found."
            ),
            None,
            True,
        )

    except Exception as exc:
        return (
            failure_result(
                f"OCR failed ({type(exc).__name__}). "
                "Check the Tesseract installation."
            ),
            None,
            True,
        )

    if selected_angle != 0:
        result.setdefault(
            "warnings",
            [],
        ).append(
            "The passport image was rotated "
            f"{selected_angle}° automatically so the MRZ "
            "could be read."
        )

    return (
        result,
        debug,
        False,
    )


# ---------------- Public API ----------------

def extract_passport(
    pil_image: Image.Image,
    include_debug: bool = False,
) -> dict:
    """Public entry point for the passport extractor."""

    try:
        result, debug, hard_error = _extract_once(
            pil_image
        )

    except Exception as exc:
        return failure_result(
            f"Unexpected error while processing ({type(exc).__name__})."
        )

    if include_debug and debug:
        result["debug"] = debug

    return result