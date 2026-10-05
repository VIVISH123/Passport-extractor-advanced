"""Stage 2 passport analysis layer.

Stage 2 is deliberately isolated from the existing Stage 1 pipeline.
It reuses the existing MRZ reader/parser/validator logic, but changes the
source priority for the Stage 2 merge so validated MRZ data is primary.

Flow:
    whole image
      -> one document-border detection
      -> crop + perspective correction
      -> normalized passport
      -> Part 1 / Part 2 / Part 3 split
      -> Part 3 MRZ OCR + existing MRZ parser/validation
      -> Part 1 + Part 2 visual OCR
      -> MRZ-primary merge

No per-part rotation is performed.
"""

from __future__ import annotations

import datetime
import math
import re
from difflib import SequenceMatcher
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from extractor import (
    _critical_check_count,
    _mrz_quality,
    _read_mrz,
)
from validator import (
    FIELDS,
    _country_text_to_code,
    _mrz_field,
    _mrz_field_valid,
    _norm_alnum,
    _visual_confidence,
    _visual_name_matches_mrz,
    _visual_valid,
    _visual_value,
    build_result,
    infer_date_roles,
    mrz_date_to_iso,
)
from visual_ocr import extract_visual_fields
from mrz_parser import parse_mrz


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MAX_DETECT_DIM = 2200

# Passport data-page geometry (approximately 125 x 88 mm = 1.42:1).
PASSPORT_ASPECT_RATIO = 125.0 / 88.0
PASSPORT_ASPECT_MIN = 1.27
PASSPORT_ASPECT_MAX = 1.58

# A real passport must occupy a meaningful part of the photograph, but must
# not simply be the whole camera frame/background.
MIN_DOCUMENT_AREA = 0.025
MAX_DOCUMENT_AREA = 0.97

# Candidate acceptance gates.
MIN_BORDER_CONFIDENCE = 0.62
MIN_RIGHT_ANGLE_SCORE = 0.55
MIN_OPPOSITE_EDGE_BALANCE = 0.55
MAX_SIDE_RATIO = 2.25

TARGET_LONG_SIDE = 3200
MAX_WORKING_LONG_SIDE = 5200
MIN_RECTIFIED_WIDTH = 500
MIN_RECTIFIED_HEIGHT = 300

# The passport data page is normally landscape. Part 3 is deliberately larger
# than a strict MRZ strip so both MRZ rows survive perspective/crop variation.
PART1_END = 0.45
PART2_END = 0.72

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
MRZ_SPARSE_CONFIG = (
    f"--oem 3 --psm 11 {MRZ_WHITELIST} "
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


# ------------------------------------------------------------
# Basic image helpers
# ------------------------------------------------------------


def _pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(
        np.array(ImageOps.exif_transpose(image).convert("RGB")),
        cv2.COLOR_RGB2BGR,
    )


def _cv2_to_pil(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _detection_image(image: Image.Image) -> tuple[np.ndarray, float]:
    """Create a bounded-resolution image for border detection."""
    cv = _pil_to_cv2(image)
    h, w = cv.shape[:2]
    scale = min(1.0, MAX_DETECT_DIM / max(h, w))
    if scale < 1.0:
        cv = cv2.resize(
            cv,
            (
                max(1, int(round(w * scale))),
                max(1, int(round(h * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return cv, scale


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Return quad in top-left, top-right, bottom-right, bottom-left order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _side_lengths(quad: np.ndarray) -> tuple[float, float, float, float]:
    q = _order_quad(quad)
    return tuple(
        float(np.linalg.norm(q[(i + 1) % 4] - q[i]))
        for i in range(4)
    )


def _quad_area(quad: np.ndarray) -> float:
    q = _order_quad(quad)
    return float(abs(cv2.contourArea(q.astype(np.float32))))


def _angle_score(quad: np.ndarray) -> float:
    """1.0 is a perfect rectangle; 0.0 is a very poor quadrilateral."""
    q = _order_quad(quad)
    scores = []
    for i in range(4):
        a = q[(i - 1) % 4] - q[i]
        b = q[(i + 1) % 4] - q[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom <= 1e-6:
            return 0.0
        cos_angle = float(np.dot(a, b) / denom)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle = math.degrees(math.acos(cos_angle))
        deviation = abs(angle - 90.0)
        scores.append(max(0.0, 1.0 - deviation / 45.0))
    return float(sum(scores) / len(scores))


def _quad_geometry(
    quad: np.ndarray,
    image_shape: tuple[int, int],
) -> dict:
    """Validate passport geometry without caring whether it is portrait or landscape."""
    h, w = image_shape[:2]
    q = _order_quad(quad)
    lengths = _side_lengths(q)

    top, right, bottom, left = lengths
    width = (top + bottom) / 2.0
    height = (left + right) / 2.0

    if width <= 1 or height <= 1:
        return {
            "valid": False,
            "aspect": 0.0,
            "aspect_error": 999.0,
            "aspect_score": 0.0,
            "area_ratio": 0.0,
            "area_score": 0.0,
            "rightness": 0.0,
            "opposite_edge_balance": 0.0,
            "perspective_score": 0.0,
            "max_side_ratio": 999.0,
            "score": 0.0,
        }

    # Orientation-independent passport ratio.
    # 1.42 is landscape; 0.704 is the same passport rotated 90 degrees.
    aspect = max(width, height) / max(1.0, min(width, height))
    aspect_error = abs(aspect - PASSPORT_ASPECT_RATIO)
    aspect_score = max(0.0, 1.0 - aspect_error / 0.30)
    aspect_valid = PASSPORT_ASPECT_MIN <= aspect <= PASSPORT_ASPECT_MAX

    area_ratio = _quad_area(q) / max(1.0, float(w * h))
    area_valid = MIN_DOCUMENT_AREA <= area_ratio <= MAX_DOCUMENT_AREA
    area_score = min(1.0, math.sqrt(max(0.0, area_ratio) / 0.08)) if area_valid else 0.0

    rightness = _angle_score(q)

    horizontal_balance = min(top, bottom) / max(1.0, max(top, bottom))
    vertical_balance = min(left, right) / max(1.0, max(left, right))
    opposite_edge_balance = (horizontal_balance + vertical_balance) / 2.0
    perspective_score = opposite_edge_balance

    max_side_ratio = max(lengths) / max(1.0, min(lengths))
    perspective_valid = max_side_ratio <= MAX_SIDE_RATIO

    valid = (
        aspect_valid
        and area_valid
        and rightness >= MIN_RIGHT_ANGLE_SCORE
        and opposite_edge_balance >= MIN_OPPOSITE_EDGE_BALANCE
        and perspective_valid
    )

    score = (
        0.42 * aspect_score
        + 0.24 * rightness
        + 0.18 * opposite_edge_balance
        + 0.11 * perspective_score
        + 0.05 * area_score
    )

    if not valid:
        score *= 0.25

    return {
        "valid": bool(valid),
        "aspect": round(float(aspect), 4),
        "aspect_error": round(float(aspect_error), 4),
        "aspect_score": round(float(aspect_score), 4),
        "area_ratio": round(float(area_ratio), 6),
        "area_score": round(float(area_score), 4),
        "rightness": round(float(rightness), 4),
        "opposite_edge_balance": round(float(opposite_edge_balance), 4),
        "perspective_score": round(float(perspective_score), 4),
        "max_side_ratio": round(float(max_side_ratio), 4),
        "score": round(float(max(0.0, min(1.0, score))), 4),
    }


def _passport_mrz_evidence_fast(
    image: Image.Image,
    quad: np.ndarray,
) -> dict:
    """Fast MRZ/content check used ONLY while ranking border candidates.

    Border detection only needs enough evidence to distinguish a passport-like
    rectangle from a random rectangle. The full MRZ ensemble remains unchanged
    and runs after the border is selected.
    """
    try:
        src = _pil_to_cv2(image)
        q = _order_quad(np.asarray(quad, dtype=np.float32))

        tl, tr, br, bl = q
        width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
        if width < 90 or height < 60:
            return {
                "score": 0.0,
                "trustworthy": False,
                "text": "",
                "reason": "candidate-too-small",
            }

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(q, destination)
        warped = cv2.warpPerspective(
            src,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if warped.size == 0:
            return {
                "score": 0.0,
                "trustworthy": False,
                "text": "",
                "reason": "empty-warp",
            }

        # Put the likely orientation first. A correctly oriented passport has
        # its MRZ along the lower edge; a quarter-turn portrait candidate needs
        # a 90/270 search instead. We stop as soon as the MRZ signature is strong.
        angles = (0, 180, 90, 270) if width >= height else (90, 270, 0, 180)

        best = None
        for angle_index, angle in enumerate(angles):
            if angle == 0:
                view = warped
            elif angle == 90:
                view = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                view = cv2.rotate(warped, cv2.ROTATE_180)
            else:
                view = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            longest = max(view.shape[:2])
            scale_up = min(4.0, max(1.0, 1500.0 / max(1.0, float(longest))))
            if scale_up > 1.02:
                view = cv2.resize(
                    view,
                    None,
                    fx=scale_up,
                    fy=scale_up,
                    interpolation=cv2.INTER_CUBIC,
                )

            vh = view.shape[0]
            crop = view[int(round(vh * 0.55)):, :]
            if crop.shape[0] < 20:
                continue

            try:
                raw = pytesseract.image_to_string(
                    crop,
                    config=MRZ_BLOCK_CONFIG,
                )
            except Exception:
                raw = ""

            normalized_lines = []
            for raw_line in str(raw or "").splitlines():
                clean = re.sub(r"[^A-Z0-9<]", "", raw_line.upper())
                if clean:
                    normalized_lines.append(clean)

            compact = "".join(normalized_lines)
            has_start = bool(re.search(r"[PV][<A-Z]", compact))
            has_filler = compact.count("<") >= 6
            long_lines = [x for x in normalized_lines if len(x) >= 22]
            two_long = len(long_lines) >= 2
            digits = sum(c.isdigit() for c in compact)

            score = 0.0
            if has_start:
                score += 0.38
            if has_filler:
                score += 0.23
            if long_lines:
                score += 0.22
            if two_long:
                score += 0.17
            if digits >= 5:
                score += 0.05
            score = min(1.0, score)

            record = {
                "score": score,
                "trustworthy": False,
                "text": "\n".join(normalized_lines),
                "selected_variant": f"fast-angle{angle}-bottom55",
                "ocr_orientation": angle,
                "has_mrz_start": has_start,
                "has_many_fillers": has_filler,
                "has_long_mrz_line": bool(long_lines),
                "has_two_long_lines": two_long,
            }
            if best is None or record["score"] > best["score"]:
                best = record

            if score >= 0.72 or (has_start and two_long):
                break

            # If the first likely edge was weak, inspect its opposite edge once.
            if angle_index == 0 and score < 0.45:
                top_crop = view[: max(20, int(round(vh * 0.45))), :]
                try:
                    raw_top = pytesseract.image_to_string(
                        top_crop,
                        config=MRZ_BLOCK_CONFIG,
                    )
                except Exception:
                    raw_top = ""

                top_lines = []
                for raw_line in str(raw_top or "").splitlines():
                    clean = re.sub(r"[^A-Z0-9<]", "", raw_line.upper())
                    if clean:
                        top_lines.append(clean)

                top_compact = "".join(top_lines)
                top_start = bool(re.search(r"[PV][<A-Z]", top_compact))
                top_fillers = top_compact.count("<") >= 6
                top_long = [x for x in top_lines if len(x) >= 22]
                top_two = len(top_long) >= 2

                top_score = 0.0
                if top_start:
                    top_score += 0.38
                if top_fillers:
                    top_score += 0.23
                if top_long:
                    top_score += 0.22
                if top_two:
                    top_score += 0.17
                top_score = min(1.0, top_score)

                top_record = {
                    "score": top_score,
                    "trustworthy": False,
                    "text": "\n".join(top_lines),
                    "selected_variant": f"fast-angle{angle}-top45",
                    "ocr_orientation": angle,
                    "has_mrz_start": top_start,
                    "has_many_fillers": top_fillers,
                    "has_long_mrz_line": bool(top_long),
                    "has_two_long_lines": top_two,
                }
                if best is None or top_score > best["score"]:
                    best = top_record

                if top_score >= 0.72 or (top_start and top_two):
                    break

        if best is None:
            return {
                "score": 0.0,
                "trustworthy": False,
                "text": "",
                "reason": "fast-mrz-ocr-no-result",
            }

        return best
    except Exception as exc:
        return {
            "score": 0.0,
            "trustworthy": False,
            "text": "",
            "reason": f"{type(exc).__name__}: {exc}",
        }

def _passport_mrz_evidence(
    image: Image.Image,
    quad: np.ndarray,
) -> dict:
    """Check whether a candidate rectangle contains real MRZ/passport evidence.

    This is intentionally tolerant of tiny source documents and rotated source
    images.  The candidate image itself is never rotated or downsampled in the
    returned pipeline; rotations/upscaling here are OCR-only temporary views.
    """
    try:
        src = _pil_to_cv2(image)
        q = _order_quad(np.asarray(quad, dtype=np.float32))

        tl, tr, br, bl = q
        width = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
        height = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))

        # The old 250x180 hard gate killed small passports embedded inside a
        # large white canvas.  Border detection should be scale-independent;
        # OCR can upscale a small candidate safely in a temporary copy.
        if width < 90 or height < 60:
            return {
                "score": 0.0,
                "trustworthy": False,
                "text": "",
                "reason": "candidate-too-small",
            }

        destination = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(q, destination)
        warped = cv2.warpPerspective(
            src,
            matrix,
            (width, height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

        if warped.size == 0:
            return {
                "score": 0.0,
                "trustworthy": False,
                "text": "",
                "reason": "empty-warp",
            }

        # Make OCR easier for tiny candidates, but NEVER modify the returned
        # passport image.  Test all four orientations because the page may be
        # photographed rotated while its MRZ remains horizontally readable in
        # the source image.
        views = []
        seen_shapes = set()
        for angle, view in (
            (0, warped),
            (90, cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)),
            (180, cv2.rotate(warped, cv2.ROTATE_180)),
            (270, cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ):
            shape_key = (view.shape[0], view.shape[1], angle % 180)
            # Do not remove 0/180 or 90/270 solely by shape; the MRZ moves to
            # opposite edges and both directions can matter.
            if shape_key in seen_shapes:
                continue
            seen_shapes.add(shape_key)

            longest = max(view.shape[:2])
            ocr_scale = min(7.0, max(1.0, 2200.0 / max(1.0, float(longest))))
            if ocr_scale > 1.02:
                view = cv2.resize(
                    view,
                    None,
                    fx=ocr_scale,
                    fy=ocr_scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            views.append((angle, view))

        best_record = None
        all_text = []

        for angle, view in views:
            ocr_pil = _cv2_to_pil(view)
            h_view = ocr_pil.height
            w_view = ocr_pil.width

            # For a correctly oriented MRZ, it is normally at the bottom.  For
            # 180-degree / upside-down images it is at the top.  Also include
            # the lower/upper 65% as a forgiving fallback for cropped pages.
            crop_specs = (
                (0.24, "bottom24", False),
                (0.32, "bottom32", False),
                (0.42, "bottom42", False),
                (0.55, "bottom55", False),
                (0.68, "bottom68", False),
                (0.24, "top24", True),
                (0.32, "top32", True),
                (0.42, "top42", True),
                (0.55, "top55", True),
                (0.68, "top68", True),
            )

            attempts = []
            for fraction, name, top_crop in crop_specs:
                if top_crop:
                    y1 = int(round(h_view * fraction))
                    crop = ocr_pil.crop((0, 0, w_view, max(1, y1)))
                else:
                    y0 = int(round(h_view * (1.0 - fraction)))
                    crop = ocr_pil.crop((0, y0, w_view, h_view))
                attempts.append(
                    {
                        "name": f"angle{angle}-{name}",
                        "kind": "block",
                        "images": [crop],
                    }
                )

            try:
                best, best_name, texts = _read_mrz(attempts)
            except Exception:
                best, best_name, texts = None, "", {}

            text_values = [
                str(t or "")
                for t in (texts.values() if isinstance(texts, dict) else texts)
            ]
            all_text.extend(text_values)
            combined = "\n".join(text_values).upper()
            compact = re.sub(r"[^A-Z0-9<]", "", combined)

            trustworthy = _mrz_trustworthy(best)
            critical_checks = _critical_check_count(best) if best else 0
            quality = _mrz_quality(best) if best else 0

            # Do not require a perfect TD3 parse to prove that a rectangle is
            # a passport.  OCR can be imperfect while the visible MRZ shape is
            # still strong enough to anchor the border.
            has_mrz_start = bool(re.search(r"[PV][<A-Z]", compact))
            has_many_fillers = compact.count("<") >= 8
            has_long_mrz_line = any(
                len(re.sub(r"[^A-Z0-9<]", "", t)) >= 24
                for t in text_values
            )
            has_two_long_lines = sum(
                1
                for t in text_values
                if len(re.sub(r"[^A-Z0-9<]", "", t)) >= 24
            ) >= 2

            if trustworthy:
                score = 1.0
            else:
                score = 0.0
                if has_mrz_start:
                    score += 0.32
                if has_many_fillers:
                    score += 0.25
                if has_long_mrz_line:
                    score += 0.22
                if has_two_long_lines:
                    score += 0.15
                if critical_checks >= 1:
                    score += 0.10
                if quality >= 0.50:
                    score += 0.10
                score = min(1.0, score)

            record = {
                "score": float(score),
                "trustworthy": bool(trustworthy),
                "critical_checks": int(critical_checks),
                "quality": float(quality),
                "text": combined[-1000:],
                "selected_variant": best_name,
                "orientation": angle,
                "has_mrz_start": has_mrz_start,
                "has_many_fillers": has_many_fillers,
                "has_long_mrz_line": has_long_mrz_line,
                "has_two_long_lines": has_two_long_lines,
                "best": best,
            }

            if best_record is None or record["score"] > best_record["score"]:
                best_record = record

        if best_record is None:
            return {
                "score": 0.0,
                "trustworthy": False,
                "text": "",
                "reason": "mrz-ocr-no-result",
            }

        return {
            "score": round(float(best_record["score"]), 4),
            "trustworthy": bool(best_record["trustworthy"]),
            "critical_checks": int(best_record["critical_checks"]),
            "quality": float(best_record["quality"]),
            "text": ("\n".join(all_text))[-1500:],
            "selected_variant": best_record["selected_variant"],
            "ocr_orientation": int(best_record["orientation"]),
            "has_mrz_start": bool(best_record["has_mrz_start"]),
            "has_td3_start": bool(best_record["has_mrz_start"]),
            "has_many_fillers": bool(best_record["has_many_fillers"]),
            "has_long_mrz_line": bool(best_record["has_long_mrz_line"]),
            "has_two_long_lines": bool(best_record["has_two_long_lines"]),
        }

    except Exception as exc:
        return {
            "score": 0.0,
            "trustworthy": False,
            "text": "",
            "reason": f"{type(exc).__name__}: {exc}",
        }

def _candidate_from_contour(
    contour: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, dict] | None:
    """Create a geometrically valid passport candidate.

    IMPORTANT: geometry alone does not prove that the rectangle is a passport.
    MRZ evidence is added later by detect_passport_border().
    """
    h, w = image_shape[:2]
    area = cv2.contourArea(contour)
    if area <= 0:
        return None

    area_ratio = area / max(1.0, float(w * h))
    if area_ratio < MIN_DOCUMENT_AREA or area_ratio > MAX_DOCUMENT_AREA:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None

    best = None

    for epsilon_factor in (
        0.005, 0.007, 0.009, 0.012, 0.016, 0.020, 0.025, 0.032, 0.040
    ):
        approx = cv2.approxPolyDP(
            contour,
            epsilon_factor * perimeter,
            True,
        )

        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        quad = _order_quad(
            approx.reshape(4, 2).astype(np.float32)
        )

        if min(_side_lengths(quad)) < 15:
            continue

        metrics = _quad_geometry(
            quad,
            image_shape,
        )

        if not metrics["valid"]:
            continue

        score = metrics["score"]
        if best is None or score > best[1]["score"]:
            best = (quad, metrics, "contour-quad")

    if best is None:
        return None

    return best[0], best[1]


def _edge_candidates(cv: np.ndarray) -> list[tuple[np.ndarray, dict, str]]:
    """Generate many real edge-defined quadrilaterals.

    No foreground/background rectangle is manufactured here.  The detector
    only considers actual contours, which are then validated with passport
    geometry and MRZ evidence.
    """
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    variants = (
        (20, 60, "raw-canny-20-60"),
        (30, 90, "raw-canny-30-90"),
        (40, 120, "raw-canny-40-120"),
        (50, 150, "raw-canny-50-150"),
        (70, 210, "raw-canny-70-210"),
    )

    results = []

    for low, high, source in variants:
        edges = cv2.Canny(gray, low, high)

        # Keep the raw contour route as well as modest closing routes.
        edge_maps = [
            (edges, source),
            (
                cv2.morphologyEx(
                    edges,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
                    iterations=1,
                ),
                source + "-close9x3",
            ),
            (
                cv2.morphologyEx(
                    edges,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
                    iterations=1,
                ),
                source + "-close15x5",
            ),
        ]

        for edge_map, map_source in edge_maps:
            contours, _ = cv2.findContours(
                edge_map,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            contours = sorted(
                contours,
                key=cv2.contourArea,
                reverse=True,
            )[:180]

            for contour in contours:
                candidate = _candidate_from_contour(
                    contour,
                    cv.shape[:2],
                )
                if candidate is None:
                    continue

                quad, metrics = candidate
                results.append(
                    (quad, metrics, map_source)
                )

    return results


def _candidate_duplicate(
    a: np.ndarray,
    b: np.ndarray,
    image_shape: tuple[int, int],
) -> bool:
    h, w = image_shape[:2]
    qa = _order_quad(a)
    qb = _order_quad(b)

    center_distance = np.linalg.norm(
        np.mean(qa, axis=0) - np.mean(qb, axis=0)
    )

    diagonal = math.hypot(w, h)
    if center_distance > diagonal * 0.025:
        return False

    area_a = _quad_area(qa)
    area_b = _quad_area(qb)

    if max(area_a, area_b) <= 1:
        return True

    return (
        min(area_a, area_b)
        / max(area_a, area_b)
        > 0.88
    )




def _longest_contiguous_span(indices: np.ndarray) -> tuple[int, int] | None:
    """Return the longest contiguous inclusive span from sorted integer indices."""
    if indices.size == 0:
        return None
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(indices) - 1]
    lengths = ends - starts + 1
    best = int(np.argmax(lengths))
    return int(indices[starts[best]]), int(indices[ends[best]])


def _white_canvas_content_window(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Find a compact document-sized content window inside a mostly white canvas.

    This is a rescue path for images such as a tiny passport page pasted/captured
    inside a much larger white image. It does not run unless the surrounding
    canvas is strongly white, so ordinary full-frame passport photographs are
    left to the existing contour/edge detector.
    """
    try:
        rgb = np.asarray(image.convert("RGB"))
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return None
        h, w = rgb.shape[:2]
        if w < 180 or h < 180:
            return None

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Require a genuinely white surrounding canvas.  Passport images on a
        # desk/wall/background should continue through the normal detector.
        corner_h = max(8, int(round(h * 0.08)))
        corner_w = max(8, int(round(w * 0.08)))
        corners = (
            gray[:corner_h, :corner_w],
            gray[:corner_h, -corner_w:],
            gray[-corner_h:, :corner_w],
            gray[-corner_h:, -corner_w:],
        )
        white_corners = sum(float(np.median(c)) >= 250.0 for c in corners)
        if white_corners < 3:
            return None

        candidates = []
        for threshold in (254, 252, 250, 247, 244, 240):
            mask = (gray < threshold).astype(np.uint8)

            # Ignore a very thin, uniformly dark strip at the image edge (for
            # example a browser/app separator captured above the document).
            row_dark = gray.mean(axis=1)
            y_start_guard = 0
            max_guard = min(24, int(round(h * 0.04)))
            while (
                y_start_guard < max_guard
                and row_dark[y_start_guard] < 80.0
                and y_start_guard + 1 < h
                and row_dark[y_start_guard + 1] > 170.0
            ):
                y_start_guard += 1
            if y_start_guard:
                mask[:y_start_guard, :] = 0

            col_occ = mask.mean(axis=0)
            row_occ = mask.mean(axis=1)

            # A real page produces a long occupied band.  Tiny isolated OCR
            # marks or UI text do not.
            x_idx = np.where(col_occ >= 0.008)[0]
            y_idx = np.where(row_occ >= 0.008)[0]
            x_span = _longest_contiguous_span(x_idx)
            y_span = _longest_contiguous_span(y_idx)
            if x_span is None or y_span is None:
                continue

            x0, x1 = x_span
            y0, y1 = y_span
            span_w = x1 - x0 + 1
            span_h = y1 - y0 + 1
            area_ratio = (span_w * span_h) / float(w * h)

            if span_w < max(100, int(round(w * 0.08))):
                continue
            if span_h < max(100, int(round(h * 0.08))):
                continue
            if not (0.008 <= area_ratio <= 0.65):
                continue

            aspect = max(span_w, span_h) / max(1.0, min(span_w, span_h))
            # Allow both normal and 90-degree page orientation.
            if not (1.18 <= aspect <= 1.85):
                continue

            # Prefer the smallest compact window that still contains a lot of
            # foreground pixels. This is ideal for a document in white space.
            fill = float(mask[y0:y1 + 1, x0:x1 + 1].mean())
            score = (
                0.45 * min(1.0, fill / 0.20)
                + 0.30 * min(1.0, (1.0 - area_ratio) / 0.90)
                + 0.25 * min(1.0, aspect / 1.42)
            )
            candidates.append((score, x0, y0, x1, y1, threshold, fill))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        _, x0, y0, x1, y1, _, _ = candidates[0]

        # Add a modest margin around the detected content. If the passport is
        # touching an image edge, clipping is intentional.
        pad_x = max(3, int(round((x1 - x0 + 1) * 0.035)))
        pad_y = max(3, int(round((y1 - y0 + 1) * 0.035)))
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(w - 1, x1 + pad_x)
        y1 = min(h - 1, y1 + pad_y)

        return int(x0), int(y0), int(x1), int(y1)
    except Exception:
        return None


def _white_canvas_border_fallback(image: Image.Image) -> dict | None:
    """Localize a document inside a mostly white canvas, then use normal
    passport geometry + MRZ evidence on that localized window.

    This path is intentionally downstream of the established detector. It only
    activates when the ordinary border detector cannot produce a valid result.
    """
    window = _white_canvas_content_window(image)
    if window is None:
        return None

    x0, y0, x1, y1 = window
    arr = np.asarray(image.convert("RGB"))
    local = Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1].copy(), mode="RGB")
    local_cv, scale = _detection_image(local)
    local_candidates = _edge_candidates(local_cv)

    evaluated = []
    for quad_small, geometry, source in local_candidates[:80]:
        local_q = quad_small / max(scale, 1e-8)
        local_q[:, 0] = np.clip(local_q[:, 0], 0, local.width - 1)
        local_q[:, 1] = np.clip(local_q[:, 1], 0, local.height - 1)
        local_q = _order_quad(local_q)
        evidence = _passport_mrz_evidence_fast(local, local_q)
        combined = (
            0.42 * geometry["score"]
            + 0.18 * geometry["aspect_score"]
            + 0.40 * evidence["score"]
        )
        evaluated.append((combined, local_q, geometry, evidence, source))

    # On extremely faint pages there may be no closed contour at all. In that
    # case the localized window itself is a safe axis-aligned candidate because
    # it was derived from the compact page footprint, not an arbitrary minAreaRect.
    # IMPORTANT: score this candidate against the ORIGINAL image dimensions so
    # its area remains meaningful. The localized crop would otherwise be 100%
    # document area by definition and fail the normal area gate.
    window_quad_local = np.array(
        [[0, 0], [local.width - 1, 0],
         [local.width - 1, local.height - 1], [0, local.height - 1]],
        dtype=np.float32,
    )
    window_quad_original = window_quad_local.copy()
    window_quad_original[:, 0] += x0
    window_quad_original[:, 1] += y0
    window_geometry = _quad_geometry(
        window_quad_original,
        (image.height, image.width),
    )
    if window_geometry["valid"]:
        window_evidence = _passport_mrz_evidence_fast(image, window_quad_original)
        combined = (
            0.42 * window_geometry["score"]
            + 0.18 * window_geometry["aspect_score"]
            + 0.40 * window_evidence["score"]
        )
        evaluated.append(
            (
                combined,
                window_quad_original,
                window_geometry,
                window_evidence,
                "white-canvas-window",
            )
        )

    if not evaluated:
        return None

    evaluated.sort(key=lambda item: item[0], reverse=True)
    combined, local_q, geometry, evidence, source = evaluated[0]

    if not (
        evidence.get("trustworthy", False)
        or evidence.get("score", 0.0) >= 0.50
    ):
        return None

    # Map local coordinates back to original image coordinates.
    mapped = np.asarray(local_q, dtype=np.float32).copy()
    mapped[:, 0] += x0
    mapped[:, 1] += y0
    mapped = _order_quad(mapped)

    final_geometry = _quad_geometry(mapped, (image.height, image.width))
    if not final_geometry["valid"]:
        return None

    return {
        "success": True,
        "method": "white-canvas-localized-border",
        "confidence": round(min(1.0, float(combined)), 4),
        "quad": mapped.astype(float).tolist(),
        "metrics": final_geometry,
        "fallback": True,
        "candidate_count": len(evaluated),
        "reason": (
            "The image contained a compact passport-sized region inside a mostly "
            "white canvas. The region was localized first, then normal passport "
            "geometry and MRZ evidence were applied without changing the source image."
        ),
        "localized_window": [x0, y0, x1, y1],
        "mrz_evidence": evidence,
        "source": source,
    }


def _mrz_anchor_border_fallback_single(image: Image.Image) -> dict | None:
    """Infer a passport rectangle from a clearly detected MRZ when the outer
    border is too faint, cropped, or touches the image edge.

    This is deliberately a LAST-RESORT path.  Normal contour/edge detection
    remains unchanged.  The inferred rectangle is accepted only after the
    existing passport MRZ evidence check succeeds, so an arbitrary text block
    cannot become the passport border.
    """
    try:
        src = np.asarray(image.convert("RGB"))
        if src.size == 0:
            return None
        bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if w < 120 or h < 120:
            return None

        # The uploaded document can be tiny inside a much larger white canvas.
        # Upscale only this temporary OCR copy; the returned quad stays in the
        # original image coordinates and no source image is downsampled.
        scale = min(5.0, max(2.5, 1600.0 / max(w, h)))
        up = cv2.resize(
            bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        configs = (
            "--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
            "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<",
        )

        boxes = []
        for config in configs:
            data = pytesseract.image_to_data(
                gray,
                config=config,
                output_type=pytesseract.Output.DICT,
            )
            for i, raw in enumerate(data.get("text", [])):
                text = re.sub(r"[^A-Z0-9<]", "", str(raw or "").upper())
                if len(text) < 8:
                    continue
                try:
                    conf = float(data["conf"][i])
                except Exception:
                    conf = 0.0
                x = float(data["left"][i]) / scale
                y = float(data["top"][i]) / scale
                ww = float(data["width"][i]) / scale
                hh = float(data["height"][i]) / scale
                # MRZ-like text is long and sits low relative to the inferred
                # page.  We don't require a P< prefix because OCR may corrupt it.
                if ww >= max(35.0, w * 0.06) and hh >= 2.0:
                    boxes.append((text, x, y, ww, hh, conf))

        if not boxes:
            return None

        # Prefer long OCR spans near the bottom.  The sample that previously
        # failed produces a single strong MRZ line, which is enough to infer
        # the document rectangle.
        boxes.sort(
            key=lambda b: (
                len(b[0]) * 3.0
                + b[4] * 0.05
                + max(0.0, b[2] / max(1.0, h)) * 20.0
                + (b[5] / 100.0),
            ),
            reverse=True,
        )

        candidates = []
        for text, x, y, ww, hh, conf in boxes[:12]:
            if y < h * 0.20:
                continue

            # MRZ normally spans almost the whole document width.  Add a small
            # margin rather than using the OCR word box as the passport border.
            doc_width = min(
                float(w),
                max(ww * 1.08, ww + 18.0),
            )
            cx = x + ww / 2.0
            x0 = cx - doc_width / 2.0
            x1 = cx + doc_width / 2.0

            # Generate both orientations.  Geometry/evidence chooses the one
            # that actually behaves like a passport.  This keeps orientation
            # preservation intact and avoids forcing portrait/landscape.
            for portrait in (True, False):
                doc_height = (
                    doc_width * PASSPORT_ASPECT_RATIO
                    if portrait
                    else doc_width / PASSPORT_ASPECT_RATIO
                )

                # MRZ is very close to the bottom edge of the passport.  Keep
                # this margin small so a tiny document inside a large canvas is
                # not pushed far below its actual bottom border.
                bottom = min(float(h - 1), y + hh + max(2.0, hh * 0.25))
                top = bottom - doc_height

                # If the source is cropped at an image edge, clipping is valid.
                # The geometry check is orientation-independent.
                q = np.array(
                    [
                        [x0, top],
                        [x1, top],
                        [x1, bottom],
                        [x0, bottom],
                    ],
                    dtype=np.float32,
                )
                q[:, 0] = np.clip(q[:, 0], 0, w - 1)
                q[:, 1] = np.clip(q[:, 1], 0, h - 1)
                q = _order_quad(q)

                metrics = _quad_geometry(q, (h, w))
                if not metrics["valid"]:
                    continue

                evidence = _passport_mrz_evidence_fast(image, q)
                score = (
                    0.34 * metrics["score"]
                    + 0.66 * evidence["score"]
                )
                if evidence["trustworthy"]:
                    score += 0.15

                candidates.append(
                    (score, q, metrics, evidence, text, conf)
                )

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        score, quad, metrics, evidence, anchor_text, anchor_conf = candidates[0]

        # This fallback must still prove that the inferred rectangle contains
        # passport/MRZ evidence.  Never accept geometry alone.
        if not (
            evidence.get("trustworthy", False)
            or evidence.get("score", 0.0) >= 0.55
        ):
            return None

        return {
            "success": True,
            "method": "mrz-anchor-inferred-border",
            "confidence": round(min(1.0, float(score)), 4),
            "quad": quad,
            "metrics": metrics,
            "fallback": True,
            "candidate_count": len(candidates),
            "reason": (
                "Outer passport border was too faint/cropped for contour detection; "
                "a passport rectangle was inferred from MRZ evidence and then validated."
            ),
            "mrz_anchor_text": anchor_text,
            "mrz_anchor_confidence": anchor_conf,
            "mrz_evidence": evidence,
        }

    except Exception:
        return None


def _map_rotated_quad_to_original(
    quad: np.ndarray,
    angle: int,
    original_size: tuple[int, int],
) -> np.ndarray:
    """Map a quad detected on a temporary rotated image back to original pixels."""
    w, h = original_size
    q = np.asarray(quad, dtype=np.float32).copy()

    if angle == 0:
        mapped = q
    elif angle == 90:
        # cv2.ROTATE_90_CLOCKWISE: (x, y) -> (y, h-1-x)
        mapped = np.column_stack((q[:, 1], (h - 1) - q[:, 0]))
    elif angle == 180:
        mapped = np.column_stack(((w - 1) - q[:, 0], (h - 1) - q[:, 1]))
    elif angle == 270:
        # cv2.ROTATE_90_COUNTERCLOCKWISE: (x, y) -> (w-1-y, x)
        mapped = np.column_stack(((w - 1) - q[:, 1], q[:, 0]))
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")

    mapped[:, 0] = np.clip(mapped[:, 0], 0, w - 1)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, h - 1)
    return _order_quad(mapped.astype(np.float32))


def _mrz_anchor_border_fallback(image: Image.Image) -> dict | None:
    """Orientation-independent MRZ-anchor fallback.

    The existing last-resort detector assumes the MRZ is at the bottom of the
    working image. Stage-1 orientation can legitimately leave a passport rotated
    90/180/270 degrees, so run that detector on temporary rotated copies too.
    The selected quad is always mapped back to the ORIGINAL image coordinates;
    the source image itself is never rotated for the subsequent perspective warp.
    """
    try:
        original = image.convert("RGB")
        src = np.asarray(original)
        h, w = src.shape[:2]
        if w < 120 or h < 120:
            return None

        rotated_images = {
            0: original,
            90: Image.fromarray(cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)),
            180: Image.fromarray(cv2.rotate(src, cv2.ROTATE_180)),
            270: Image.fromarray(cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        }

        candidates = []
        for angle, rotated in rotated_images.items():
            result = _mrz_anchor_border_fallback_single(rotated)
            if not result or not result.get("success") or result.get("quad") is None:
                continue

            mapped_quad = _map_rotated_quad_to_original(
                np.asarray(result["quad"], dtype=np.float32),
                angle,
                (w, h),
            )

            # Re-check evidence in ORIGINAL orientation/coordinates. This is
            # essential: rotation is only a search aid, never the final image.
            original_evidence = _passport_mrz_evidence_fast(original, mapped_quad)
            score = float(result.get("confidence", 0.0))
            evidence_score = float(original_evidence.get("score", 0.0))
            combined = 0.55 * score + 0.45 * evidence_score
            if original_evidence.get("trustworthy", False):
                combined += 0.10

            candidates.append((combined, angle, mapped_quad, result, original_evidence))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        combined, angle, quad, result, evidence = candidates[0]

        if not (
            evidence.get("trustworthy", False)
            or evidence.get("score", 0.0) >= 0.55
        ):
            return None

        return {
            **result,
            "success": True,
            "method": "mrz-anchor-inferred-border-rotated-search",
            "confidence": round(min(1.0, float(combined)), 4),
            "quad": quad,
            "mrz_evidence": evidence,
            "search_rotation": angle,
            "reason": (
                "Outer border was too faint for contour detection. MRZ was found "
                "on a temporary rotated search image and the validated passport "
                "rectangle was mapped back to the original orientation."
            ),
        }
    except Exception:
        return None

def detect_passport_border(image: Image.Image) -> dict:
    """Detect the passport border using geometry PLUS internal MRZ evidence.

    A face/photo rectangle may satisfy passport dimensions.  It cannot satisfy
    the MRZ evidence gate, so it will not be selected as the passport.
    """
    detect_cv, scale = _detection_image(image)
    candidates = _edge_candidates(detect_cv)

    if not candidates:
        canvas_fallback = _white_canvas_border_fallback(image)
        if canvas_fallback is not None:
            return canvas_fallback
        fallback = _mrz_anchor_border_fallback(image)
        if fallback is not None:
            return fallback
        return {
            "success": False,
            "method": "no-passport-candidate",
            "confidence": 0.0,
            "quad": None,
            "metrics": {},
            "fallback": False,
            "candidate_count": 0,
            "reason": "No passport-like quadrilateral was detected.",
        }

    # Deduplicate before expensive OCR evidence scoring.
    unique = []
    for candidate in candidates:
        quad, metrics, source = candidate
        if any(
            _candidate_duplicate(
                quad,
                old_quad,
                detect_cv.shape[:2],
            )
            for old_quad, _, _ in unique
        ):
            continue
        unique.append(candidate)

    # First rank by geometry.  Keep enough candidates that a faint outer
    # passport border can still win after MRZ evidence is considered.
    unique.sort(
        key=lambda item: item[1]["score"],
        reverse=True,
    )

    # Border verification is the performance-critical part of Stage 2.
    # Use a cheap MRZ signature check on geometry-ranked candidates first.
    # The established full MRZ ensemble remains available as a fallback.
    shortlist = unique[:24]

    evidence_candidates = []

    for quad_small, geometry, source in shortlist:
        quad_original = quad_small / max(scale, 1e-8)
        quad_original[:, 0] = np.clip(quad_original[:, 0], 0, image.width - 1)
        quad_original[:, 1] = np.clip(quad_original[:, 1], 0, image.height - 1)
        quad_original = _order_quad(quad_original)

        evidence = _passport_mrz_evidence_fast(
            image,
            quad_original,
        )

        combined_score = (
            0.42 * geometry["score"]
            + 0.18 * geometry["aspect_score"]
            + 0.40 * evidence["score"]
        )

        evidence_candidates.append(
            (
                quad_original,
                geometry,
                evidence,
                source,
                combined_score,
            )
        )

        # Strong geometry + strong MRZ evidence is enough. Do not OCR the
        # remaining candidates once the passport is effectively identified.
        if (
            evidence.get("score", 0.0) >= 0.72
            and geometry.get("score", 0.0) >= 0.78
        ):
            break

    # --------------------------------------------------------
    # HARD MRZ GATE
    # --------------------------------------------------------
    # Do not accept a generic rectangle merely because it has passport-like
    # dimensions.  A candidate needs meaningful MRZ evidence.
    # --------------------------------------------------------
    valid_evidence = [
        item
        for item in evidence_candidates
        if (
            item[2]["trustworthy"]
            or item[2]["score"] >= 0.55
        )
    ]

    if not valid_evidence:
        # Accuracy-preserving slow path: use the original full MRZ evidence
        # routine only on a few top geometry candidates.
        deep_candidates = []
        for quad_small, geometry, source in unique[:8]:
            quad_original = quad_small / max(scale, 1e-8)
            quad_original[:, 0] = np.clip(quad_original[:, 0], 0, image.width - 1)
            quad_original[:, 1] = np.clip(quad_original[:, 1], 0, image.height - 1)
            quad_original = _order_quad(quad_original)

            evidence = _passport_mrz_evidence(image, quad_original)
            combined_score = (
                0.42 * geometry["score"]
                + 0.18 * geometry["aspect_score"]
                + 0.40 * evidence["score"]
            )
            deep_candidates.append(
                (quad_original, geometry, evidence, source, combined_score)
            )

        valid_evidence = [
            item
            for item in deep_candidates
            if (
                item[2]["trustworthy"]
                or item[2]["score"] >= 0.55
            )
        ]
        if valid_evidence:
            evidence_candidates = deep_candidates

    if not valid_evidence:
        canvas_fallback = _white_canvas_border_fallback(image)
        if canvas_fallback is not None:
            return canvas_fallback
        fallback = _mrz_anchor_border_fallback(image)
        if fallback is not None:
            return fallback
        return {
            "success": False,
            "method": "no-passport-content-evidence",
            "confidence": 0.0,
            "quad": None,
            "metrics": {},
            "fallback": False,
            "candidate_count": len(unique),
            "reason": (
                "Geometric rectangles were found, but none contained "
                "sufficient TD3 MRZ/passport evidence."
            ),
        }

    valid_evidence.sort(
        key=lambda item: item[4],
        reverse=True,
    )

    selected_quad, geometry, evidence, source, combined_score = (
        valid_evidence[0]
    )

    # Final full-resolution geometry validation.
    geometry = _quad_geometry(
        selected_quad,
        (image.height, image.width),
    )

    # Strong MRZ evidence can rescue a slightly imperfect contour, but we still
    # require basic passport geometry.
    if not geometry["valid"]:
        return {
            "success": False,
            "method": "full-resolution-geometry-failed",
            "confidence": 0.0,
            "quad": None,
            "metrics": geometry,
            "fallback": False,
            "candidate_count": len(unique),
            "reason": "Best MRZ-supported candidate failed final geometry validation.",
        }

    confidence = max(
        0.0,
        min(
            1.0,
            combined_score,
        ),
    )

    if confidence < MIN_BORDER_CONFIDENCE:
        canvas_fallback = _white_canvas_border_fallback(image)
        if canvas_fallback is not None:
            return canvas_fallback
        fallback = _mrz_anchor_border_fallback(image)
        if fallback is not None:
            return fallback
        return {
            "success": False,
            "method": "passport-confidence-too-low",
            "confidence": round(confidence, 3),
            "quad": None,
            "metrics": geometry,
            "fallback": False,
            "candidate_count": len(unique),
            "reason": "Best passport candidate did not reach the confidence threshold.",
        }

    return {
        "success": True,
        "method": source,
        "confidence": round(confidence, 3),
        "quad": selected_quad.astype(float).tolist(),
        "metrics": geometry,
        "mrz_evidence": evidence,
        "fallback": False,
        "candidate_count": len(unique),
    }


def _safe_perspective_crop(
    image: Image.Image,
    border: dict,
) -> tuple[Image.Image, dict]:
    """Perspective-correct the passport without changing its orientation."""
    if not border.get("success", False):
        raise ValueError("No valid passport border was detected.")
    if not border.get("quad"):
        raise ValueError("Passport detector returned no quadrilateral.")

    src = _pil_to_cv2(image)
    q = _order_quad(np.asarray(border["quad"], dtype=np.float32))
    geometry = _quad_geometry(q, src.shape[:2])

    if not geometry["valid"] or geometry["score"] < MIN_BORDER_CONFIDENCE:
        raise ValueError("Passport quadrilateral failed geometry validation.")

    tl, tr, br, bl = q
    detected_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    detected_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0

    # Preserve detected orientation. Do NOT swap width/height and do NOT rotate.
    # Keep the native detected dimensions; there is no arbitrary 6000px cap.
    out_w = max(MIN_RECTIFIED_WIDTH, int(round(detected_width)))
    out_h = max(MIN_RECTIFIED_HEIGHT, int(round(detected_height)))

    destination = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(q, destination)
    warped = cv2.warpPerspective(
        src,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if warped.size == 0:
        raise ValueError("Perspective transformation produced an empty image.")

    rectified_ratio = max(warped.shape[1], warped.shape[0]) / max(
        1.0, min(warped.shape[1], warped.shape[0])
    )
    if not PASSPORT_ASPECT_MIN <= rectified_ratio <= PASSPORT_ASPECT_MAX:
        raise ValueError(
            f"Rectified passport ratio {rectified_ratio:.3f} is outside the allowed passport range."
        )

    return _cv2_to_pil(warped), {
        "method": "validated-perspective-warp",
        "warped": True,
        "orientation_preserved": True,
        "detected_orientation": "landscape" if detected_width >= detected_height else "portrait",
        "geometry": geometry,
        "output_size": [int(warped.shape[1]), int(warped.shape[0])],
        "rectified_aspect": round(rectified_ratio, 4),
    }

def normalize_passport(image: Image.Image) -> Image.Image:
    """Return the corrected passport at native resolution.

    OCR routines are responsible for their own temporary enlargement. The
    document displayed by the UI and passed to the splitter is never silently
    downsampled here.
    """
    return ImageOps.exif_transpose(image).convert("RGB")


def split_passport(image: Image.Image) -> dict[str, Image.Image]:
    """Split the already normalized passport once; no per-part redetection."""
    image = ImageOps.exif_transpose(image).convert("RGB")

    # IMPORTANT: NEVER force the passport into landscape here.
    #
    # Stage 1 is the orientation authority whenever it recovered even one
    # readable field. The border detector/perspective warp must preserve that
    # orientation. Rotating here would undo the Stage 1 orientation lock and
    # can turn a correctly readable passport sideways again.

    h = image.height
    y1 = max(1, min(h - 2, int(round(h * PART1_END))))
    y2 = max(y1 + 1, min(h - 1, int(round(h * PART2_END))))

    return {
        "part1": image.crop((0, 0, image.width, y1)),
        "part2": image.crop((0, y1, image.width, y2)),
        "part3": image.crop((0, y2, image.width, h)),
    }


# ------------------------------------------------------------
# Stage 2 MRZ
# ------------------------------------------------------------


def _mrz_upscale(image: Image.Image, target_width: int = 2800) -> Image.Image:
    """Prepare MRZ for OCR without changing the source passport image."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)

    if gray.width < target_width:
        scale = target_width / max(1, gray.width)
        gray = gray.resize(
            (target_width, max(1, int(round(gray.height * scale)))),
            Image.Resampling.LANCZOS,
        )

    # Mild local contrast enhancement. This is an OCR working copy only.
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    return gray


def _detect_mrz_row_bands(gray: np.ndarray) -> list[tuple[int, int]]:
    """Detect the two printed MRZ text rows from the Part 3 crop.

    Do not guess the rows by splitting Part 3 at 50%.  Part 3 can contain
    different amounts of whitespace, borders, and passport artwork.  Instead,
    use the horizontal ink profile to find the two broad text bands.  Border
    lines touching the crop edge are ignored.
    """
    if gray is None or gray.size == 0:
        return []

    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape[:2]
    if h < 20 or w < 100:
        return []

    # MRZ characters are dark on a relatively pale background.  Otsu gives a
    # document-specific threshold; clamp it so faint scans still contribute.
    otsu_value, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = int(max(125, min(205, otsu_value + 8)))
    dark = (gray < threshold).astype(np.uint8)

    # Count dark pixels across each row. Smooth only vertically so characters
    # in the same printed row become one broad peak.
    row_score = dark.sum(axis=1).astype(np.float32)
    row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 5), 0).ravel()

    # Require enough horizontal ink to be a text row, but do not require a
    # fixed percentage because passport images vary heavily in contrast.
    min_pixels = max(12.0, w * 0.045)
    active = row_score >= min_pixels

    groups = []
    i = 0
    while i < h:
        if not active[i]:
            i += 1
            continue
        start = i
        gap = 0
        i += 1
        while i < h:
            if active[i]:
                gap = 0
            else:
                gap += 1
                if gap > 2:
                    break
            i += 1
        end = i - gap
        if end > start:
            height = end - start + 1
            # Ignore page borders / huge background bands.
            touches_edge = start <= 2 or end >= h - 3
            if not touches_edge and height <= max(12, int(h * 0.24)):
                strength = float(row_score[start:end + 1].sum())
                groups.append((start, end, strength))

    if len(groups) < 2:
        return []

    # The two MRZ rows are normally the strongest two broad text bands.
    # Prefer groups that are separated vertically and reasonably similar in
    # height; this suppresses isolated passport artwork/stamp edges.
    groups.sort(key=lambda x: x[2], reverse=True)
    selected = []
    for group in groups:
        if all(abs(((group[0] + group[1]) / 2) - ((g[0] + g[1]) / 2)) > max(6, h * 0.055) for g in selected):
            selected.append(group)
        if len(selected) == 2:
            break

    if len(selected) != 2:
        return []

    selected.sort(key=lambda x: x[0])
    return [(a, b) for a, b, _ in selected]


def _mrz_row_variants(row: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return OCR variants for one detected MRZ row."""
    if row is None or row.size == 0:
        return []

    # Add a little vertical context; too-tight crops clip ascenders/descenders.
    gray = row if row.ndim == 2 else cv2.cvtColor(row, cv2.COLOR_RGB2GRAY)
    variants = [("gray", gray)]

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    variants.append(("otsu", cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]))

    adaptive = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11,
    )
    variants.append(("adaptive", adaptive))

    # A mild unsharp mask helps faint MRZ strokes without changing the source.
    sharpen = cv2.addWeighted(gray, 1.7, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.7, 0)
    variants.append(("sharp", sharpen))
    return variants


def _mrz_attempts_from_part3(part3: Image.Image) -> list[dict]:
    """Generate MRZ OCR views with explicit first-row/second-row identity.

    The previous implementation cut Part 3 using fixed 44/56% bands. That
    often caused PSM 7 to see pieces of both rows and produced nonsense such as
    treating the second row as a first row. Here the two printed rows are first
    located from the horizontal ink profile, then OCR'd independently.
    """
    gray_image = _mrz_upscale(part3, target_width=2800)
    gray = np.array(gray_image)
    h, w = gray.shape[:2]
    variants: list[dict] = []

    def add(name: str, arr: np.ndarray, row: str | None = None, kind: str = "block"):
        if arr is None or arr.size == 0:
            return
        variants.append({
            "name": name,
            "kind": kind,
            "row": row,
            "images": [_cv2_to_pil(np.ascontiguousarray(arr))],
        })

    # Keep a couple of block views for recovery when row detection is poor.
    for start_frac, label in ((0.00, "part3-full"), (0.10, "part3-lower90"), (0.18, "part3-lower82")):
        y0 = int(round(h * start_frac))
        crop = gray[y0:, :]
        if crop.shape[0] >= 60:
            add(f"{label}-gray", crop, None, "block")
            add(f"{label}-gray-sparse", crop, None, "sparse")
            add(
                f"{label}-otsu",
                cv2.threshold(cv2.GaussianBlur(crop, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                None,
                "block",
            )
            add(
                f"{label}-otsu-sparse",
                cv2.threshold(cv2.GaussianBlur(crop, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                None,
                "sparse",
            )

    bands = _detect_mrz_row_bands(gray)
    if len(bands) == 2:
        # Add several paddings around the detected row.  This is deliberately
        # wider than the character strokes but never includes the other row.
        for row_index, (start, end) in enumerate(bands, start=1):
            row_name = "line1" if row_index == 1 else "line2"
            for pad in (3, 5, 7):
                y0 = max(0, start - pad)
                y1 = min(h, end + pad + 1)
                row = gray[y0:y1, :]
                for mode, arr in _mrz_row_variants(row):
                    add(
                        f"detected-{row_name}-pad{pad}-{mode}",
                        arr,
                        row_name,
                        "lines",
                    )

    # Fallback overlapping row cuts. These are explicitly labelled so the
    # parser can still pair them by row when the projection detector fails.
    if len(bands) != 2:
        for top, bottom, row_name in (
            (0.10, 0.55, "line1"),
            (0.45, 0.90, "line2"),
            (0.06, 0.52, "line1"),
            (0.48, 0.94, "line2"),
        ):
            y0, y1 = int(h * top), int(h * bottom)
            row = gray[y0:y1, :]
            if row.shape[0] >= 25:
                for mode, arr in _mrz_row_variants(row):
                    add(f"fallback-{row_name}-{top:.2f}-{mode}", arr, row_name, "lines")

    return variants

def _normalize_mrz_ocr_text(text: str) -> str:
    """Normalize OCR noise without destroying potentially useful evidence."""
    if not text:
        return ""
    text = text.upper().replace("\u00ab", "<").replace("\u00bb", "<")
    text = text.replace("|", "I")
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"[^A-Z0-9<]", "", raw.upper())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _visual_mrz_ocr(part3: Image.Image) -> dict:
    """Independent MRZ OCR confirmation using both dense and sparse layouts."""
    gray_image = _mrz_upscale(part3, target_width=2800)
    cv = np.array(gray_image)
    h = cv.shape[0]
    candidate_images = []

    for start_frac in (0.0, 0.08, 0.16):
        y0 = int(round(h * start_frac))
        candidate_images.append((f"lower-{start_frac:.2f}", cv[y0:, :]))

    best_text = ""
    best_score = -1.0
    attempts = []

    for crop_name, arr in candidate_images:
        if arr.size == 0:
            continue
        for mode, image_arr in (
            ("gray", arr),
            ("otsu", cv2.threshold(
                cv2.GaussianBlur(arr, (3, 3), 0),
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1]),
        ):
            for layout, config in (
                ("block", MRZ_BLOCK_CONFIG),
                ("sparse", MRZ_SPARSE_CONFIG),
            ):
                try:
                    raw = pytesseract.image_to_string(
                        _cv2_to_pil(image_arr),
                        config=config,
                    )
                except Exception:
                    raw = ""

                normalized = _normalize_mrz_ocr_text(raw)
                lengths = [len(x) for x in normalized.splitlines() if x]
                long_lines = sum(1 for n in lengths if n >= 30)
                near_td3 = sum(1 for n in lengths if 38 <= n <= 48)
                has_start = int(any(
                    line.startswith(("P<", "V<"))
                    for line in normalized.splitlines()
                    if line
                ))
                score = (
                    long_lines * 80.0
                    + near_td3 * 120.0
                    + has_start * 160.0
                    + min(max(lengths, default=0), 44)
                )
                attempts.append({
                    "name": f"visual-mrz-{crop_name}-{mode}-{layout}",
                    "raw": raw,
                    "normalized": normalized,
                    "score": score,
                })
                if score > best_score:
                    best_score = score
                    best_text = normalized

    return {
        "text": best_text,
        "attempts": attempts,
    }


def _independent_visual_ocr_part3(part3: Image.Image) -> dict:
    """Run ordinary, non-MRZ-specific OCR over Part 3.

    This is deliberately separate from the MRZ whitelist/parser path.  It keeps
    normal OCR characters and uses the printed MRZ region as an ordinary text
    image so that a second OCR engine configuration can independently challenge
    the MRZ parser's spelling.
    """
    work = _mrz_upscale(part3, target_width=3200)
    arr = np.array(work.convert("L"))
    h, w = arr.shape[:2]

    variants = []
    for name, image_arr in (("gray", arr),):
        variants.append((name, image_arr))
        variants.append((
            f"otsu-{name}",
            cv2.threshold(
                cv2.GaussianBlur(image_arr, (3, 3), 0),
                0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[1],
        ))
        variants.append((
            f"adaptive-{name}",
            cv2.adaptiveThreshold(
                image_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 11,
            ),
        ))

    attempts = []
    configs = (
        ("psm6", "--oem 3 --psm 6"),
        ("psm11", "--oem 3 --psm 11"),
        ("psm12", "--oem 3 --psm 12"),
    )

    for variant_name, image_arr in variants:
        for config_name, config in configs:
            try:
                raw = pytesseract.image_to_string(
                    _cv2_to_pil(image_arr),
                    config=config,
                )
            except Exception:
                raw = ""
            lines = []
            for raw_line in raw.splitlines():
                line = re.sub(r"[^A-Za-z0-9<]", "", raw_line.upper())
                if line:
                    lines.append(line)
            text = "\n".join(lines)
            score = 0.0
            for line in lines:
                if 35 <= len(line) <= 48:
                    score += 80
                if line.startswith(("P<", "V<")):
                    score += 120
                if sum(c.isdigit() for c in line) >= 5:
                    score += 30
            attempts.append({
                "name": f"part3-visual-{variant_name}-{config_name}",
                "raw": raw,
                "text": text,
                "lines": lines,
                "score": score,
            })

    best = max(attempts, key=lambda x: x["score"], default=None)
    return {
        "text": best["text"] if best else "",
        "lines": best["lines"] if best else [],
        "attempts": attempts,
        "method": "independent-normal-visual-ocr-on-part3",
        "image_size": [w, h],
    }


def _part3_visual_mrz_candidate(visual: dict) -> dict | None:
    """Turn ordinary Part-3 visual OCR into a candidate without trusting it."""
    lines = list(visual.get("lines", []) or [])
    line1 = next(
        (x for x in lines if x.startswith(("P<", "V<")) and len(x) >= 28),
        None,
    )
    line2 = next(
        (x for x in lines if x != line1 and sum(c.isdigit() for c in x) >= 5 and len(x) >= 28),
        None,
    )
    return {"line1": line1, "line2": line2} if line1 or line2 else None

def _parse_all_mrz_attempts(attempts: list[dict]) -> tuple[dict | None, str | None, dict]:
    """Parse MRZ attempts while preserving the physical identity of each row."""
    parsed_attempts = []
    texts = {}
    line1_candidates = []
    line2_candidates = []

    for attempt in attempts:
        name = attempt["name"]
        config = MRZ_LINE_CONFIG if attempt["kind"] == "lines" else (MRZ_SPARSE_CONFIG if attempt["kind"] == "sparse" else MRZ_BLOCK_CONFIG)
        combined = []
        for image in attempt.get("images", []):
            try:
                raw = pytesseract.image_to_string(image, config=config)
            except Exception:
                raw = ""
            normalized = _normalize_mrz_ocr_text(raw)
            if normalized:
                combined.append(normalized)

        normalized_text = "\n".join(combined)
        texts[name] = normalized_text
        lines = [x for x in normalized_text.splitlines() if len(x) >= 20]

        row = attempt.get("row")
        if row in {"line1", "line2"}:
            for line in lines:
                compact = line.replace("\n", "")
                if row == "line1":
                    line1_candidates.append((name, compact))
                else:
                    line2_candidates.append((name, compact))

        # Block OCR can still contain a complete pair.
        candidates = [normalized_text]
        if len(lines) >= 2:
            candidates.append("\n".join(lines[-2:]))
        for candidate in candidates:
            try:
                parsed = parse_mrz(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("success"):
                parsed_attempts.append((name, parsed, normalized_text))

    # If the row detector was unavailable, classify unlabelled long OCR lines
    # conservatively. A line beginning P</V< can ONLY be line 1; a line with
    # several digits is a line 2 candidate. Never pair two line-2 candidates.
    for name, text in list(texts.items()):
        for line in [x for x in text.splitlines() if len(x) >= 20]:
            compact = line.replace("\n", "")
            if compact and compact[0] in {"P", "V"} and len(compact) >= 28 and compact[1:2] == "<" and "<<" in compact[5:]:
                line1_candidates.append((name, compact))
            elif sum(c.isdigit() for c in compact) >= 5:
                line2_candidates.append((name, compact))

    def unique(items):
        out, seen = [], set()
        for name, line in items:
            key = line
            if key in seen:
                continue
            seen.add(key)
            out.append((name, line))
        return out

    line1_candidates = unique(line1_candidates)
    line2_candidates = unique(line2_candidates)

    def line1_score(item):
        _, line = item
        score = 0
        score += 100 if len(line) == 44 else max(0, 40 - abs(len(line) - 44) * 4)
        score += 100 if line.startswith(("P<", "V<")) else 0
        score += 80 if "<<" in line[5:] else 0
        score += min(40, line.count("<"))
        score -= 3 * sum(c.isdigit() for c in line)
        return score

    def line2_score(item):
        _, line = item
        score = 100 if len(line) == 44 else max(0, 45 - abs(len(line) - 44) * 5)
        score += min(80, sum(c.isdigit() for c in line) * 4)
        return score

    line1_candidates.sort(key=line1_score, reverse=True)
    line2_candidates.sort(key=line2_score, reverse=True)

    # Pair ONLY a first-row candidate with a second-row candidate. This fixes
    # the previous failure mode where two noisy second rows were combined.
    for n1, l1 in line1_candidates[:12]:
        for n2, l2 in line2_candidates[:12]:
            try:
                parsed = parse_mrz(f"{l1}\n{l2}")
            except Exception:
                continue
            if isinstance(parsed, dict) and parsed.get("success"):
                parsed_attempts.append((f"{n1}+{n2}", parsed, f"{l1}\n{l2}"))

    if not parsed_attempts:
        return None, None, texts

    def score(item):
        name, parsed, raw = item
        line1 = parsed.get("line1", "")
        line2 = parsed.get("line2", "")
        checks = _critical_check_count(parsed)
        exact = int(len(line1) == 44) + int(len(line2) == 44)
        structure = int(line1.startswith(("P<", "V<"))) + int("<<" in line1[5:])
        names = parsed.get("data", {})
        nonempty = sum(bool(names.get(k)) for k in ("surname", "given_names", "passport_number", "nationality", "date_of_birth", "date_of_expiry"))
        return checks * 1000 + exact * 500 + structure * 180 + nonempty * 40 + _mrz_quality(parsed)

    best_name, best, _ = max(parsed_attempts, key=score)
    return best, best_name, texts

def read_stage2_mrz(part3: Image.Image) -> dict:
    """MRZ-first Stage 2 reader.

    Two independent OCR paths are used:
      1. Existing MRZ OCR + parser + check-digit validation.
      2. Independent raw visual OCR of the same MRZ region.

    The parser/check digits decide the structured result; visual OCR is
    evidence used to confirm that the selected lines really look like the
    printed MRZ.
    """
    attempts = _mrz_attempts_from_part3(part3)
    best, best_name, texts = _parse_all_mrz_attempts(attempts)

    # The old extractor ensemble is a fallback only. Running it after every
    # direct line-by-line parse duplicates a large amount of Tesseract work.
    # Use it only when the dedicated MRZ parser could not produce a candidate.
    legacy_best = None
    legacy_name = None
    legacy_texts = {}
    if best is None:
        try:
            legacy_best, legacy_name, legacy_texts = _read_mrz(attempts)
        except Exception:
            legacy_best, legacy_name, legacy_texts = None, None, {}

    candidates = []
    if best is not None:
        candidates.append((best_name, best))
    if legacy_best is not None:
        candidates.append((legacy_name, legacy_best))

    if candidates:
        selected_name, selected = max(
            candidates,
            key=lambda item: (
                _critical_check_count(item[1]),
                _mrz_quality(item[1]),
                int(len(item[1].get("line1", "")) == 44),
                int(len(item[1].get("line2", "")) == 44),
            ),
        )
    else:
        selected_name, selected = None, None

    # SECOND PASS: ordinary visual OCR over the exact same Part 3 image.
    # This happens only after the MRZ parser has finished, so the two paths are
    # genuinely independent and can challenge one another.
    part3_visual = _independent_visual_ocr_part3(part3)
    part3_visual_candidate = _part3_visual_mrz_candidate(part3_visual)

    visual_mrz = _visual_mrz_ocr(part3)
    parser_lines = "".join(
        [
            selected.get("line1", "") if selected else "",
            selected.get("line2", "") if selected else "",
        ]
    )
    visual_lines = re.sub(
        r"[^A-Z0-9<]",
        "",
        visual_mrz.get("text", "").upper(),
    )
    mrz_visual_similarity = (
        SequenceMatcher(None, parser_lines, visual_lines).ratio()
        if parser_lines and visual_lines
        else 0.0
    )

    trustworthy = _mrz_trustworthy(selected)

    debug = {
        "attempt_names": [a["name"] for a in attempts],
        "ocr_texts": texts,
        "legacy_ocr_texts": legacy_texts,
        "selected_variant": selected_name,
        "trustworthy": trustworthy,
        "critical_checks": _critical_check_count(selected) if selected else 0,
        "quality_score": _mrz_quality(selected) if selected else 0,
        "validation": selected.get("validation") if selected else None,
        "lines": [
            selected.get("line1") if selected else None,
            selected.get("line2") if selected else None,
        ],
        "visual_mrz_ocr": visual_mrz,
        "part3_visual_ocr": part3_visual,
        "part3_visual_candidate": part3_visual_candidate,
        "visual_mrz_similarity": round(mrz_visual_similarity, 3),
    }

    return {
        "result": selected,
        "trustworthy": trustworthy,
        "debug": debug,
    }


def _mrz_trustworthy(mrz_result: dict | None) -> bool:
    if not isinstance(mrz_result, dict) or not mrz_result.get("success"):
        return False

    validation = mrz_result.get("validation", {})
    line1 = mrz_result.get("line1", "")
    nationality = mrz_result.get("data", {}).get("nationality", "")

    # Support TD3 passport and visa-style rows. Do not require the issuing
    # country/nationality to be present in an ISO database: synthetic/test
    # documents can legitimately use structurally valid codes such as UTO/XXX.
    if len(line1) != 44 or line1[0] not in {"P", "V"} or line1[1] != "<":
        return False
    if not validation.get("country_format_valid"):
        return False
    if not validation.get("nationality_format_valid"):
        return False
    if not nationality:
        return False
    if not validation.get("birth_date_format_valid", True):
        return False
    if not validation.get("expiry_date_format_valid", True):
        return False
    if not validation.get("sex_format_valid", True):
        return False

    # Keep the minimum check-digit requirement, but reject structurally
    # malformed dates/sex values even if an OCR error happens to preserve a
    # check digit. Synthetic/unknown country codes remain acceptable.
    return _critical_check_count(mrz_result) >= 2


# ------------------------------------------------------------
# Stage 2 visual OCR
# ------------------------------------------------------------


def _merge_visual_parts(results: list[dict]) -> tuple[str, dict]:
    """Merge Part 1/Part 2 visual OCR without re-running page cropping."""
    results = [r for r in results if isinstance(r, dict)]
    full_text_parts = []
    merged: dict = {}

    for result in results:
        meta = result.get("_meta", {})
        if isinstance(meta, dict) and meta.get("full_text"):
            full_text_parts.append(str(meta["full_text"]))

    for field in (
        "surname",
        "given_names",
        "nationality",
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
        "sex",
        "passport_number",
    ):
        candidates = []
        for result in results:
            item = result.get(field, {})
            if not isinstance(item, dict):
                continue
            value = item.get("value")
            if value in (None, ""):
                continue
            try:
                confidence = float(item.get("confidence", 0) or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            candidates.append((confidence, item))

        if candidates:
            # Highest confidence candidate wins. Keep the original field item
            # because the validator already understands its shape/method.
            candidates.sort(key=lambda x: x[0], reverse=True)
            merged[field] = dict(candidates[0][1])
        else:
            merged[field] = {
                "value": None,
                "raw": "",
                "confidence": 0.0,
                "method": "no-candidate",
            }

    date_candidates = set()
    date_scan_texts = {}
    identity_sizes = []

    for index, result in enumerate(results):
        meta = result.get("_meta", {})
        if not isinstance(meta, dict):
            continue
        identity_sizes.append(meta.get("identity_size"))
        for value in meta.get("date_candidates", []) or []:
            try:
                date_candidates.add(datetime.date.fromisoformat(str(value)).isoformat())
            except (TypeError, ValueError):
                continue
        for name, text in (meta.get("date_scan_texts", {}) or {}).items():
            date_scan_texts[f"part{index + 1}-{name}"] = text

    merged["_meta"] = {
        "full_text": "\n".join(full_text_parts),
        "identity_size": identity_sizes,
        "date_candidates": sorted(date_candidates),
        "date_scan_texts": date_scan_texts,
    }
    return merged["_meta"]["full_text"], merged


def _filter_visual_fields(result: dict, allowed: set[str]) -> dict:
    """Keep only fields that physically belong to this passport split."""
    result = result if isinstance(result, dict) else {}
    filtered = {
        field: result.get(
            field,
            {"value": None, "raw": "", "confidence": 0.0, "method": "region-not-scanned"},
        )
        for field in allowed
    }
    meta = result.get("_meta", {})
    if isinstance(meta, dict):
        filtered["_meta"] = meta
    return filtered


def _visual_pass_parts(parts: dict[str, Image.Image]) -> dict:
    """Run visual OCR only after MRZ processing, using physical field locations.

    Part 1: passport number + surname + given names.
    Part 2: dates.
    Part 3: MRZ is handled exclusively by read_stage2_mrz() before this function.
    """
    # Part 1 is intentionally limited to identity/document-number fields.
    part1_raw = extract_visual_fields(
        parts["part1"],
        region_mode=True,
    )
    part1 = _filter_visual_fields(
        part1_raw,
        {"passport_number", "surname", "given_names"},
    )

    # Part 2 is intentionally limited to the three visual dates. The existing
    # date-role inference is then applied to these candidates only.
    part2_raw = extract_visual_fields(
        parts["part2"],
        region_mode=True,
    )
    part2 = _filter_visual_fields(
        part2_raw,
        {"date_of_birth", "date_of_issue", "date_of_expiry"},
    )

    visual_text, visual_fields = _merge_visual_parts(
        [part1, part2]
    )

    return {
        "part_results": {
            "part1": part1,
            "part2": part2,
            # Part 3 is deliberately not sent through the normal visual-field
            # extractor. Its OCR is MRZ-specific and was completed first.
            "part3": {
                "_meta": {
                    "full_text": "",
                    "purpose": "MRZ handled by dedicated MRZ OCR/parser before visual VIZ OCR",
                }
            },
        },
        "visual_text": visual_text,
        "visual_fields": visual_fields,
        "part3_visual_text": "",
    }


def _manual_visual_ocr_all_parts(parts: dict[str, Image.Image]) -> dict:
    """Final independent visual pass over all three physical split regions.

    This is deliberately a *last-pass* safety net. It does not replace the
    existing MRZ/Part-1/Part-2 pipeline. It simply asks the ordinary visual OCR
    stack to look at each split independently, including Part 3, so a final
    cross-validated guess can be produced when the primary result is missing,
    partial, or ambiguous.
    """
    results = {}
    merged_text = []

    for name in ("part1", "part2", "part3"):
        image = parts.get(name)
        if image is None:
            results[name] = {"fields": {}, "text": "", "raw": {}}
            continue
        try:
            raw = extract_visual_fields(image, region_mode=True)
            fields = {
                key: value
                for key, value in raw.items()
                if key != "_meta"
            }
            meta = raw.get("_meta", {}) if isinstance(raw, dict) else {}
            text = meta.get("full_text", "") if isinstance(meta, dict) else ""
        except Exception as exc:
            raw = {}
            fields = {}
            text = ""
            results[name] = {
                "fields": {},
                "text": "",
                "raw": {},
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue

        results[name] = {
            "fields": fields,
            "text": text,
            "raw": raw,
        }
        if text:
            merged_text.append(f"[{name}]\n{text}")

    # Part 3 gets an additional ordinary OCR pass that is intentionally MRZ-
    # aware only for candidate extraction, not for the actual OCR itself.
    part3 = parts.get("part3")
    if part3 is not None:
        try:
            p3_mrz_visual = _independent_visual_ocr_part3(part3)
        except Exception as exc:
            p3_mrz_visual = {"text": "", "lines": [], "attempts": [], "error": f"{type(exc).__name__}: {exc}"}
        results.setdefault("part3", {})["independent_mrz_visual"] = p3_mrz_visual
        if p3_mrz_visual.get("text"):
            merged_text.append("[part3-independent]\n" + p3_mrz_visual["text"])

    return {
        "parts": results,
        "text": "\n\n".join(merged_text),
        "method": "final-independent-visual-ocr-all-three-parts",
    }


def _field_candidate_from_visual(item: dict, field: str):
    """Safely pull a value from visual_ocr's structured field object."""
    if not isinstance(item, dict):
        return None
    value = item.get(field)
    if isinstance(value, dict):
        value = value.get("value")
    return value if value not in (None, "") else None


def _normalise_guess_name(value):
    if value in (None, ""):
        return None
    value = str(value).upper().replace("<", " ")
    value = re.sub(r"[^A-Z ]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _normalise_guess_alnum(value):
    if value in (None, ""):
        return None
    value = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    return value or None


def _normalise_guess_date(value):
    if value in (None, ""):
        return None
    text = str(value).strip()
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _recover_td3_fields_from_noisy_text(texts) -> list[dict]:
    """Recover individually readable TD3 fields even when a full MRZ parse fails.

    This is deliberately field-wise. A dropped passport-number character must not
    make a perfectly readable DOB, sex, or expiry disappear. We use the TD3
    anchors around nationality/date/sex/date and validate the two date check
    digits independently. The result is evidence for the final-guess layer and
    never silently replaces the primary Stage-2 result.
    """
    if isinstance(texts, dict):
        raw_values = list(texts.values())
    elif isinstance(texts, (list, tuple)):
        raw_values = list(texts)
    else:
        raw_values = [texts]

    digit_map = str.maketrans({
        "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
        "Z": "2", "S": "5", "G": "6", "B": "8",
    })
    alpha_map = str.maketrans({
        "1": "I", "0": "O", "5": "S", "2": "Z", "8": "B", "6": "G",
    })

    recovered = []
    seen = set()
    for raw in raw_values:
        for raw_line in str(raw or "").splitlines():
            line = re.sub(r"[^A-Z0-9<]", "", raw_line.upper())
            if len(line) < 25 or sum(c.isdigit() for c in line) < 4:
                continue

            # Keep the date digits untouched. Only the three-character
            # nationality slot needs alpha/digit confusion repair (for example
            # OCR often returns 1ND for IND).
            match = re.search(
                r"([A-Z1-9]{3})([0-9OQIDZSGB]{6})([0-9OQIDZSGB])([MF<])([0-9OQIDZSGB]{6})([0-9OQIDZSGB])",
                line,
            )
            if not match:
                continue

            nat = match.group(1).translate(alpha_map)
            dob = match.group(2).translate(digit_map)
            dob_cd = match.group(3).translate(digit_map)
            sex = match.group(4)
            exp = match.group(5).translate(digit_map)
            exp_cd = match.group(6).translate(digit_map)

            # Validate the date fields independently. This is the key safety
            # property: we can recover dates even when the passport-number
            # prefix is incomplete or shifted.
            try:
                dob_ok = bool(re.fullmatch(r"\d{6}", dob)) and _mrz_check_digit(dob) == dob_cd
                exp_ok = bool(re.fullmatch(r"\d{6}", exp)) and _mrz_check_digit(exp) == exp_cd
            except Exception:
                dob_ok = exp_ok = False

            if not (dob_ok or exp_ok):
                continue

            before_nat = line[:match.start()]
            # The character immediately before nationality is the passport-number
            # check digit. Everything before it is the OCR passport-number prefix.
            partial_number = before_nat[:-1] if len(before_nat) >= 2 and before_nat[-1].isdigit() else before_nat
            partial_number = re.sub(r"[^A-Z0-9]", "", partial_number)

            item = {
                "source_line": line,
                "nationality": nat,
                "date_of_birth_raw": dob if dob_ok else None,
                "date_of_birth_check_valid": dob_ok,
                "sex": sex if sex in {"M", "F"} else None,
                "date_of_expiry_raw": exp if exp_ok else None,
                "date_of_expiry_check_valid": exp_ok,
                "partial_passport_number": partial_number or None,
                "field_validation": {
                    "nationality_format_valid": bool(re.fullmatch(r"[A-Z]{3}", nat)),
                    "birth_date_valid": dob_ok,
                    "expiry_date_valid": exp_ok,
                    "sex_format_valid": sex in {"M", "F", "<"},
                },
            }
            key = (
                item["nationality"], item["date_of_birth_raw"], item["sex"],
                item["date_of_expiry_raw"], item["partial_passport_number"],
            )
            if key not in seen:
                seen.add(key)
                recovered.append(item)

    return recovered


def _mrz_check_digit(value: str) -> str:
    """Local ICAO check digit helper used by loose field recovery."""
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(value):
        if ch.isdigit():
            n = int(ch)
        elif "A" <= ch <= "Z":
            n = ord(ch) - 55
        else:
            n = 0
        total += n * weights[i % 3]
    return str(total % 10)


def _extract_independent_part3_names(text: str, nationality_hint: str | None = None) -> tuple[str | None, str | None]:
    """Read names from ordinary Part-3 OCR without requiring a perfect MRZ line."""
    surname = given = None
    nat = (nationality_hint or "").upper()
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"[^A-Z0-9<]", "", raw_line.upper())
        if "<<" not in line:
            continue
        left, right = line.split("<<", 1)
        # Common OCR collapse: P<IND may become FIND. If the known nationality
        # occurs in the prefix, strip it and use everything after it as surname.
        name_left = None
        if nat and nat in left:
            idx = left.rfind(nat)
            name_left = left[idx + 3:]
        elif left.startswith(("P<", "V<")) and len(left) > 5:
            name_left = left[5:]
        elif left.startswith(("FIND", "PIND", "VIND")):
            name_left = left[4:]
        else:
            # Ordinary OCR often turns P<IND into a noisy four-character prefix.
            # Remove a short leading block only when the remaining text is a
            # plausible name.
            m = re.search(r"[A-Z]{3,}", left)
            name_left = left[m.start():] if m else left

        name_left = re.sub(r"[^A-Z ]", " ", name_left or "")
        name_left = re.sub(r"\s+", " ", name_left).strip()
        # Stop at the long MRZ filler run. Single < characters between given
        # names are spaces; a run of three or more is the end of the name.
        right = re.split(r"<{3,}", right, maxsplit=1)[0]
        right = right.replace("<", " ")
        right = re.sub(r"[^A-Z ]", " ", right)
        right = re.sub(r"\s+", " ", right).strip()

        if name_left and len(name_left) >= 2 and re.fullmatch(r"[A-Z]+(?: [A-Z]+)*", name_left):
            surname = name_left
        if right and re.fullmatch(r"[A-Z]+(?: [A-Z]+)*", right):
            given = right
        if surname or given:
            return surname, given
    return None, None


def _mrz_date_to_iso_guess(value: str | None, kind: str) -> str | None:
    if not value or not re.fullmatch(r"\d{6}", str(value)):
        return None
    yy, mm, dd = int(value[:2]), int(value[2:4]), int(value[4:6])
    if kind == "birth":
        year = 2000 + yy if 2000 + yy <= datetime.date.today().year else 1900 + yy
    else:
        year = 2000 + yy
    try:
        return datetime.date(year, mm, dd).isoformat()
    except ValueError:
        return None


def _build_final_guess(
    current_result: dict,
    mrz_stage2: dict,
    visual: dict,
    manual_visual: dict,
) -> dict:
    """Produce a separately displayed, conservative final guess.

    The guess never mutates the normal Stage 2 result. A field is accepted only
    when it has structural validation (MRZ/check-digit or valid date/format) or
    independent visual corroboration. Ambiguous disagreements remain flagged
    instead of being silently invented.
    """
    data = current_result.get("data", {}) if isinstance(current_result, dict) else {}
    mrz_result = mrz_stage2.get("result") if isinstance(mrz_stage2, dict) else None
    manual_parts = manual_visual.get("parts", {}) if isinstance(manual_visual, dict) else {}

    def field_value(part_name, field):
        part = manual_parts.get(part_name, {})
        fields = part.get("fields", {}) if isinstance(part, dict) else {}
        return _field_candidate_from_visual(fields, field)

    p1_number = field_value("part1", "passport_number")
    p1_surname = field_value("part1", "surname")
    p1_given = field_value("part1", "given_names")
    p1_dob = field_value("part1", "date_of_birth")
    p1_issue = field_value("part1", "date_of_issue")
    p1_exp = field_value("part1", "date_of_expiry")

    p2_dob = field_value("part2", "date_of_birth")
    p2_issue = field_value("part2", "date_of_issue")
    p2_exp = field_value("part2", "date_of_expiry")
    p2_sex = field_value("part2", "sex")

    p3_fields = manual_parts.get("part3", {}).get("fields", {})
    p3_number = _field_candidate_from_visual(p3_fields, "passport_number")
    p3_surname = _field_candidate_from_visual(p3_fields, "surname")
    p3_given = _field_candidate_from_visual(p3_fields, "given_names")
    p3_nat = _field_candidate_from_visual(p3_fields, "nationality")
    p3_dob = _field_candidate_from_visual(p3_fields, "date_of_birth")
    p3_sex = _field_candidate_from_visual(p3_fields, "sex")
    p3_exp = _field_candidate_from_visual(p3_fields, "date_of_expiry")

    # The dedicated Part-3 ordinary OCR can be parsed independently through the
    # same positional MRZ parser, but only as evidence for the final guess.
    independent_p3 = manual_parts.get("part3", {}).get("independent_mrz_visual", {})
    independent_candidate = _part3_visual_mrz_candidate(independent_p3) if independent_p3 else None

    # Field-wise MRZ recovery is intentionally separate from the full parser.
    # It lets a valid DOB/sex/expiry survive when OCR dropped passport-number
    # characters or shifted the line.
    loose_mrz = _recover_td3_fields_from_noisy_text(
        (mrz_stage2.get("debug", {}) if isinstance(mrz_stage2, dict) else {}).get("ocr_texts", {})
    )
    loose_best = None
    if loose_mrz:
        loose_best = max(
            loose_mrz,
            key=lambda x: sum(bool(x.get(k)) for k in ("date_of_birth_raw", "date_of_expiry_raw", "sex", "nationality"))
        )
        p3_nat = p3_nat or loose_best.get("nationality")
        p3_dob = p3_dob or loose_best.get("date_of_birth_raw")
        p3_sex = p3_sex or loose_best.get("sex")
        p3_exp = p3_exp or loose_best.get("date_of_expiry_raw")
    if independent_candidate and independent_candidate.get("line1") and independent_candidate.get("line2"):
        try:
            parsed_p3 = parse_mrz(
                independent_candidate["line1"] + "\n" + independent_candidate["line2"]
            )
        except Exception:
            parsed_p3 = None
        if isinstance(parsed_p3, dict) and parsed_p3.get("success"):
            pd = parsed_p3.get("data", {})
            p3_number = p3_number or pd.get("passport_number")
            p3_surname = p3_surname or pd.get("surname")
            p3_given = p3_given or pd.get("given_names")
            p3_nat = p3_nat or pd.get("nationality")
            p3_dob = p3_dob or pd.get("date_of_birth")
            p3_sex = p3_sex or pd.get("sex")
            p3_exp = p3_exp or pd.get("date_of_expiry")

    # If the full Part-3 parser returned a malformed one-character nationality
    # or shifted dates, prefer the field-wise recovery because its date fields
    # have already passed their own check digits.
    if loose_best:
        if loose_best.get("field_validation", {}).get("nationality_format_valid"):
            p3_nat = loose_best.get("nationality") or p3_nat
        if loose_best.get("field_validation", {}).get("birth_date_valid"):
            p3_dob = loose_best.get("date_of_birth_raw") or p3_dob
        if loose_best.get("field_validation", {}).get("expiry_date_valid"):
            p3_exp = loose_best.get("date_of_expiry_raw") or p3_exp
        if loose_best.get("field_validation", {}).get("sex_format_valid"):
            p3_sex = loose_best.get("sex") or p3_sex

    raw_surname, raw_given = _extract_independent_part3_names(
        independent_p3.get("text", "") if isinstance(independent_p3, dict) else "",
        p3_nat,
    )
    p3_surname = raw_surname or p3_surname
    p3_given = raw_given or p3_given

    candidates = {}
    mrz_data = mrz_result.get("data", {}) if isinstance(mrz_result, dict) else {}
    mrz_valid = bool(mrz_result and mrz_result.get("success"))

    def add(field, value, source):
        if value not in (None, ""):
            candidates.setdefault(field, []).append((str(value).strip(), source))

    # Primary result is evidence, not automatic truth for the final guess.
    for field in FIELDS:
        add(field, data.get(field), "current-stage2")
        add(field, mrz_data.get(field), "mrz-parser")

    add("passport_number", p1_number, "part1-visual")
    add("passport_number", p3_number, "part3-visual")
    add("surname", p1_surname, "part1-visual")
    add("surname", p3_surname, "part3-visual")
    add("given_names", p1_given, "part1-visual")
    add("given_names", p3_given, "part3-visual")
    add("nationality", p3_nat, "part3-visual")
    add("date_of_birth", p2_dob, "part2-visual")
    add("date_of_birth", p3_dob, "part3-visual")
    add("sex", p2_sex, "part2-visual")
    add("sex", p3_sex, "part3-visual")
    add("date_of_issue", p2_issue or p1_issue, "part2-visual")
    add("date_of_expiry", p2_exp, "part2-visual")
    add("date_of_expiry", p3_exp, "part3-visual")

    loose_valid = {}
    if loose_best:
        if loose_best.get("nationality"):
            add("nationality", loose_best.get("nationality"), "mrz-loose-field")
            loose_valid["nationality"] = bool(loose_best.get("field_validation", {}).get("nationality_format_valid"))
        partial = _normalise_guess_alnum(loose_best.get("partial_passport_number"))
        p1_norm = _normalise_guess_alnum(p1_number)
        p3_norm = _normalise_guess_alnum(p3_number)
        # A partially visible MRZ passport number is not sufficient by itself.
        # It becomes strong evidence when an independent Part-1/Part-3 visual
        # candidate contains the same long suffix/prefix. This is how a dropped
        # leading "ZA" can be recovered without inventing characters.
        if partial and len(partial) >= 5:
            for candidate in (p1_norm, p3_norm):
                if candidate and (partial in candidate or candidate in partial):
                    loose_valid["passport_number"] = True
                    break
        if loose_best.get("date_of_birth_raw"):
            add("date_of_birth", _mrz_date_to_iso_guess(loose_best.get("date_of_birth_raw"), "birth"), "mrz-loose-field")
            loose_valid["date_of_birth"] = bool(loose_best.get("field_validation", {}).get("birth_date_valid"))
        if loose_best.get("date_of_expiry_raw"):
            add("date_of_expiry", _mrz_date_to_iso_guess(loose_best.get("date_of_expiry_raw"), "expiry"), "mrz-loose-field")
            loose_valid["date_of_expiry"] = bool(loose_best.get("field_validation", {}).get("expiry_date_valid"))
        if loose_best.get("sex"):
            add("sex", loose_best.get("sex"), "mrz-loose-field")
            loose_valid["sex"] = bool(loose_best.get("field_validation", {}).get("sex_format_valid"))

    guess = {}
    validation = {}
    source_map = {}

    # Passport-number recovery: when MRZ OCR lost leading characters, do not
    # accept the truncated parser value. A Part-1 visual candidate is accepted
    # only when the visible MRZ prefix independently contains at least five
    # matching characters. This turns e.g. 567643 + Part-1 ZA567643 into a
    # validated recovery without guessing arbitrary missing characters.
    partial = _normalise_guess_alnum(loose_best.get("partial_passport_number")) if loose_best else None
    for candidate, source in ((p1_number, "part1-visual"), (p3_number, "part3-visual")):
        normalized = _normalise_guess_alnum(candidate)
        if partial and normalized and len(partial) >= 5 and (partial in normalized or normalized in partial):
            if re.fullmatch(r"[A-Z0-9]{6,12}", normalized):
                guess["passport_number"] = normalized
                validation["passport_number"] = "validated — visual + MRZ positional anchor"
                source_map["passport_number"] = [source, "mrz-loose-field"]
                break

    # Strong visual name consensus is allowed to correct a shifted MRZ name.
    # Names have no check digits, so require two independent visual sources.
    for field, a, b in (("surname", p1_surname, p3_surname), ("given_names", p1_given, p3_given)):
        na = _normalise_guess_name(a)
        nb = _normalise_guess_name(b)
        if na and nb and na == nb:
            guess[field] = na
            validation[field] = "validated — independent visual consensus"
            source_map[field] = ["part1-visual", "part3-visual"]

    # A structurally valid nationality recovered from the MRZ tail is safer
    # than a one-character parser artifact such as N. It is accepted only when
    # it is exactly three letters and came from the fixed MRZ nationality slot.
    if loose_best and loose_valid.get("nationality"):
        nat = _normalise_guess_alnum(loose_best.get("nationality"))
        if nat and re.fullmatch(r"[A-Z]{3}", nat):
            guess["nationality"] = nat
            validation["nationality"] = "validated — MRZ positional field"
            source_map["nationality"] = ["mrz-loose-field"]

    def consensus(field, normalizer):
        vals = []
        for raw, source in candidates.get(field, []):
            value = normalizer(raw)
            if value:
                vals.append((value, source))
        groups = {}
        for value, source in vals:
            groups.setdefault(value, set()).add(source)
        if not groups:
            return None, set()
        best_value, sources = max(groups.items(), key=lambda x: (len(x[1]), sum(1 for _ in x[1])))
        return best_value, sources

    # MRZ-structured values are accepted only when the parser validates them.
    for field in ("passport_number", "surname", "given_names", "nationality", "date_of_birth", "sex", "date_of_expiry"):
        normalizer = _normalise_guess_alnum if field in {"passport_number", "nationality", "sex"} else _normalise_guess_name
        if field in {"date_of_birth", "date_of_expiry"}:
            normalizer = _normalise_guess_date
        value, sources = consensus(field, normalizer)
        if value is None:
            continue
        if field in guess:
            continue

        structural = False
        if field == "passport_number":
            structural = bool(re.fullmatch(r"[A-Z0-9]{6,12}", value))
            if mrz_valid and mrz_data.get("passport_number"):
                structural = structural and bool(_mrz_field_valid(mrz_result, "document_number_valid"))
        elif field == "nationality":
            structural = bool(re.fullmatch(r"[A-Z]{3}", value))
        elif field == "sex":
            structural = value in {"M", "F", "X", "<"}
        elif field in {"surname", "given_names"}:
            structural = bool(re.fullmatch(r"[A-Z]+(?: [A-Z]+)*", value))
        else:
            try:
                dt = datetime.date.fromisoformat(value)
                structural = 1900 <= dt.year <= 2100
            except ValueError:
                structural = False

        # At least two independent sources are required for a visual-only
        # correction. A validated MRZ is allowed to stand alone.
        independent_sources = {s for s in sources if s not in {"current-stage2", "mrz-parser"}}
        mrz_support = mrz_valid and field in mrz_data and _normalise_guess_date(mrz_data.get(field)) == value if field in {"date_of_birth", "date_of_expiry"} else mrz_valid and _normalise_guess_alnum(mrz_data.get(field)) == value if field in {"passport_number", "nationality", "sex"} else mrz_valid and _normalise_guess_name(mrz_data.get(field)) == value
        accepted = structural and (len(independent_sources) >= 2 or mrz_support or bool(loose_valid.get(field)))
        if accepted:
            guess[field] = value
            validation[field] = "validated"
            source_map[field] = sorted(sources)

    # Issue date is never an MRZ field. Require a valid date from Part 2 visual
    # evidence and do not manufacture it from unrelated regions.
    issue = _normalise_guess_date(p2_issue or p1_issue)
    if issue:
        try:
            datetime.date.fromisoformat(issue)
            guess["date_of_issue"] = issue
            validation["date_of_issue"] = "validated-visual"
            source_map["date_of_issue"] = ["part2-visual"]
        except ValueError:
            pass

    # If the same field has an unresolved disagreement, explicitly preserve it
    # for the UI rather than choosing a random OCR answer.
    conflicts = {}
    for field, vals in candidates.items():
        normalizer = _normalise_guess_date if field in {"date_of_birth", "date_of_expiry"} else _normalise_guess_alnum if field in {"passport_number", "nationality", "sex"} else _normalise_guess_name
        distinct = sorted({normalizer(v) for v, _ in vals if normalizer(v)})
        if len(distinct) > 1:
            conflicts[field] = distinct

    return {
        "data": {field: guess.get(field) for field in FIELDS},
        "validation": validation,
        "sources": source_map,
        "conflicts": conflicts,
        "validated_field_count": len(validation),
        "method": "MRZ parser + independent visual OCR over Part 1/Part 2/Part 3 with conservative validation",
    }


# ------------------------------------------------------------
# Stage 2 MRZ-primary result builder
# ------------------------------------------------------------


def _clean_output(value):
    if isinstance(value, str):
        value = value.replace("<", "").replace(">", "").strip()
    return value if value not in ("", None) else None


def _field_primary(
    field: str,
    mrz_value: str | None,
    visual_value: str | None,
    mrz_valid: bool,
    visual_confidence: float,
    repaired: bool,
) -> tuple[object, str, str]:
    """MRZ-primary field merge used only by Stage 2."""
    if mrz_value:
        if field in {"surname", "given_names"} and visual_value:
            agrees = _visual_name_matches_mrz(visual_value, mrz_value)
        elif field == "passport_number" and visual_value:
            agrees = _norm_alnum(visual_value) == _norm_alnum(mrz_value)
        else:
            agrees = bool(
                visual_value
                and str(visual_value).upper().strip() == str(mrz_value).upper().strip()
            )

        if visual_value and agrees:
            return (
                mrz_value,
                "reliable",
                f"MRZ and visual OCR agree on the {field.replace('_', ' ')}.",
            )

        if visual_value and not agrees:
            return (
                mrz_value,
                "needs verification",
                (
                    f"MRZ reads {mrz_value}, while visual OCR reads {visual_value}. "
                    "MRZ remains the primary Stage 2 value; verify against the passport."
                ),
            )

        if repaired:
            return (
                mrz_value,
                "needs verification",
                (
                    f"MRZ supplied the {field.replace('_', ' ')}, but the OCR parse "
                    "required a repair. Verify the passport image."
                ),
            )

        if mrz_valid:
            return (
                mrz_value,
                "mrz validated",
                f"MRZ supplied the {field.replace('_', ' ')} and the relevant validation passed.",
            )

        return (
            mrz_value,
            "needs verification",
            f"MRZ supplied the {field.replace('_', ' ')}, but independent validation is incomplete.",
        )

    if visual_value and _visual_valid(field, visual_value):
        return (
            visual_value,
            "reliable" if visual_confidence >= 80 else "needs verification",
            f"The {field.replace('_', ' ')} was recovered from visual OCR because no MRZ value was available.",
        )

    return None, "not detected", f"Could not detect the {field.replace('_', ' ')}."



def _extract_part3_visual_fields(part3_visual_text: str) -> dict:
    """Extract an independent visual-OCR reading of Part 3.

    The normal visual OCR is intentionally *not* treated as authoritative.
    However, when it produces two near-TD3 rows, run those rows through the
    same positional parser used by the MRZ path. This lets the known TD3 field
    positions recover a dropped/extra OCR character before the comparison.
    """
    lines = []
    for raw in str(part3_visual_text or "").splitlines():
        line = re.sub(r"[^A-Z0-9<]", "", raw.upper())
        if line:
            lines.append(line)

    line1 = next((x for x in lines if x.startswith(("P<", "V<")) and len(x) >= 28), "")
    line2 = next((x for x in lines if x != line1 and len(x) >= 38 and sum(c.isdigit() for c in x) >= 5), "")

    out = {
        "line1": line1 or None,
        "line2": line2 or None,
        "surname": None,
        "given_names": None,
        "passport_number": None,
        "nationality": None,
        "date_of_birth_raw": None,
        "sex": None,
        "date_of_expiry_raw": None,
        "parser_recovered": False,
        "parser_validation": {},
    }

    # Let the positional TD3 parser repair a near-44-character visual row.
    if line1 and line2:
        try:
            visual_parsed = parse_mrz(f"{line1}\n{line2}")
        except Exception:
            visual_parsed = None
        if isinstance(visual_parsed, dict) and visual_parsed.get("success"):
            vd = visual_parsed.get("data", {})
            out.update({
                "line1": visual_parsed.get("line1") or line1,
                "line2": visual_parsed.get("line2") or line2,
                "surname": vd.get("surname") or None,
                "given_names": vd.get("given_names") or None,
                "passport_number": vd.get("passport_number") or None,
                "nationality": vd.get("nationality") or None,
                "date_of_birth_raw": vd.get("date_of_birth") or None,
                "sex": vd.get("sex") or None,
                "date_of_expiry_raw": vd.get("date_of_expiry") or None,
                "parser_recovered": True,
                "parser_validation": visual_parsed.get("validation", {}),
            })

            # Even a successful parser can have a one-character name alignment
            # error. The ordinary OCR text is allowed to challenge the spelling
            # without changing MRZ field identity.
            loose = _recover_td3_fields_from_noisy_text([part3_visual_text])
            loose_best = max(loose, key=lambda x: sum(bool(x.get(k)) for k in (
                "nationality", "date_of_birth_raw", "sex", "date_of_expiry_raw"
            )), default=None)
            if loose_best:
                if loose_best.get("nationality"):
                    out["nationality"] = loose_best.get("nationality")
                if loose_best.get("date_of_birth_raw") and loose_best.get("field_validation", {}).get("birth_date_valid"):
                    out["date_of_birth_raw"] = loose_best.get("date_of_birth_raw")
                if loose_best.get("sex") in {"M", "F"}:
                    out["sex"] = loose_best.get("sex")
                if loose_best.get("date_of_expiry_raw") and loose_best.get("field_validation", {}).get("expiry_date_valid"):
                    out["date_of_expiry_raw"] = loose_best.get("date_of_expiry_raw")

            raw_surname, raw_given = _extract_independent_part3_names(
                part3_visual_text, out.get("nationality")
            )
            if raw_surname:
                out["surname"] = raw_surname
            if raw_given:
                out["given_names"] = raw_given
            return out

    # Conservative fallback if the ordinary OCR only produced one usable row.
    if line1:
        name_area = line1[5:]
        parts = name_area.split("<<", 1)
        out["surname"] = parts[0].replace("<", " ").strip() or None
        if len(parts) > 1:
            out["given_names"] = parts[1].replace("<", " ").strip() or None

    if len(line2) == 44:
        out["passport_number"] = line2[0:9].replace("<", "").strip() or None
        out["nationality"] = line2[10:13].replace("<", "").strip() or None
        out["date_of_birth_raw"] = line2[13:19] or None
        out["sex"] = line2[20:21] or None
        out["date_of_expiry_raw"] = line2[21:27] or None

    loose = _recover_td3_fields_from_noisy_text([part3_visual_text])
    loose_best = max(loose, key=lambda x: sum(bool(x.get(k)) for k in (
        "nationality", "date_of_birth_raw", "sex", "date_of_expiry_raw"
    )), default=None)
    if loose_best:
        out["nationality"] = loose_best.get("nationality") or out.get("nationality")
        if loose_best.get("field_validation", {}).get("birth_date_valid"):
            out["date_of_birth_raw"] = loose_best.get("date_of_birth_raw")
        if loose_best.get("sex") in {"M", "F"}:
            out["sex"] = loose_best.get("sex")
        if loose_best.get("field_validation", {}).get("expiry_date_valid"):
            out["date_of_expiry_raw"] = loose_best.get("date_of_expiry_raw")

    raw_surname, raw_given = _extract_independent_part3_names(
        part3_visual_text, out.get("nationality")
    )
    if raw_surname:
        out["surname"] = raw_surname
    if raw_given:
        out["given_names"] = raw_given

    return out

def build_stage2_mrz_primary_result(
    mrz_result: dict,
    visual_text: str,
    visual_fields: dict,
    part3_visual_text: str = "",
) -> dict:
    """Merge Stage 2 evidence with validated MRZ values as the primary source."""
    d = mrz_result.get("data", {}) if isinstance(mrz_result, dict) else {}
    validation = mrz_result.get("validation", {}) if isinstance(mrz_result, dict) else {}
    repaired = set(mrz_result.get("repaired", []) or []) if isinstance(mrz_result, dict) else set()
    part3_visual = _extract_part3_visual_fields(part3_visual_text)

    # Field-wise recovery from the raw independent Part-3 OCR. This is used
    # only when the full MRZ parser has lost alignment; valid date/sex fields
    # are still accepted only when their own MRZ check digits pass.
    loose_part3 = _recover_td3_fields_from_noisy_text([part3_visual_text])
    loose_part3_best = max(
        loose_part3,
        key=lambda x: sum(bool(x.get(k)) for k in (
            "nationality", "date_of_birth_raw", "sex", "date_of_expiry_raw"
        )),
        default=None,
    )
    if loose_part3_best:
        if not part3_visual.get("nationality"):
            part3_visual["nationality"] = loose_part3_best.get("nationality")
        if not part3_visual.get("date_of_birth_raw"):
            part3_visual["date_of_birth_raw"] = loose_part3_best.get("date_of_birth_raw")
        if not part3_visual.get("sex"):
            part3_visual["sex"] = loose_part3_best.get("sex")
        if not part3_visual.get("date_of_expiry_raw"):
            part3_visual["date_of_expiry_raw"] = loose_part3_best.get("date_of_expiry_raw")

    loose_nat = loose_part3_best.get("nationality") if loose_part3_best else None
    loose_dob = mrz_date_to_iso(loose_part3_best.get("date_of_birth_raw", ""), "birth") if loose_part3_best else None
    loose_exp = mrz_date_to_iso(loose_part3_best.get("date_of_expiry_raw", ""), "expiry") if loose_part3_best else None
    loose_sex = loose_part3_best.get("sex") if loose_part3_best else None
    loose_partial_number = _normalise_guess_alnum(loose_part3_best.get("partial_passport_number")) if loose_part3_best else None

    raw_surname, raw_given = _extract_independent_part3_names(
        part3_visual_text,
        part3_visual.get("nationality"),
    )
    if raw_surname:
        part3_visual["surname"] = raw_surname
    if raw_given:
        part3_visual["given_names"] = raw_given

    # Independent visual evidence from Part 3 can now be structurally repaired
    # using TD3 positions before it is compared with the dedicated MRZ reader.
    # This is evidence only; it never bypasses MRZ validation by itself.
    part3_passport = part3_visual.get("passport_number")
    part3_surname = part3_visual.get("surname")
    part3_given = part3_visual.get("given_names")
    part3_dob = mrz_date_to_iso(part3_visual.get("date_of_birth_raw", ""), "birth")
    part3_exp = mrz_date_to_iso(part3_visual.get("date_of_expiry_raw", ""), "expiry")

    mrz_dob = mrz_date_to_iso(d.get("date_of_birth", ""), "birth")
    mrz_exp = mrz_date_to_iso(d.get("date_of_expiry", ""), "expiry")

    data = {field: None for field in FIELDS}
    status = {field: "not detected" for field in FIELDS}
    reasons = {field: "No value found." for field in FIELDS}
    warnings = list(mrz_result.get("warnings", []) or []) if isinstance(mrz_result, dict) else []

    def set_field(field, value, state, reason):
        data[field] = _clean_output(value)
        status[field] = state if data[field] is not None else "not detected"
        reasons[field] = reason if data[field] is not None else "No value found."

    # The existing validator's targeted field extraction remains useful as
    # secondary evidence.
    visual_number = _visual_value(visual_fields, "passport_number")
    if not visual_number:
        candidates = [
            token
            for token in re.findall(r"\b[A-Z0-9]{6,12}\b", visual_text.upper())
            if any(c.isdigit() for c in token) and any(c.isalpha() for c in token)
        ]
        if candidates:
            visual_number = candidates[0]

    # ---------------- Passport number ----------------
    mrz_number = _mrz_field(mrz_result, "passport_number")
    # Three-source consensus: Part 1 VIZ + independent Part 3 visual OCR can
    # correct a bad MRZ OCR character. The MRZ check digit still wins when the
    # visual sources do not independently agree.
    number_consensus = bool(
        visual_number and part3_passport
        and _norm_alnum(visual_number) == _norm_alnum(part3_passport)
    )
    loose_number_match = bool(
        visual_number and loose_partial_number and len(loose_partial_number) >= 5
        and (loose_partial_number in _norm_alnum(visual_number) or _norm_alnum(visual_number) in loose_partial_number)
    )
    if loose_number_match and (not mrz_number or _norm_alnum(mrz_number) != _norm_alnum(visual_number)):
        set_field(
            "passport_number", visual_number, "needs verification",
            f"Part 1 visual OCR reads {visual_number}, and the visible MRZ passport-number segment contains the independently matching {loose_partial_number} anchor. The full MRZ line was OCR-shifted, so verify the leading characters against the passport."
        )
    elif mrz_number and number_consensus and _norm_alnum(mrz_number) != _norm_alnum(visual_number):
        set_field(
            "passport_number", visual_number, "reliable",
            f"Part 1 visual OCR and independent Part 3 visual OCR agree on {visual_number}; the MRZ OCR differs, so the two independent visual readings were used to correct the MRZ OCR error."
        )
    else:
        value, state, reason = _field_primary(
            "passport_number",
            mrz_number,
            visual_number,
            _mrz_field_valid(mrz_result, "document_number_valid"),
            _visual_confidence(visual_fields, "passport_number") or (70.0 if visual_number else 0.0),
            "passport_number" in repaired,
        )
        set_field("passport_number", value, state, reason)

    # ---------------- Names ----------------
    for field in ("surname", "given_names"):
        mrz_value = _mrz_field(mrz_result, field)
        visual_value = _visual_value(visual_fields, field)
        part3_value = part3_surname if field == "surname" else part3_given

        # Names have no individual MRZ check digit. If the ordinary OCR of
        # Part 3 and the ordinary VIZ OCR from Part 1 independently agree,
        # that consensus can correct a one-character MRZ OCR mistake such as
        # SAMELS / SAMUELK -> SAMUEL.
        p3_and_viz_agree = bool(
            part3_value and visual_value
            and _visual_name_matches_mrz(part3_value, visual_value)
        )
        if mrz_value:
            if p3_and_viz_agree and not _visual_name_matches_mrz(mrz_value, part3_value):
                set_field(
                    field, visual_value, "reliable",
                    f"Part 3 visual OCR and Part 1 visual OCR agree on the {field.replace('_', ' ')}, while the MRZ OCR differs; the visual consensus was used to reduce an OCR spelling error."
                )
            elif visual_value and _visual_name_matches_mrz(visual_value, mrz_value):
                set_field(field, mrz_value, "reliable", f"MRZ and visual OCR agree on the {field.replace('_', ' ')}.")
            elif part3_value and _visual_name_matches_mrz(part3_value, mrz_value):
                set_field(field, mrz_value, "reliable", f"MRZ and independent Part 3 visual OCR agree on the {field.replace('_', ' ')}.")
            elif visual_value or part3_value:
                set_field(field, mrz_value, "needs verification", f"MRZ reads {mrz_value}, while independent visual OCR evidence differs. MRZ remains the structured source unless the two visual readings agree on a correction.")
            else:
                state = "needs verification" if field in repaired else "mrz validated"
                set_field(field, mrz_value, state, f"MRZ structure identifies the {field.replace('_', ' ')}; visual OCR did not provide confirmation.")
        elif visual_value or part3_value:
            value = visual_value or part3_value
            conf = _visual_confidence(visual_fields, field)
            set_field(field, value, "reliable" if p3_and_viz_agree or conf >= 80 else "needs verification", f"The {field.replace('_', ' ')} was recovered from independent visual OCR evidence.")

    # ---------------- Nationality ----------------
    visual_nat = _country_text_to_code(_visual_value(visual_fields, "nationality"))
    mrz_nat = _mrz_field(mrz_result, "nationality")
    if loose_nat and re.fullmatch(r"[A-Z]{3}", loose_nat) and (not mrz_nat or len(mrz_nat) != 3):
        value, state, reason = loose_nat, "needs verification", "The MRZ OCR was shifted, but the independent Part-3 OCR recovered the fixed three-character nationality slot as IND-compatible structured text."
    else:
        value, state, reason = _field_primary(
            "nationality",
            mrz_nat,
            visual_nat,
            bool(validation.get("nationality_format_valid")),
            _visual_confidence(visual_fields, "nationality"),
            False,
        )
    set_field("nationality", value, state, reason)

    # ---------------- Dates ----------------
    date_candidates = []
    try:
        for value in visual_fields.get("_meta", {}).get("date_candidates", []) or []:
            date_candidates.append(datetime.date.fromisoformat(str(value)))
    except (AttributeError, TypeError, ValueError):
        pass

    # infer_date_roles also re-parses date scan text and targeted date fields.
    date_roles = infer_date_roles(
        date_candidates,
        mrz_dob,
        mrz_exp,
        visual_fields=visual_fields,
    )
    visual_dob = date_roles.get("dob")
    visual_issue = date_roles.get("issue")
    visual_expiry = date_roles.get("expiry")

    # DOB: MRZ primary, but a field-wise MRZ recovery can survive a shifted
    # passport-number prefix. Its own check digit has already passed.
    if not mrz_dob and loose_dob:
        set_field("date_of_birth", loose_dob, "needs verification", "Full MRZ alignment failed, but the independent Part-3 OCR recovered the DOB from the fixed TD3 date slot and its check digit passed.")
    elif mrz_dob and visual_dob and part3_dob and visual_dob.isoformat() == part3_dob and visual_dob.isoformat() != mrz_dob:
        set_field("date_of_birth", visual_dob.isoformat(), "reliable", f"Part 2 visual OCR and independent Part 3 visual OCR agree on {visual_dob.isoformat()}; the MRZ OCR differs, so visual consensus corrected the MRZ OCR alignment.")
    elif mrz_dob:
        if isinstance(visual_dob, datetime.date):
            if visual_dob.isoformat() == mrz_dob:
                set_field("date_of_birth", mrz_dob, "reliable", "MRZ supplies the date of birth and the visual date set independently confirms it.")
            else:
                set_field("date_of_birth", mrz_dob, "needs verification", f"MRZ reads {mrz_dob}, while visual date inference reads {visual_dob.isoformat()}. MRZ remains primary.")
        else:
            set_field("date_of_birth", mrz_dob, "needs verification" if "date_of_birth" in repaired else "mrz validated", "MRZ supplies the date of birth; visual OCR did not provide confirming evidence.")
    elif isinstance(visual_dob, datetime.date):
        set_field("date_of_birth", visual_dob.isoformat(), "needs verification", "Date of birth was recovered from visual date-role inference without MRZ confirmation.")

    # Expiry: same three-source consensus rule as DOB.
    if not mrz_exp and loose_exp:
        set_field("date_of_expiry", loose_exp, "needs verification", "Full MRZ alignment failed, but the independent Part-3 OCR recovered the expiry date from the fixed TD3 date slot and its check digit passed.")
    elif mrz_exp and visual_expiry and part3_exp and visual_expiry.isoformat() == part3_exp and visual_expiry.isoformat() != mrz_exp:
        set_field("date_of_expiry", visual_expiry.isoformat(), "reliable", f"Part 2 visual OCR and independent Part 3 visual OCR agree on {visual_expiry.isoformat()}; the MRZ OCR differs, so visual consensus corrected the MRZ OCR alignment.")
    elif mrz_exp:
        if isinstance(visual_expiry, datetime.date):
            if visual_expiry.isoformat() == mrz_exp:
                set_field("date_of_expiry", mrz_exp, "reliable", "MRZ supplies the date of expiry and the visual date set independently confirms it.")
            else:
                set_field("date_of_expiry", mrz_exp, "needs verification", f"MRZ reads {mrz_exp}, while visual date inference reads {visual_expiry.isoformat()}. MRZ remains primary.")
        else:
            set_field("date_of_expiry", mrz_exp, "needs verification" if "date_of_expiry" in repaired else "mrz validated", "MRZ supplies the date of expiry; visual OCR did not provide confirming evidence.")
    elif isinstance(visual_expiry, datetime.date):
        set_field("date_of_expiry", visual_expiry.isoformat(), "needs verification", "Date of expiry was recovered from visual date-role inference without MRZ confirmation.")

    # Issue date is intentionally visual-only in TD3.
    if isinstance(visual_issue, datetime.date):
        set_field("date_of_issue", visual_issue.isoformat(), "reliable", f"Visual OCR inferred the date of issue using the existing date-role logic ({date_roles.get('method')}).")
    else:
        targeted_issue = _visual_value(visual_fields, "date_of_issue")
        if targeted_issue and _visual_valid("date_of_issue", targeted_issue):
            set_field("date_of_issue", targeted_issue, "needs verification", "Visual OCR produced a targeted issue date, but the date set was not unambiguous enough for the chronological role to be confirmed.")
        else:
            set_field("date_of_issue", None, "not detected", "Could not identify the date of issue from the Stage 2 visual date evidence.")
            warnings.append("Date of issue was not confidently detected by Stage 2; verify or enter it manually.")

    # ---------------- Sex ----------------
    mrz_sex = (_mrz_field(mrz_result, "sex") or "").upper()
    visual_sex = (_visual_value(visual_fields, "sex") or "").upper()
    if mrz_sex not in {"M", "F"} and loose_sex in {"M", "F"}:
        set_field("sex", loose_sex, "needs verification", "The full MRZ parse was shifted, but the independent Part-3 OCR recovered sex from the fixed TD3 slot.")
    elif mrz_sex in {"M", "F"}:
        if visual_sex in {"M", "F"} and visual_sex != mrz_sex:
            set_field("sex", mrz_sex, "needs verification", f"MRZ reads {mrz_sex}, while visual OCR reads {visual_sex}. MRZ remains primary.")
        elif visual_sex == mrz_sex:
            set_field("sex", mrz_sex, "reliable", "MRZ supplies the sex value and visual OCR independently confirms it.")
        else:
            set_field("sex", mrz_sex, "mrz validated", "MRZ supplies the sex value; visual OCR did not provide confirmation.")
    elif visual_sex in {"M", "F"}:
        set_field("sex", visual_sex, "needs verification", "Sex was recovered from visual OCR without MRZ confirmation.")

    # ---------------- Cross-check evidence ----------------
    cross = {}

    for field, mrz_value, visual_value in (
        ("passport_number", _mrz_field(mrz_result, "passport_number"), visual_number),
        ("surname", _mrz_field(mrz_result, "surname"), _visual_value(visual_fields, "surname")),
        ("given_names", _mrz_field(mrz_result, "given_names"), _visual_value(visual_fields, "given_names")),
        ("nationality", _mrz_field(mrz_result, "nationality"), visual_nat),
        ("date_of_birth", mrz_dob, visual_dob.isoformat() if isinstance(visual_dob, datetime.date) else None),
        ("date_of_issue", None, visual_issue.isoformat() if isinstance(visual_issue, datetime.date) else _visual_value(visual_fields, "date_of_issue")),
        ("date_of_expiry", mrz_exp, visual_expiry.isoformat() if isinstance(visual_expiry, datetime.date) else None),
        ("sex", mrz_sex or None, visual_sex or None),
    ):
        item = {
            "mrz_value": mrz_value,
            "visual_value": visual_value,
            "status": "not available",
        }
        if mrz_value and visual_value:
            if field in {"surname", "given_names"}:
                agrees = _visual_name_matches_mrz(visual_value, mrz_value)
            elif field == "passport_number":
                agrees = _norm_alnum(visual_value) == _norm_alnum(mrz_value)
            else:
                agrees = str(visual_value).upper() == str(mrz_value).upper()
            item["status"] = "agree" if agrees else "conflict"
        elif mrz_value:
            item["status"] = "mrz only"
        elif visual_value:
            item["status"] = "visual only"
        cross[field] = item

    # Independent Part 3 visual OCR evidence. This is deliberately kept
    # separate from the Part 1/Part 2 visual fields above.
    cross["part3_visual_ocr_fields"] = {}
    part3_field_values = {
        "passport_number": part3_visual.get("passport_number"),
        "surname": part3_visual.get("surname"),
        "given_names": part3_visual.get("given_names"),
        "nationality": part3_visual.get("nationality"),
        "date_of_birth": part3_dob,
        "sex": part3_visual.get("sex"),
        "date_of_expiry": part3_exp,
        # Issue date is not present in the MRZ.
        "date_of_issue": None,
    }
    mrz_field_values = {
        "passport_number": _mrz_field(mrz_result, "passport_number"),
        "surname": _mrz_field(mrz_result, "surname"),
        "given_names": _mrz_field(mrz_result, "given_names"),
        "nationality": _mrz_field(mrz_result, "nationality"),
        "date_of_birth": mrz_dob,
        "sex": mrz_sex or None,
        "date_of_expiry": mrz_exp,
        "date_of_issue": None,
    }
    for field in FIELDS:
        mv = mrz_field_values.get(field)
        pv = part3_field_values.get(field)
        if mv and pv:
            if field in {"surname", "given_names"}:
                agree = _visual_name_matches_mrz(pv, mv)
            elif field == "passport_number":
                agree = _norm_alnum(pv) == _norm_alnum(mv)
            else:
                agree = str(pv).upper() == str(mv).upper()
            comparison = "agree" if agree else "conflict"
        elif mv:
            comparison = "mrz only"
        elif pv:
            comparison = "independent visual only"
        else:
            comparison = "not available"
        cross["part3_visual_ocr_fields"][field] = {
            "mrz": mv,
            "part3_visual": pv,
            "comparison": comparison,
            "agree": comparison == "agree",
        }


    # Physical-location cross-checks. These are intentionally strict about
    # WHERE a value came from:
    #   - passport number -> Part 1
    #   - surname/given names -> Part 1
    #   - DOB/issue/expiry -> Part 2
    # The MRZ remains primary whenever it is structurally validated.
    cross["physical_split_logic"] = {
        "passport_number_source": "part1",
        "name_source": "part1",
        "date_source": "part2",
        "mrz_source": "part3",
        "rule": "MRZ first; compare MRZ fields against the VIZ field from its expected split part.",
    }

    cross["part1_checks"] = {
        "passport_number": {
            "mrz": _mrz_field(mrz_result, "passport_number"),
            "visual": visual_number,
            "agree": bool(
                _mrz_field(mrz_result, "passport_number")
                and visual_number
                and _norm_alnum(_mrz_field(mrz_result, "passport_number"))
                == _norm_alnum(visual_number)
            ),
        },
        "surname": {
            "mrz": _mrz_field(mrz_result, "surname"),
            "visual": _visual_value(visual_fields, "surname"),
            "agree": bool(
                _mrz_field(mrz_result, "surname")
                and _visual_value(visual_fields, "surname")
                and _visual_name_matches_mrz(
                    _visual_value(visual_fields, "surname"),
                    _mrz_field(mrz_result, "surname"),
                )
            ),
        },
        "given_names": {
            "mrz": _mrz_field(mrz_result, "given_names"),
            "visual": _visual_value(visual_fields, "given_names"),
            "agree": bool(
                _mrz_field(mrz_result, "given_names")
                and _visual_value(visual_fields, "given_names")
                and _visual_name_matches_mrz(
                    _visual_value(visual_fields, "given_names"),
                    _mrz_field(mrz_result, "given_names"),
                )
            ),
        },
    }

    cross["part2_checks"] = {
        "date_of_birth": {
            "mrz": mrz_dob,
            "visual": visual_dob.isoformat() if isinstance(visual_dob, datetime.date) else None,
            "agree": bool(
                mrz_dob
                and isinstance(visual_dob, datetime.date)
                and mrz_dob == visual_dob.isoformat()
            ),
        },
        "date_of_expiry": {
            "mrz": mrz_exp,
            "visual": visual_expiry.isoformat() if isinstance(visual_expiry, datetime.date) else None,
            "agree": bool(
                mrz_exp
                and isinstance(visual_expiry, datetime.date)
                and mrz_exp == visual_expiry.isoformat()
            ),
        },
        "date_of_issue": {
            "mrz": None,
            "visual": visual_issue.isoformat() if isinstance(visual_issue, datetime.date) else _visual_value(visual_fields, "date_of_issue"),
            "agree": None,
            "note": "Issue date is VIZ-only; MRZ does not contain it.",
        },
    }

    cross["date_roles"] = {
        "candidates": [
            d.isoformat() for d in date_roles.get("candidates", []) if isinstance(d, datetime.date)
        ],
        "dob": data.get("date_of_birth"),
        "issue": data.get("date_of_issue"),
        "expiry": data.get("date_of_expiry"),
        "method": date_roles.get("method"),
    }

    if part3_visual_text:
        mrz_reference = "".join(
            [
                str(mrz_result.get("line1", "")),
                str(mrz_result.get("line2", "")),
            ]
        )
        mrz_compact = re.sub(r"[^A-Z0-9<]", "", mrz_reference.upper())
        visual_compact = re.sub(r"[^A-Z0-9<]", "", part3_visual_text.upper())
        similarity = (
            SequenceMatcher(None, mrz_compact, visual_compact).ratio()
            if mrz_compact and visual_compact
            else 0.0
        )
        cross["part3_visual_ocr"] = {
            "text": part3_visual_text,
            "purpose": "secondary confirmation of the MRZ region",
            "similarity_to_selected_mrz": round(similarity, 3),
            "status": (
                "supporting-agreement"
                if similarity >= 0.80
                else "secondary-confirmation-not-established"
            ),
        }

    # Expiry warning remains useful and mirrors the Stage 1 behavior.
    if mrz_exp:
        try:
            if datetime.date.fromisoformat(mrz_exp) < datetime.date.today():
                warnings.append("The MRZ indicates that this passport is expired.")
        except ValueError:
            pass

    # Stage 2 succeeds when it has a structurally trustworthy MRZ or enough
    # visual recovery to be useful. The caller can retain Stage 1 on failure.
    usable = sum(data[field] not in (None, "") for field in FIELDS)
    success = _mrz_trustworthy(mrz_result) or usable >= 3

    return {
        "success": bool(success),
        "data": {field: data[field] for field in FIELDS},
        "field_status": {field: status[field] for field in FIELDS},
        "status_reasons": {field: reasons[field] for field in FIELDS},
        "warnings": list(dict.fromkeys(warnings)),
        "cross_check": cross,
        "stage2_source_priority": "MRZ-primary",
        "stage2_mrz_trustworthy": _mrz_trustworthy(mrz_result),
    }


# ------------------------------------------------------------
# Public Stage 2 runner
# ------------------------------------------------------------


def _fallback_stage2_result(
    stage1_result: dict | None,
    stage1_debug: dict | None,
    visual_text: str,
    visual_fields: dict,
) -> dict:
    """Recover from Part 1/Part 2 visual OCR even when MRZ is unavailable.

    The old fallback returned the untouched Stage 1 result when Stage 2 MRZ
    failed. That made a perfectly readable Stage 2 split appear as 'nothing
    detected'. Stage 2 visual evidence must still be used when its MRZ gate
    is not met.
    """
    stage1_mrz = None
    if isinstance(stage1_debug, dict):
        lines = stage1_debug.get("mrz_lines", []) or []
        if len(lines) >= 2 and lines[0] and lines[1]:
            try:
                stage1_mrz = parse_mrz(f"{lines[0]}\n{lines[1]}")
            except Exception:
                stage1_mrz = None

    # If Stage 1 has a usable MRZ, retain the mature validator merge while
    # allowing Stage 2 Part 1/Part 2 visual OCR to contribute.
    if stage1_mrz is not None:
        try:
            recovered = build_result(
                stage1_mrz,
                visual_text,
                visual_fields=visual_fields,
            )
            recovered.setdefault("warnings", []).append(
                "Stage 2 MRZ was not sufficiently trustworthy; Stage 2 visual recovery was used with the existing validator."
            )
            recovered["stage2_source_priority"] = "visual-recovery-fallback"
            recovered["stage2_mrz_trustworthy"] = False
            return recovered
        except Exception:
            pass

    # CRITICAL: do not discard Stage 2 visual evidence just because MRZ failed.
    # validator.build_result() supports mrz_result=None and will extract the
    # physical Part 1/Part 2 fields from visual_fields.
    try:
        recovered = build_result(
            None,
            visual_text,
            visual_fields=visual_fields,
        )
        recovered.setdefault("warnings", []).append(
            "Stage 2 MRZ was not sufficiently trustworthy; values shown here come from the physical Part 1/Part 2 visual OCR regions and should be verified."
        )
        recovered["stage2_source_priority"] = "stage2-visual-only-fallback"
        recovered["stage2_mrz_trustworthy"] = False
        return recovered
    except Exception:
        pass

    # Last resort: retain Stage 1 rather than returning an empty synthetic
    # object, but only after the Stage 2 visual merge has been attempted.
    result = dict(stage1_result) if isinstance(stage1_result, dict) else {
        "success": False,
        "data": {field: None for field in FIELDS},
        "field_status": {field: "not detected" for field in FIELDS},
        "status_reasons": {field: "No value found." for field in FIELDS},
        "warnings": [],
        "cross_check": {},
    }
    result.setdefault("warnings", []).append(
        "Stage 2 MRZ and visual recovery both failed; the Stage 1 result was retained."
    )
    result["stage2_source_priority"] = "stage1-retained"
    result["stage2_mrz_trustworthy"] = False
    return result

def _stage1_has_readable_field(
    stage1_result: dict | None,
    stage1_debug: dict | None,
) -> bool:
    """Return True when Stage 1 recovered at least one usable passport field.

    A field does not have to be marked reliable. A value marked
    "needs verification" is still readable and therefore still locks the
    Stage 1 orientation for Stage 2.
    """
    if isinstance(stage1_result, dict):
        data = stage1_result.get("data") or {}
        if isinstance(data, dict):
            for value in data.values():
                if value not in (None, ""):
                    return True

    if isinstance(stage1_debug, dict):
        visual_fields = stage1_debug.get("visual_fields") or {}
        if isinstance(visual_fields, dict):
            for key, value in visual_fields.items():
                if key == "_meta":
                    continue
                if isinstance(value, dict):
                    value = value.get("value") or value.get("text")
                if value not in (None, ""):
                    return True

        # A successfully parsed MRZ field is also readable Stage 1 evidence.
        mrz_lines = stage1_debug.get("mrz_lines") or []
        if any(isinstance(line, str) and line.strip() for line in mrz_lines):
            return True

    return False



def _final_manual_visual_needed(
    result: dict,
    mrz_stage2: dict,
) -> bool:
    """Run the expensive three-part safety pass only when it can add value."""
    if not isinstance(result, dict):
        return True

    data = result.get("data", {}) if isinstance(result.get("data"), dict) else {}
    statuses = result.get("field_status", {}) if isinstance(result.get("field_status"), dict) else {}

    if any(data.get(field) in (None, "") for field in FIELDS):
        return True

    for status in statuses.values():
        text_status = str(status or "").lower()
        if any(token in text_status for token in ("verify", "not detected", "uncertain", "conflict")):
            return True

    if isinstance(mrz_stage2, dict) and not mrz_stage2.get("trustworthy", False):
        return True

    return False


def _manual_visual_from_existing_evidence(
    visual: dict,
    mrz_stage2: dict,
) -> dict:
    """Build the safety-pass structure from OCR already collected."""
    parts = {}
    source_parts = visual.get("part_results", {}) if isinstance(visual, dict) else {}

    for name in ("part1", "part2"):
        value = source_parts.get(name, {}) if isinstance(source_parts, dict) else {}
        fields = {
            key: item
            for key, item in value.items()
            if key != "_meta"
        } if isinstance(value, dict) else {}
        meta = value.get("_meta", {}) if isinstance(value, dict) else {}
        parts[name] = {
            "fields": fields,
            "text": meta.get("full_text", "") if isinstance(meta, dict) else "",
            "raw": value,
        }

    p3_debug = mrz_stage2.get("debug", {}) if isinstance(mrz_stage2, dict) else {}
    p3_visual = p3_debug.get("part3_visual_ocr", {}) if isinstance(p3_debug, dict) else {}
    parts["part3"] = {
        "fields": {},
        "text": p3_visual.get("text", "") if isinstance(p3_visual, dict) else "",
        "raw": {},
        "independent_mrz_visual": p3_visual if isinstance(p3_visual, dict) else {},
    }

    return {
        "parts": parts,
        "text": "",
        "method": "final-independent-visual-ocr-reused-existing-evidence",
        "skipped_extra_ocr": True,
    }


def run_stage2(
    image: Image.Image,
    stage1_result: dict | None = None,
    stage1_debug: dict | None = None,
) -> dict:
    """Run Stage 2 with Stage 1 orientation locked whenever any field is readable."""
    original = ImageOps.exif_transpose(image).convert("RGB")

    stage1_orientation_locked = _stage1_has_readable_field(
        stage1_result,
        stage1_debug,
    )

    selected_orientation = None
    stage2_input = original
    orientation_source = "original-image"

    if isinstance(stage1_debug, dict):
        selected_orientation = stage1_debug.get("selected_orientation")
        # IMPORTANT: never feed Stage 2 the resized/skew-corrected Stage 1 debug
        # image. Stage 1 uses a 1600px OCR working copy. Stage 2 must start from
        # the original uploaded pixels and apply only the orientation selected by
        # Stage 1. Quarter-turns use lossless pixel rearrangement.
        if stage1_orientation_locked and selected_orientation in (0, 90, 180, 270):
            if selected_orientation == 0:
                stage2_input = original.copy()
            elif selected_orientation == 90:
                stage2_input = original.transpose(Image.Transpose.ROTATE_90)
            elif selected_orientation == 180:
                stage2_input = original.transpose(Image.Transpose.ROTATE_180)
            else:
                stage2_input = original.transpose(Image.Transpose.ROTATE_270)
            orientation_source = "stage1-selected-orientation-lossless"
        else:
            stage2_input = original
            orientation_source = "stage1-no-readable-field"

    debug = {
        "input_size": [stage2_input.width, stage2_input.height],
        "selected_orientation_from_stage1": selected_orientation,
        "stage1_readable_field_found": stage1_orientation_locked,
        "orientation_locked": stage1_orientation_locked,
        "orientation_source": orientation_source,
        "orientation_policy": (
            "LOCKED_TO_STAGE1"
            if stage1_orientation_locked
            else "ORIGINAL_IMAGE"
        ),
        # Keep the exact image used for border detection so the UI can show
        # the detected quadrilateral over the real input.
        "stage2_input": stage2_input,
    }

    try:
        border = detect_passport_border(stage2_input)
        debug["border_detection"] = border

        if border.get("success", False):
            passport_crop, crop_debug = _safe_perspective_crop(stage2_input, border)
            debug["crop"] = crop_debug
            normalized = normalize_passport(passport_crop)
        else:
            # Debug-first fallback: Stage 2 must still expose the exact image
            # it splits. No fake rectangle is invented.
            normalized = stage2_input.copy()
            debug["crop"] = {
                "method": "border-detection-failed-original-fallback",
                "warped": False,
                "orientation_preserved": True,
                "fallback": True,
                "reason": border.get("reason", "Unknown border detection failure"),
            }
            debug["border_fallback_used"] = True
        debug["normalized_size"] = [normalized.width, normalized.height]
        debug["normalized_passport"] = normalized

        parts = split_passport(normalized)
        debug["split"] = {name: [img.width, img.height] for name, img in parts.items()}
        debug["parts"] = parts

        # ============================================================
        # MRZ FIRST — this is the highest-priority evidence.
        # Do not run the Part 1 / Part 2 VIZ OCR until the MRZ parser,
        # validator and independent MRZ visual OCR have completed.
        # ============================================================
        mrz_stage2 = read_stage2_mrz(parts["part3"])
        debug["stage2_mrz"] = mrz_stage2["debug"]

        # Only after MRZ processing do we inspect the two VIZ regions.
        visual = _visual_pass_parts(parts)
        debug["stage2_visual_fields"] = {
            key: value
            for key, value in visual["visual_fields"].items()
            if key != "_meta"
        }
        debug["stage2_visual_text"] = visual["visual_text"]
        debug["stage2_part_visual_ocr"] = {
            name: {
                "text": value.get("_meta", {}).get("full_text", "")
                if isinstance(value.get("_meta", {}), dict)
                else "",
                "fields": {
                    key: item
                    for key, item in value.items()
                    if key != "_meta"
                },
            }
            for name, value in visual["part_results"].items()
        }

        # Build the normal Stage 2 result first. Only run the expensive
        # all-three-part safety pass when the current evidence needs it.
        if mrz_stage2.get("result"):
            final_result = build_stage2_mrz_primary_result(
                mrz_stage2["result"],
                visual["visual_text"],
                visual["visual_fields"],
                part3_visual_text=(
                    mrz_stage2.get("debug", {})
                    .get("part3_visual_ocr", {})
                    .get("text", "")
                ),
            )
        else:
            final_result = _fallback_stage2_result(
                stage1_result,
                stage1_debug,
                visual["visual_text"],
                visual["visual_fields"],
            )

        if _final_manual_visual_needed(final_result, mrz_stage2):
            manual_visual = _manual_visual_ocr_all_parts(parts)
        else:
            manual_visual = _manual_visual_from_existing_evidence(
                visual,
                mrz_stage2,
            )
        debug["stage2_final_manual_visual_ocr"] = manual_visual

        # Never overwrite the normal Stage 2 result with this safety-net
        # guess. It is displayed separately so the employee can review it.
        final_result["final_guess"] = _build_final_guess(
            final_result,
            mrz_stage2,
            visual,
            manual_visual,
        )

        final_result["stage2"] = {
            "success": bool(final_result.get("success")),
            "source_priority": final_result.get("stage2_source_priority"),
            "mrz_trustworthy": bool(final_result.get("stage2_mrz_trustworthy")),
            "border_confidence": border.get("confidence"),
            "border_method": border.get("method"),
            "border_fallback_used": bool(debug.get("border_fallback_used")),
            "perspective_warped": bool(crop_debug.get("warped")),
            "orientation_locked": stage1_orientation_locked,
            "orientation_source": orientation_source,
            "orientation_preserved": True,
        }
        final_result["debug"] = debug
        return final_result

    except Exception as exc:
        debug["error"] = f"{type(exc).__name__}: {exc}"
        fallback = dict(stage1_result) if isinstance(stage1_result, dict) else {
            "success": False,
            "data": {field: None for field in FIELDS},
            "field_status": {field: "not detected" for field in FIELDS},
            "status_reasons": {field: "No value found." for field in FIELDS},
            "warnings": [],
            "cross_check": {},
        }
        fallback.setdefault("warnings", []).append(
            f"Stage 2 analysis failed ({type(exc).__name__}); the Stage 1 result was retained."
        )
        fallback["stage2"] = {
            "success": False,
            "source_priority": "stage1-retained",
            "mrz_trustworthy": False,
            "orientation_locked": stage1_orientation_locked,
            "orientation_source": orientation_source,
        }
        fallback["debug"] = debug
        return fallback

