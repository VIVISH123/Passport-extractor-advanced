"""Visual OCR for printed passport identity fields.

Visual OCR is the primary source for the printed passport fields. The MRZ is
handled separately by the validator as a fallback and independent cross-check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytesseract
from PIL import Image, ImageEnhance, ImageOps
from pytesseract import Output


VISUAL_PSM11 = "--oem 3 --psm 11 -c user_defined_dpi=300"
VISUAL_NAME_CONFIG = (
    "--oem 3 --psm 13 -c user_defined_dpi=300 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)
VISUAL_NAME_FALLBACK = "--oem 3 --psm 7 -c user_defined_dpi=300"
VISUAL_DATE_CONFIG = (
    "--oem 3 --psm 13 -c user_defined_dpi=300 "
    "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-/."
)
VISUAL_DATE_FALLBACK = (
    "--oem 3 --psm 7 -c user_defined_dpi=300 "
    "-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-/."
)
VISUAL_DATE_ALT = "--oem 3 --psm 6 -c user_defined_dpi=300"
VISUAL_COUNTRY_CONFIG = "--oem 3 --psm 6 -c user_defined_dpi=300"
VISUAL_SEX_CONFIG = "--oem 3 --psm 11 -c user_defined_dpi=300 -c tessedit_char_whitelist=MF"


@dataclass
class Token:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2


def _prepare_identity(image: Image.Image, region_mode: bool = False) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")

    if region_mode:
        # Stage 2 supplies an already-cropped passport region. Do not apply the
        # full-page percentage crop again or the useful text area can be
        # cropped away a second time.
        crop = image
    else:
        w, h = image.size
        crop = image.crop(
            (
                int(w * 0.28),
                int(h * 0.10),
                int(w * 0.99),
                int(h * 0.86),
            )
        )

    gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    if gray.width != 2600:
        scale = 2600 / max(gray.width, 1)
        gray = gray.resize((2600, max(1, int(gray.height * scale))), Image.Resampling.LANCZOS)
    return gray


def _data_tokens(image: Image.Image) -> tuple[str, list[Token]]:
    data = pytesseract.image_to_data(image, config=VISUAL_PSM11, output_type=Output.DICT)
    tokens: list[Token] = []
    text_parts: list[str] = []
    for i, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        token = Token(
            text=text,
            left=int(data["left"][i]),
            top=int(data["top"][i]),
            width=int(data["width"][i]),
            height=int(data["height"][i]),
            conf=conf,
        )
        tokens.append(token)
        text_parts.append(text)
    return " ".join(text_parts), tokens


def _norm(text: str) -> str:
    return re.sub(r"[^A-Z]", "", (text or "").upper())


def _similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _find_label_anchor(tokens: list[Token], field: str) -> tuple[int, int, int, int] | None:
    """Return a bounding box around the best label for a field."""
    if not tokens:
        return None

    if field == "passport_number":
        aliases = (
            "PASSPORT", "PASSP0RT", "P4SSPORT", "PASSPORTNO",
            "PASSPORTNUMBER", "NUMBER", "NO",
        )
        # Prefer tokens containing PASSPORT. A nearby NO/NUMBER token is useful
        # but not mandatory because Tesseract often drops the punctuation.
        passport_tokens = [
            t for t in tokens
            if _similar(t.text, "PASSPORT") >= 0.58
            or _norm(t.text).startswith("PASSPORT")
        ]
        if not passport_tokens:
            return None
        base = max(passport_tokens, key=lambda t: t.conf)
        row = [
            t for t in tokens
            if abs(t.center_y - base.center_y) <= max(55, base.height * 1.2)
        ]
        return _union(tuple(row))

    if field == "surname":
        candidates = [(t,) for t in tokens if _similar(t.text, "SURNAME") >= 0.62 or _norm(t.text).startswith("SUR")]
        return _union(candidates[0]) if candidates else None

    if field == "given_names":
        candidates = [t for t in tokens if _norm(t.text) == "GIVEN"]
        best = None
        for given in candidates:
            nearby = [t for t in tokens if abs(t.center_y - given.center_y) < max(35, given.height * 0.8)]
            names = [t for t in nearby if _similar(t.text, "NAMES") >= 0.50]
            if names:
                box = _union((given, names[0]))
                score = given.conf + names[0].conf
                if best is None or score > best[0]:
                    best = (score, box)
        return best[1] if best else None

    if field == "nationality":
        candidates = [
            t for t in tokens
            if _similar(t.text, "NATIONALITY") >= 0.72
            or _similar(t.text, "NACIONALIDAD") >= 0.72
        ]
        if not candidates:
            return None
        base = max(candidates, key=lambda t: t.conf)
        same_row = [t for t in candidates if abs(t.center_y - base.center_y) <= max(50, base.height * 0.85)]
        return _union(tuple(same_row))

    if field == "sex":
        candidates = [t for t in tokens if _norm(t.text) in {"SEX", "SEXE", "SEXO"}]
        if not candidates:
            return None
        base = max(candidates, key=lambda t: t.conf)
        same_row = [t for t in candidates if abs(t.center_y - base.center_y) <= max(45, base.height * 0.8)]
        return _union(tuple(same_row))

    if field == "date_of_birth":
        return _find_date_label(tokens, ("DATE", "BIRTH"))

    if field == "date_of_issue":
        return _find_date_label(tokens, ("DATE", "ISSUE"))

    if field == "date_of_expiry":
        return _find_expiry_label(tokens)

    return None


def _find_date_label(tokens: list[Token], keywords: tuple[str, str]) -> tuple[int, int, int, int] | None:
    """Find a date label using multilingual passport wording and OCR-tolerant matching."""
    first_aliases = (
        "DATE", "DATEE", "DALE", "DATЕ",
    )
    second_aliases = {
        "BIRTH": ("BIRTH", "NAISSANCE", "NACIMIENTO", "BIRTHDATE"),
        "ISSUE": ("ISSUE", "DELIVRANCE", "DELIVERANCE", "EXPEDITION", "EXPEDICION", "EMISSION"),
    }
    requested = second_aliases.get(keywords[1], keywords[1:])

    best = None
    for anchor_token in tokens:
        anchor_score = max(_similar(anchor_token.text, a) for a in first_aliases)
        if anchor_score < 0.55:
            continue

        row = [
            t for t in tokens
            if abs(t.center_y - anchor_token.center_y) <= max(50, anchor_token.height * 1.1)
        ]
        matching = []
        for t in row:
            score = max(_similar(t.text, alias) for alias in requested)
            if score >= 0.45:
                matching.append((score, t))
        if not matching:
            continue

        second_score, second_best = max(matching, key=lambda x: x[0])
        label_left = min(anchor_token.left, second_best.left)
        label_right = max(anchor_token.right, second_best.right)
        relevant = [
            t for t in row
            if t.left >= label_left - 50 and t.left <= label_right + 1350
        ]
        box = _union(relevant or [anchor_token, second_best])
        # Strongly reward the requested field keyword; this separates Birth / Issue rows.
        score = second_score * 100.0 + anchor_score * 25.0 + min(anchor_token.conf, 100.0) / 10.0
        if best is None or score > best[0]:
            best = (score, box)
    return best[1] if best else None

def _find_expiry_label(tokens: list[Token]) -> tuple[int, int, int, int] | None:
    aliases = (
        "EXPIRATION", "EXPIRY", "CADUCIDAD", "EXPIRACION",
        "EXPIRATIONDATE", "EXPIRATION",
    )
    choices = []
    for t in tokens:
        score = max(_similar(_norm(t.text), alias) for alias in aliases)
        if score >= 0.68:
            choices.append((score, t))
    if not choices:
        return None
    _, best = max(choices, key=lambda x: x[0] * 100 + min(x[1].conf, 100.0) / 10.0)
    return _union((best,))

def _union(tokens: tuple[Token, ...] | list[Token]) -> tuple[int, int, int, int]:
    left = min(t.left for t in tokens)
    top = min(t.top for t in tokens)
    right = max(t.right for t in tokens)
    bottom = max(t.bottom for t in tokens)
    return left, top, right, bottom


def _crop_from_anchor(image: Image.Image, field: str, anchor: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = anchor
    w, h = image.size

    if field == "surname":
        box = (max(0, left - 70), min(h, bottom), min(w, 1500), min(h, bottom + 190))
    elif field == "given_names":
        box = (max(0, left - 70), max(0, bottom - 45), min(w, 1500), min(h, bottom + 190))
    elif field == "nationality":
        box = (max(0, left - 110), max(0, top - 20), min(w, 1700), min(h, bottom + 190))
    elif field == "sex":
        box = (max(0, left - 100), max(0, top + 35), min(w, right + 100), min(h, bottom + 230))
    elif field in {"date_of_birth", "date_of_issue"}:
        box = (max(0, left - 80), min(h, bottom - 8), min(w, 1700), min(h, bottom + 235))
    elif field == "date_of_expiry":
        box = (max(0, left - 80), min(h, bottom - 8), min(w, 1100), min(h, bottom + 220))
    else:
        box = (max(0, left - 50), min(h, bottom), min(w, 1600), min(h, bottom + 200))

    if box[2] <= box[0] or box[3] <= box[1]:
        return image.crop((0, 0, 1, 1))
    return image.crop(box)


def _band_tokens(tokens: list[Token], anchor: tuple[int, int, int, int], field: str) -> list[Token]:
    left, top, right, bottom = anchor
    center = (top + bottom) / 2.0
    if field == "nationality":
        lo, hi = center - 25, center + 155
        xmax = 1750
    elif field == "sex":
        lo, hi = center - 110, center + 150
        xmax = 2600
    else:
        lo, hi = center + 35, center + 165
        xmax = 1750

    blocked = {
        "SURNAME", "SUMAME", "GIVEN", "NAMES", "NATIONALITY", "NATIONETY",
        "NATIONALTE", "NACIONALIDAD", "DATE", "DALE", "DATEOF", "BIRTH",
        "ISSUE", "EXPIRATION", "EXPIRY", "CADUCIDAD", "SEX", "SEXE", "SEXO",
        "PLACE", "AUTHORITY", "AUTORITE", "AUTORIDAD", "ENDORSEMENTS",
        "MENTIONS", "SPECIALES", "ANOTACIONES",
    }
    out = []
    for t in tokens:
        text_norm = _norm(t.text)
        if (not text_norm and field not in {"date_of_birth", "date_of_issue", "date_of_expiry"}) or t.right < 0 or t.left > xmax:
            continue
        if not (lo <= t.center_y <= hi):
            continue
        if text_norm in blocked:
            continue
        if field in {"surname", "given_names"} and not re.fullmatch(r"[A-Za-z]+", t.text):
            continue
        if field in {"surname", "given_names"} and len(t.text) <= 2 and t.conf < 20:
            continue
        if field == "nationality" and not re.fullmatch(r"[A-Za-z]+", t.text):
            continue
        out.append(t)
    return sorted(out, key=lambda t: (t.center_y, t.left))


def _best_token_row(tokens: list[Token]) -> list[Token]:
    if not tokens:
        return []
    rows: list[list[Token]] = []
    for token in sorted(tokens, key=lambda t: t.center_y):
        placed = False
        for row in rows:
            avg_y = sum(x.center_y for x in row) / len(row)
            if abs(token.center_y - avg_y) <= max(65, token.height * 0.9):
                row.append(token)
                placed = True
                break
        if not placed:
            rows.append([token])
    return max(
        rows,
        key=lambda row: (sum(max(0.0, min(100.0, t.conf)) for t in row), sum(t.width for t in row)),
    )


def _token_value(tokens: list[Token], field: str) -> str | None:
    if field == "nationality":
        # The country name can span several OCR rows because of perspective and
        # font size. Keep alphabetic words from the target band and discard
        # multilingual label fragments.
        label_noise = {
            "NATIONALETY", "NATIONALITY", "NATIONALTE", "NATIONALT",
            "NACIONALIDAD", "NATIONALITE", "NATIONALIT",
        }
        words = [
            t.text for t in sorted(tokens, key=lambda t: (t.left, t.center_y))
            if _norm(t.text) not in label_noise
        ]
        # Prefer the prominent country-name sequence rather than tiny OCR noise.
        words = [w for w in words if len(_norm(w)) >= 2]
        return _country_value(" ".join(words)) if words else None

    row = _best_token_row(tokens)
    if not row:
        return None
    row = sorted(row, key=lambda t: t.left)
    if field == "sex":
        for t in row:
            hit = _sex_value(t.text)
            if hit:
                return hit
        return None
    text = " ".join(t.text for t in row)
    if field in {"date_of_birth", "date_of_issue", "date_of_expiry"}:
        return _parse_date(text)
    if field in {"surname", "given_names"}:
        return _clean_name(text)
    if field == "nationality":
        return _country_value(text)
    return None


def _ocr_text(image: Image.Image, config: str) -> str:
    return pytesseract.image_to_string(image, config=config).strip()


def _parse_date(text: str):
    """Parse the most common printed passport date formats."""
    import datetime

    clean = (text or "").upper().replace("—", "-").replace("–", "-").replace("|", " ")
    clean = re.sub(r"[^A-Z0-9./\- ]", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    month_aliases = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        # common OCR confusions
        "AUGG": 8, "AUO": 8, "0CT": 10, "DЕC": 12,
    }

    def make(day, mon, year):
        try:
            return datetime.date(int(year), int(mon), int(day)).isoformat()
        except (ValueError, TypeError):
            return None

    # Day Month Year, including 07-Aug-1999 and 07 Aug 1999.
    named = re.search(r"\b(\d{1,2})\s*[-./ ]*([A-Z0-9]{3,9})\s*[-./ ]*(\d{4})\b", clean)
    if named:
        day, mon, year = named.group(1), named.group(2)[:3], named.group(3)
        if mon in month_aliases:
            value = make(day, month_aliases[mon], year)
            if value:
                return value

    # Month Day Year, occasionally used by documents / OCR layouts.
    named_us = re.search(r"\b([A-Z0-9]{3,9})\s*(\d{1,2})\s*[-./ ]*?(\d{4})\b", clean)
    if named_us:
        mon, day, year = named_us.group(1)[:3], named_us.group(2), named_us.group(3)
        if mon in month_aliases:
            value = make(day, month_aliases[mon], year)
            if value:
                return value

    numeric = re.search(r"\b(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})\b", clean)
    if numeric:
        day, month, year = map(int, numeric.groups())
        value = make(day, month, year)
        if value:
            return value

    compact = re.search(r"\b(\d{1,2})([A-Z0-9]{3})(\d{4})\b", clean)
    if compact:
        day, mon, year = compact.group(1), compact.group(2), compact.group(3)
        if mon in month_aliases:
            value = make(day, month_aliases[mon], year)
            if value:
                return value
    return None


def _date_candidates(text: str) -> list[str]:
    """
    Return every structured date candidate found in an OCR string.

    The older implementation parsed only the first date it could find in a
    whole OCR block or on each line. That is brittle when one OCR pass contains
    several passport dates. Here we explicitly locate all common date shapes and
    parse each match independently.
    """
    values: list[str] = []
    seen: set[str] = set()

    raw = text or ""

    patterns = (
        # Day Month Year: 24 OCT 2022 / 24-OCT-2022 / 24 OCTOBER 2022
        r"\b\d{1,2}\s*[-./ ]*"
        r"[A-Z0-9]{3,9}\s*[-./ ]*\d{4}\b",

        # Month Day Year: OCT 24 2022
        r"\b[A-Z0-9]{3,9}\s*\d{1,2}\s*[-./ ]*\d{4}\b",

        # Numeric: 24/10/2022, 24.10.2022, 24-10-2022
        r"\b\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*\d{4}\b",

        # Compact: 24OCT2022
        r"\b\d{1,2}[A-Z0-9]{3}\d{4}\b",
    )

    for pattern in patterns:
        for match in re.findall(pattern, raw.upper()):
            value = _parse_date(match)
            if value and value not in seen:
                seen.add(value)
                values.append(value)

    # Keep the old line-by-line fallback because some OCR engines insert unusual
    # separators that are easier to normalize when the line is considered whole.
    for line in raw.splitlines():
        value = _parse_date(line)
        if value and value not in seen:
            seen.add(value)
            values.append(value)

    return values


def _clean_name(text: str) -> str | None:
    value = re.sub(r"[^A-Za-z' -]", " ", text or "").upper()
    value = re.sub(r"\s+", " ", value).strip(" -")
    blocked = {
        "GIVEN", "NAMES", "SURNAME", "NATIONALITY", "DATE", "BIRTH", "ISSUE",
        "EXPIRATION", "EXPIRY", "SEX", "SEXE", "SEXO", "PLACE", "AUTHORITY",
    }
    words = [w for w in value.split() if w not in blocked]
    if not words:
        return None
    return " ".join(words)


def _country_value(text: str) -> str | None:
    value = re.sub(r"\s+", " ", (text or "").upper()).strip()
    value = value.replace("‘", "").replace("'", "")
    # Keep the printed phrase; validator maps it against the MRZ country code.
    if not value:
        return None
    return value


def _sex_value(text: str) -> str | None:
    hits = re.findall(r"\b([MF])\b", (text or "").upper())
    return hits[0] if hits else None




def _prepare_date_scan(image: Image.Image) -> Image.Image:
    """Prepare a broad passport-page crop specifically for date detection."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    w, h = image.size

    # Keep almost all of the biodata page, while reducing the MRZ area.
    crop = image.crop(
        (int(w * 0.04), int(h * 0.07), int(w * 0.99), int(h * 0.90))
    )

    gray = ImageOps.grayscale(crop)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.35)

    if gray.width != 3000:
        scale = 3000 / max(gray.width, 1)
        gray = gray.resize(
            (3000, max(1, int(gray.height * scale))),
            Image.Resampling.LANCZOS,
        )

    return gray


def _scan_date_tiles(image: Image.Image) -> tuple[set[str], dict[str, str]]:
    """Fallback date scan using overlapping page tiles.

    This is deliberately layout-agnostic. When the full-page OCR misses a
    printed date, smaller crops give Tesseract much larger characters to read.
    """
    page = ImageOps.exif_transpose(image).convert("RGB")
    w, h = page.size

    found: set[str] = set()
    raw_texts: dict[str, str] = {}

    # 3 x 3 overlapping tiles. The overlap is intentionally generous because
    # passport date fields can sit close to a tile boundary.
    x_edges = [0, int(w * 0.34), int(w * 0.67), w]
    y_edges = [0, int(h * 0.34), int(h * 0.67), h]

    for row in range(3):
        for col in range(3):
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]

            pad_x = int(w * 0.09)
            pad_y = int(h * 0.12)
            box = (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(w, x1 + pad_x),
                min(h, y1 + pad_y),
            )

            crop = page.crop(box)
            gray = ImageOps.autocontrast(ImageOps.grayscale(crop))

            scale = 3000 / max(gray.width, 1)
            gray = gray.resize(
                (3000, max(1, int(gray.height * scale))),
                Image.Resampling.LANCZOS,
            )

            # Generic OCR works better here than the narrow date whitelist on
            # tiny passport crops; _date_candidates() filters the result later.
            raw = _ocr_text(
                gray,
                "--oem 3 --psm 6 -c user_defined_dpi=300",
            )
            name = f"date-tile-r{row}c{col}"
            raw_texts[name] = raw

            for value in _date_candidates(raw):
                found.add(value)

    # A few deliberately narrow crops catch compact fields that can disappear
    # inside a larger tile, especially the expiry-date corner.
    focused_boxes = {
        "date-focus-right-middle": (
            int(w * 0.72),
            int(h * 0.35),
            w,
            int(h * 0.92),
        ),
        "date-focus-right-lower": (
            int(w * 0.65),
            int(h * 0.50),
            w,
            h,
        ),
        "date-focus-center-lower": (
            int(w * 0.30),
            int(h * 0.42),
            int(w * 0.85),
            h,
        ),
    }

    for name, box in focused_boxes.items():
        crop = page.crop(box)
        gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
        scale = 3200 / max(gray.width, 1)
        gray = gray.resize(
            (3200, max(1, int(gray.height * scale))),
            Image.Resampling.LANCZOS,
        )

        raw = _ocr_text(
            gray,
            "--oem 3 --psm 6 -c user_defined_dpi=300",
        )
        raw_texts[name] = raw
        for value in _date_candidates(raw):
            found.add(value)

    return found, raw_texts


def _scan_all_visual_dates(
    image: Image.Image,
) -> tuple[list[str], dict[str, str]]:
    """
    Scan the printed passport page for all date-shaped OCR candidates.

    The normal full-page scan runs first. If it finds fewer than three dates,
    an overlapping tile scan is used to recover small corner/edge date fields.
    """
    scan = _prepare_date_scan(image)
    sharpened = ImageEnhance.Sharpness(scan).enhance(1.6)

    attempts = [
        ("date-psm11", scan, VISUAL_DATE_CONFIG),
        ("date-psm7", scan, VISUAL_DATE_FALLBACK),
        ("date-psm6", scan, VISUAL_DATE_ALT),
        ("date-psm11-sharp", sharpened, VISUAL_DATE_CONFIG),
    ]

    found: set[str] = set()
    raw_texts: dict[str, str] = {}

    for name, img, config in attempts:
        raw = _ocr_text(img, config)
        raw_texts[name] = raw
        for value in _date_candidates(raw):
            found.add(value)

    if len(found) < 3:
        tile_found, tile_texts = _scan_date_tiles(image)
        found.update(tile_found)
        raw_texts.update(tile_texts)

    return sorted(found), raw_texts


def _sex_from_roi(roi: Image.Image) -> tuple[str | None, float]:
    data = pytesseract.image_to_data(
        roi,
        config="--oem 3 --psm 11 -c user_defined_dpi=300",
        output_type=Output.DICT,
    )
    hits = []
    for i, raw in enumerate(data.get("text", [])):
        text = (raw or "").strip().upper()
        if text not in {"M", "F"}:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if conf >= 20:
            hits.append((text, conf, int(data["top"][i])))
    if not hits:
        return None, 0.0
    # The sex value is normally the first isolated M/F below the label, not a
    # character buried inside another printed word.
    hits.sort(key=lambda x: (-x[1], x[2]))
    return hits[0][0], hits[0][1]


def _direct_passport_number_scan(image: Image.Image) -> tuple[str | None, str, float, str]:
    """Read the printed document number directly when its label is missed."""
    page = ImageOps.exif_transpose(image).convert("RGB")
    w, h = page.size
    # Upper-right Part-1 number zone. This is intentionally separate from MRZ.
    crop = page.crop((int(w * 0.70), int(h * 0.22), w, int(h * 0.50)))
    gray = ImageOps.autocontrast(ImageOps.grayscale(crop))
    candidates = []
    confusion = str.maketrans({"O":"0", "Q":"0", "D":"0", "I":"1", "L":"1", "Z":"2", "S":"5", "G":"6", "B":"8"})

    for contrast in (1.3, 1.7, 2.0):
        work = ImageEnhance.Contrast(gray).enhance(contrast)
        work = work.resize((max(1, work.width * 8), max(1, work.height * 8)), Image.Resampling.LANCZOS)
        for psm in (6, 11, 7):
            raw = _ocr_text(work, f"--oem 3 --psm {psm} -c user_defined_dpi=300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
            # Normal 8-char candidate and one-character insertion/deletion tolerance.
            chunks = [clean[i:i+8] for i in range(max(0, len(clean)-7))]
            if len(clean) == 9:
                chunks += [clean[:i] + clean[i+1:] for i in range(9)]
            for chunk in chunks:
                if len(chunk) != 8 or not chunk[0].isalpha():
                    continue
                value = chunk[0] + chunk[1:].translate(confusion)
                if re.fullmatch(r"[A-Z][0-9]{7}", value):
                    score = 0.0
                    # Favor candidates that look like a clean passport number and
                    # occur near the upper-right field. Do not make this the final
                    # truth; MRZ validation still outranks it.
                    score += 20.0
                    if psm == 6: score += 4.0
                    if psm == 11: score += 6.0
                    if contrast >= 1.7: score += 2.0
                    candidates.append((value, raw, score))

    if not candidates:
        return None, "", 0.0, "direct-passport-number-no-candidate"

    # Prefer the candidate with the strongest repeated OCR support. Break ties by
    # score, while avoiding a single arbitrary substring from a longer word.
    from collections import Counter
    counts = Counter(v for v, _, _ in candidates)
    best_count = max(counts.values())
    pool = [(v, r, sc) for v, r, sc in candidates if counts[v] == best_count]
    value, raw, score = max(pool, key=lambda x: x[2])
    return value, raw, min(78.0, 50.0 + best_count * 5.0 + score / 5.0), f"direct-part1-passport-number-ocr:consensus={best_count}"


def extract_visual_fields(image: Image.Image, region_mode: bool = False) -> dict:
    """Extract field-specific candidates from the printed passport area.

    Stage 1 keeps the original full-page crop behavior. Stage 2 can pass
    ``region_mode=True`` for an already-cropped passport section.
    """
    identity = _prepare_identity(image, region_mode=region_mode)
    full_text, tokens = _data_tokens(identity)

    fields: dict[str, dict] = {}
    configs = {
        "passport_number": (VISUAL_DATE_CONFIG, VISUAL_DATE_FALLBACK),
        "surname": (VISUAL_NAME_CONFIG, VISUAL_NAME_FALLBACK),
        "given_names": (VISUAL_NAME_CONFIG, VISUAL_NAME_FALLBACK),
        "nationality": (VISUAL_COUNTRY_CONFIG, VISUAL_COUNTRY_CONFIG),
        "date_of_birth": (VISUAL_DATE_CONFIG, VISUAL_DATE_FALLBACK),
        "date_of_issue": (VISUAL_DATE_CONFIG, VISUAL_DATE_FALLBACK),
        "date_of_expiry": (VISUAL_DATE_CONFIG, VISUAL_DATE_FALLBACK),
        "sex": (VISUAL_SEX_CONFIG, VISUAL_SEX_CONFIG),
    }

    for field, (config, fallback_config) in configs.items():
        anchor = _find_label_anchor(tokens, field)
        if anchor is None:
            if field == "passport_number":
                direct_value, direct_raw, direct_conf, direct_method = _direct_passport_number_scan(image)
                fields[field] = {
                    "value": direct_value,
                    "raw": direct_raw,
                    "confidence": round(direct_conf, 1),
                    "method": direct_method,
                }
            else:
                fields[field] = {"value": None, "raw": "", "confidence": 0.0, "method": "label-not-found"}
            continue

        candidates = _band_tokens(tokens, anchor, field)
        token_value = _token_value(candidates, field)

        # Sex is particularly small, so use a focused OCR fallback even when token
        # OCR did not return a single M/F token. Other fields only OCR a small ROI
        # when the data pass cannot produce a structured value.
        raw = ""
        value = token_value
        method = "label-guided-token-ocr" if token_value else "label-guided-targeted-ocr"

        # Date fields get a second, focused OCR pass even when token OCR already found a
        # candidate. This prevents a bad token grouping from becoming the only evidence.
        if field == "passport_number":
            roi = _crop_from_anchor(identity, field, anchor)
            raw = _ocr_text(
                roi,
                "--oem 3 --psm 7 -c user_defined_dpi=300 "
                "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
            cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
            # Indian passport numbers are normally one letter followed by seven
            # digits, but keep a broad 6-12 character candidate for other passports.
            m = re.search(r"[A-Z][A-Z0-9]{5,11}", cleaned)
            value = m.group(0) if m else None
            method = "label-guided-passport-number-ocr"
        elif field in {"date_of_birth", "date_of_issue", "date_of_expiry"}:
            roi = _crop_from_anchor(identity, field, anchor)
            raw = _ocr_text(roi, config)
            candidates = _date_candidates(raw)
            if not candidates:
                raw2 = _ocr_text(roi, fallback_config)
                candidates = _date_candidates(raw2)
                if raw2 and raw2 != raw:
                    raw = raw + " || " + raw2
            roi_value = candidates[0] if candidates else None
            if token_value and roi_value and token_value == roi_value:
                value = token_value
                method = "label-guided-token+roi-agree"
            elif token_value:
                value = token_value
                method = "label-guided-token-ocr"
            else:
                value = roi_value
                method = "label-guided-targeted-ocr"
        elif field == "sex" or value is None:
            roi = _crop_from_anchor(identity, field, anchor)
            if field == "sex":
                value, sex_conf = _sex_from_roi(roi)
                raw = value or ""
            else:
                raw = _ocr_text(roi, config)
                if field in {"surname", "given_names"}:
                    value = _clean_name(raw)
                    if value is None:
                        raw2 = _ocr_text(roi, fallback_config)
                        value = _clean_name(raw2)
                        if raw2 and raw2 != raw:
                            raw = raw + " || " + raw2
                elif field == "nationality":
                    value = _country_value(raw)

        fields[field] = {
            "value": value,
            "raw": raw or (" ".join(t.text for t in _best_token_row(candidates))),
            "confidence": round(
                sum(max(0.0, min(100.0, t.conf)) for t in _best_token_row(candidates))
                / max(len(_best_token_row(candidates)), 1),
                1,
            ),
            "method": method,
        }

    # Broad date-only OCR is separate from label-guided OCR.
    # The validator uses these candidates for:
    #     earliest = DOB
    #     middle   = Issue
    #     latest   = Expiry
    date_candidates, date_scan_texts = _scan_all_visual_dates(image)

    # Helpful debug metadata, without storing image bytes.
    fields["_meta"] = {
        "full_text": full_text,
        "identity_size": list(identity.size),
        "date_candidates": date_candidates,
        "date_scan_texts": date_scan_texts,
    }
    return fields