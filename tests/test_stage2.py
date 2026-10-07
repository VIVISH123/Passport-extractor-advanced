from datetime import date

import numpy as np
from PIL import Image, ImageDraw

from mrz_parser import check_digit
from stage2 import detect_passport_border, split_passport, build_stage2_mrz_primary_result


def _fake_mrz():
    doc = "A1234567"
    birth = "800101"
    exp = "300101"
    optional = "12345678901234"
    composite_source = (
        doc
        + "0"
        + "ZAF0000000"  # filler-like source not used directly in this fake fixture
    )
    line1 = "P<ZAFSPECTRENK<<JANE<K<<<<<<<<<<<<<<<<<<<<<"
    line1 = line1[:44].ljust(44, "<")
    field = (
        doc
        + check_digit(doc)
        + "ZAF"
        + birth
        + check_digit(birth)
        + "F"
        + exp
        + check_digit(exp)
        + optional
    )
    field = field[:42]
    composite = (
        doc + check_digit(doc) + "ZAF" + birth + check_digit(birth) + "F"
        + exp + check_digit(exp) + optional[:14]
    )
    line2 = field + check_digit(composite)
    line2 = line2[:44].ljust(44, "<")
    return {
        "success": True,
        "line1": line1,
        "line2": line2,
        "data": {
            "passport_number": doc,
            "surname": "SPECTRENK",
            "given_names": "JANE K",
            "nationality": "ZAF",
            "date_of_birth": birth,
            "sex": "F",
            "date_of_expiry": exp,
        },
        "validation": {
            "document_number_valid": True,
            "birth_date_valid": True,
            "expiry_date_valid": True,
            "composite_valid": True,
            "country_format_valid": True,
            "nationality_format_valid": True,
            "overall_valid": True,
        },
        "repaired": [],
        "warnings": [],
    }


def test_border_detector_finds_tiny_passport_on_white_background():
    canvas = Image.new("RGB", (2400, 1800), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((900, 650, 1770, 1260), fill=(238, 238, 238), outline=(30, 30, 30), width=12)
    result = detect_passport_border(canvas)
    assert result["success"]
    assert result["quad"]
    assert 1.1 <= result["metrics"]["aspect"] <= 2.3


def test_split_comes_from_one_normalized_image():
    image = Image.new("RGB", (1000, 700), "white")
    parts = split_passport(image)
    assert set(parts) == {"part1", "part2", "part3"}
    assert parts["part1"].width == 1000
    assert parts["part2"].width == 1000
    assert parts["part3"].width == 1000
    assert parts["part1"].height + parts["part2"].height + parts["part3"].height == 700


def test_stage2_mrz_primary_keeps_mrz_on_visual_conflict():
    mrz = _fake_mrz()
    visual_fields = {
        "passport_number": {"value": "A1234568", "confidence": 95, "raw": "A1234568"},
        "surname": {"value": "SPECTRENK", "confidence": 95, "raw": "SPECTRENK"},
        "given_names": {"value": "JANE K", "confidence": 95, "raw": "JANE K"},
        "nationality": {"value": "ZAF", "confidence": 95, "raw": "ZAF"},
        "sex": {"value": "F", "confidence": 95, "raw": "F"},
        "_meta": {
            "date_candidates": ["1980-01-01", "2009-03-30", "2030-01-01"],
            "date_scan_texts": {},
            "full_text": "SPECTRENK JANE K ZAF 01 JAN 1980 30 MAR 2009 01 JAN 2030",
        },
    }
    result = build_stage2_mrz_primary_result(
        mrz,
        visual_fields["_meta"]["full_text"],
        visual_fields,
    )
    assert result["data"]["passport_number"] == "A1234567"
    assert result["field_status"]["passport_number"] == "needs verification"
    assert result["stage2_source_priority"] == "MRZ-primary"
