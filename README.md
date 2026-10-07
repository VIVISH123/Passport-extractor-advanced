# Passport Extractor

Local passport information extraction prototype using Tesseract OCR, MRZ parsing/validation, visual OCR, document normalization, and Stage 2 MRZ-primary analysis.

## What it does

The application extracts:

- Passport number
- Surname
- Given names
- Nationality
- Date of birth
- Sex
- Date of issue
- Date of expiry

Employees must verify the extracted values against the original passport before using them for booking.

## Architecture

```text
Uploaded passport
      |
      v
STAGE 1 — existing working pipeline
      |
      +--> EXIF correction / preprocessing
      +--> MRZ OCR (multiple attempts)
      +--> MRZ parser + OCR repair
      +--> check-digit validation
      +--> MRZ quality + orientation selection
      +--> targeted visual OCR
      +--> existing visual-first validator
      |
      v
Initial result shown to employee
      |
      v
STAGE 2 — always runs
      |
      +--> detect passport border ONCE
      +--> crop + safe perspective correction
      +--> normalize working resolution
      +--> split normalized passport into Part 1 / Part 2 / Part 3
      |
      +--> Part 1 -> visual OCR
      +--> Part 2 -> visual OCR
      +--> Part 3 -> existing MRZ OCR/parser/check validation
      |
      +--> MRZ is PRIMARY for Stage 2
      +--> visual OCR is secondary confirmation
      +--> existing date-role inference for Date of Issue
      |
      v
Final Stage 2 result
```

### Stage 1 preservation

The original Stage 1 MRZ integration is kept intact. Stage 2 is implemented as a separate layer so the existing OCR, parser, check-digit, orientation, name, date, and validator logic remains available.

### Stage 2 border and split rules

The border is detected against the whole image once. The detected passport is then cropped, safely perspective-corrected when the quadrilateral is trustworthy, and normalized. Part 1, Part 2 and Part 3 are all slices of that same normalized passport. There is no per-part rotation sweep or per-part border redetection.

### Stage 2 source priority

For Stage 2:

| Field | Primary | Secondary |
|---|---|---|
| Passport number | MRZ | Visual OCR |
| Surname | MRZ structure | Visual OCR |
| Given names | MRZ structure | Visual OCR |
| Nationality | MRZ | Visual OCR |
| Date of birth | MRZ | Visual OCR |
| Sex | MRZ | Visual OCR |
| Date of issue | Visual OCR + existing date-role logic | — |
| Date of expiry | MRZ | Visual OCR |

A conflicting visual value never silently replaces a trustworthy Stage 2 MRZ value; the MRZ value is retained and the field is flagged for verification.

## Install

Create and activate a virtual environment if desired, then:

```bash
pip install -r requirements.txt
```

Install Tesseract OCR separately. On Windows, a common installer is the UB Mannheim build.

Verify it:

```bash
python check_tesseract.py
```

## Run

```bash
streamlit run app.py
```

Or double-click `run.bat` on Windows.

## Privacy

The extractor is designed for local processing. It does not intentionally send passport images to cloud OCR or AI services. Uploaded images are processed in memory by the extraction code and are not intentionally persisted by the extractor.

Do not put real customer passports into Git, public repositories, logs, or test fixtures.

## Important limitation

OCR output is not guaranteed to be correct. MRZ check digits improve structural validation but do not make OCR infallible. The workflow therefore requires employee verification before booking data is used.
