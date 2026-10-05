import json
from io import BytesIO

import streamlit as st
import cv2
import numpy as np

from PIL import Image, ImageOps, UnidentifiedImageError

from extractor import extract_passport
from stage2 import run_stage2


MAX_BYTES = 15 * 1024 * 1024


LABELS = {
    "passport_number": "Passport Number",
    "surname": "Surname",
    "given_names": "Given Names",
    "nationality": "Nationality",
    "date_of_birth": "Date of Birth",
    "sex": "Sex",
    "date_of_issue": "Date of Issue",
    "date_of_expiry": "Date of Expiry",
}


STATUS = {
    "reliable": "✅ Reliable",
    "mrz validated": "✅ MRZ Validated",
    "needs verification": "⚠️ Verify",
    "not detected": "❌ Not detected",
}


st.set_page_config(
    page_title="Passport Extractor",
    layout="centered",
)


st.title("Passport Information Extractor")

st.caption(
    "Local prototype: passport images are processed on this computer. "
    "Employees should verify extracted values before use."
)


# ============================================================
# Helpers
# ============================================================


def render_result_table(result: dict, title: str, success_message: str | None = None):
    st.header(title)

    if result.get("success"):
        st.success(
            success_message
            or "Passport read. Please verify the fields below against the document."
        )
    else:
        st.error(
            "The passport could not be read reliably. "
            "Please enter the details manually."
        )

    cross_check = result.get("cross_check", {})
    rows = []

    for field, label in LABELS.items():
        value = result.get("data", {}).get(field)
        field_status = result.get("field_status", {}).get(field, "not detected")
        reason = result.get("status_reasons", {}).get(field, "")

        visual_candidate = None
        if field in {
            "date_of_birth",
            "date_of_expiry",
            "date_of_issue",
            "passport_number",
            "sex",
        }:
            visual_candidate = (
                cross_check.get(field, {}).get("visual_value")
                if isinstance(cross_check.get(field, {}), dict)
                else None
            )

        display_reason = reason
        if visual_candidate and field_status == "needs verification":
            display_reason = f"{reason} Visual candidate: {visual_candidate}."

        rows.append(
            {
                "Field": label,
                "Value": value if value not in (None, "") else "—",
                "Status": STATUS.get(
                    field_status,
                    f"⚠️ {field_status}",
                ),
                "Why": display_reason,
            }
        )

    st.table(rows)

    # Keep an independent Part 3 visual-OCR opinion visibly separate from
    # the final MRZ-primary value whenever the two disagree. This prevents
    # the secondary OCR path from silently overwriting the structured MRZ
    # result while still giving the reviewer the alternate reading.
    part3_evidence = cross_check.get("part3_visual_ocr_fields", {})
    disagreements = []
    for field, item in part3_evidence.items():
        if not isinstance(item, dict):
            continue
        independent = item.get("part3_visual")
        final_value = result.get("data", {}).get(field)
        if independent not in (None, "") and final_value not in (None, ""):
            if field in {"surname", "given_names"}:
                same = str(independent).replace("<", " ").strip().upper() == str(final_value).replace("<", " ").strip().upper()
            elif field == "passport_number":
                same = str(independent).replace("<", "").strip().upper() == str(final_value).replace("<", "").strip().upper()
            else:
                same = str(independent).strip().upper() == str(final_value).strip().upper()
            if not same:
                disagreements.append({
                    "Field": LABELS.get(field, field),
                    "Final / MRZ-primary": final_value,
                    "Independent Part 3 OCR": independent,
                    "Comparison": "DIFFERENT — verify",
                })

    if disagreements:
        st.warning("Independent Part 3 visual OCR has a different opinion. It is shown separately and has NOT silently replaced the final MRZ-primary value.")
        st.table(disagreements)


def render_warnings(result: dict, title: str = "Validation and warnings"):
    st.header(title)

    flagged = [
        LABELS[field]
        for field in LABELS
        if result.get("field_status", {}).get(field, "not detected")
        != "reliable"
    ]

    if flagged:
        st.warning("Employee must verify: " + ", ".join(flagged))
    else:
        st.success("All fields were independently confirmed.")

    for warning in result.get("warnings", []):
        st.warning(warning)


# ============================================================
# 1. Upload
# ============================================================

st.header("1. Upload")

uploaded = st.file_uploader(
    "Passport photo or scan (JPG, JPEG or PNG)",
    type=["jpg", "jpeg", "png"],
)

show_tech = st.checkbox(
    "Show technical details (OCR output contains passport data)"
)

if uploaded is None:
    st.info("Upload a passport data-page image to begin.")
    st.stop()

if uploaded.size > MAX_BYTES:
    st.error("That file is larger than 15 MB. Please upload a smaller image.")
    st.stop()

try:
    # Decode directly from the uploaded bytes. We never re-encode the upload
    # before Stage 1/Stage 2; the original pixel resolution is retained.
    uploaded_bytes = uploaded.getvalue()
    image = Image.open(BytesIO(uploaded_bytes))
    image.load()
    image = ImageOps.exif_transpose(image).convert("RGB")
except (
    UnidentifiedImageError,
    OSError,
    Image.DecompressionBombError,
):
    st.error("That file could not be opened as an image.")
    st.stop()

st.image(
    image,
    caption=f"{uploaded.name} · {image.width} × {image.height} · {image.format}",
    use_container_width=True,
)


# ============================================================
# 2. Stage 1 — existing working pipeline
# ============================================================

st.header("2. Stage 1 — Existing extraction")

with st.status(
    "Preprocessing, reading MRZ, reading printed text, cross-checking…",
    expanded=False,
) as stage1_status:
    try:
        stage1_result = extract_passport(
            image,
            include_debug=True,
        )
    except Exception as exc:
        stage1_result = {
            "success": False,
            "data": {field: None for field in LABELS},
            "field_status": {field: "not detected" for field in LABELS},
            "status_reasons": {field: "Stage 1 raised an unexpected error." for field in LABELS},
            "warnings": [f"Stage 1 error: {type(exc).__name__}: {exc}"],
            "cross_check": {},
            "debug": {},
        }

    if stage1_result.get("success"):
        stage1_status.update(label="Stage 1 complete", state="complete")
    else:
        stage1_status.update(
            label="Stage 1 produced an incomplete result; Stage 2 will still run",
            state="complete",
        )

stage1_debug = stage1_result.get("debug", {})


def stage1_fully_satisfied(result: dict) -> bool:
    """Return True only when Stage 1 has a complete, trusted result.

    Stage 2 is skipped only when every expected field has a value and every
    field is already in a final trusted state. A value marked "needs
    verification" is intentionally NOT considered complete.
    """
    if not isinstance(result, dict) or not result.get("success"):
        return False

    data = result.get("data") or {}
    status = result.get("field_status") or {}

    allowed_states = {"reliable", "mrz validated"}

    for field in LABELS:
        value = data.get(field)
        field_state = str(status.get(field, "")).strip().lower()

        if value in (None, ""):
            return False

        if field_state not in allowed_states:
            return False

    return True


stage1_complete = stage1_fully_satisfied(stage1_result)

# ============================================================
# 3. Initial Stage 1 result — shown BEFORE Stage 2
# ============================================================

render_result_table(
    stage1_result,
    "3. Initial extraction result",
)

st.info(
    "Initial extraction result. Additional analysis may still be performed. "
    "Verify all passport fields against the original document."
)


# ============================================================
# 4. Stage 2 — conditional
# ============================================================

st.header("4. Stage 2 — Document normalization and MRZ-primary analysis")

if stage1_complete:
    # Stage 1 already has every requested field in a trusted final state.
    # Do NOT run Stage 2 at all. This preserves the exact Stage 1 output and
    # avoids unnecessary border detection / perspective correction.
    stage2_result = dict(stage1_result)
    stage2_result["stage2"] = {
        "success": True,
        "skipped": True,
        "source_priority": "stage1-complete",
        "reason": "Stage 1 already satisfied every required field.",
    }
    stage2_result["debug"] = {
        "stage2_skipped": True,
        "reason": "Stage 1 was fully satisfied; Stage 2 was not executed.",
    }

    st.success(
        "Stage 2 skipped — Stage 1 already satisfied every required field. "
        "No border detection, perspective correction, or Stage 2 OCR was run."
    )

else:
    with st.status(
        "Detecting passport border, normalizing document, splitting regions, and re-validating the MRZ…",
        expanded=False,
    ) as stage2_status:
        try:
            stage2_result = run_stage2(
                image,
                stage1_result=stage1_result,
                stage1_debug=stage1_debug,
            )
        except Exception as exc:
            stage2_result = dict(stage1_result)
            stage2_result.setdefault("warnings", []).append(
                f"Stage 2 error: {type(exc).__name__}: {exc}. The Stage 1 result was retained."
            )
            stage2_result["stage2"] = {
                "success": False,
                "source_priority": "stage1-retained",
                "mrz_trustworthy": False,
            }
            stage2_result["debug"] = {"error": str(exc)}

        source = stage2_result.get(
            "stage2_source_priority",
            stage2_result.get("stage2", {}).get("source_priority"),
        )
        if source == "MRZ-primary":
            stage2_status.update(
                label="Stage 2 complete — MRZ-primary",
                state="complete",
            )
        elif source in {"visual-recovery-fallback", "stage1-retained"}:
            stage2_status.update(
                label="Stage 2 completed with fallback/recovery",
                state="complete",
            )
        else:
            stage2_status.update(
                label="Stage 2 completed",
                state="complete",
            )

stage2_debug = stage2_result.get("debug", {})

# ============================================================
# 5. Stage 2 document visualization
# ============================================================

st.header("5. Stage 2 document analysis")

if stage1_complete:
    st.info(
        "Stage 2 was not executed because Stage 1 is already complete. "
        "There is therefore no Stage 2 corrected image or Stage 2 split to display."
    )
else:
    st.subheader("Corrected passport (before splitting)")

    normalized = stage2_debug.get("normalized_passport")
    if normalized is not None:
        st.caption(
            "This is the exact passport image produced after border detection and "
            "perspective correction, before Part 1 / Part 2 / Part 3 splitting."
        )
        border = stage2_debug.get("border_detection", {})
        crop_info = stage2_debug.get("crop", {})
        st.caption(
            f"Border method: {border.get('method', 'unknown')} · "
            f"Detection confidence: {border.get('confidence', 0):.3f} · "
            f"Perspective warp: {'yes' if crop_info.get('warped') else 'no'} · "
            f"Orientation preserved: {'yes' if crop_info.get('orientation_preserved') else 'unknown'} · "
            f"Working size: {normalized.width} × {normalized.height}"
        )
        # Show the exact detected border as a visual sanity check.
        border_quad = border.get("quad")
        stage2_input = stage2_debug.get("stage2_input")
        if border_quad is not None and stage2_input is not None:
            try:
                overlay = np.array(stage2_input.convert("RGB"))
                pts = np.asarray(border_quad, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], True, (255, 0, 0), max(2, overlay.shape[1] // 300))
                st.image(
                    overlay,
                    caption="Detected passport border — red outline",
                    use_container_width=True,
                )
            except Exception as exc:
                st.caption(f"Border overlay unavailable: {type(exc).__name__}")

        st.image(
            normalized,
            caption="Stage 2 corrected passport — BEFORE SPLITTING",
            use_container_width=True,
        )
    else:
        border = stage2_debug.get("border_detection", {}) or {}
        pipeline_error = stage2_debug.get("error") or stage2_debug.get("stage2", {}).get("pipeline_error")
        if border.get("success") is False and pipeline_error:
            st.warning(
                "Stage 2 kept the original-quality image because the pipeline "
                f"encountered an error after/before normalization: {pipeline_error}."
            )
        elif border.get("success") is False:
            st.warning(
                "Passport border detection did not pass. Stage 2 kept the "
                "original Stage-1-oriented image as a safe fallback."
            )
        else:
            st.warning(
                "Stage 2 did not produce a corrected image, but the border "
                "result is being preserved in Technical details for diagnosis."
            )
        st.image(
            stage2_debug.get("stage2_input", image),
            caption="Stage 2 input — original-quality fallback (no border correction)",
            use_container_width=True,
        )

    # ALWAYS show the actual images passed into the three split regions whenever
    # Stage 2 ran. This is intentionally image-based rather than just showing
    # coordinates, so a bad split is immediately visible.
    parts = stage2_debug.get("parts", {})
    st.subheader("Stage 2 split regions — visual inspection")

    if parts:
        for name, label in (
            ("part1", "Part 1 — visual passport details"),
            ("part2", "Part 2 — visual passport details"),
            ("part3", "Part 3 — MRZ region"),
        ):
            part = parts.get(name)
            if part is not None:
                st.image(
                    part,
                    caption=f"{label} · {part.width} × {part.height}",
                    use_container_width=True,
                )
    else:
        st.warning(
            "Stage 2 produced no split images. This indicates a processing error, not a border-detection rejection."
        )

# ============================================================
# 6. Final Stage 2 result
# ============================================================

final_result = stage2_result

source_priority = final_result.get(
    "stage2_source_priority",
    final_result.get("stage2", {}).get("source_priority", "unknown"),
)

if source_priority == "MRZ-primary":
    final_message = (
        "Final result produced by Stage 2. MRZ-validated fields are primary; "
        "visual OCR is secondary confirmation."
    )
elif source_priority == "visual-recovery-fallback":
    final_message = (
        "Stage 2 MRZ was not sufficiently trustworthy, so the existing visual "
        "recovery validator was used. Verify the flagged fields."
    )
else:
    final_message = (
        "Stage 2 could not replace the Stage 1 result, so the Stage 1 result was retained."
    )

render_result_table(
    final_result,
    "6. Final extracted passport information",
    success_message=final_message,
)

# ------------------------------------------------------------
# Final independent guess / validation box
# ------------------------------------------------------------
# This is intentionally separate from the normal Stage 2 result. It is a
# conservative last-pass interpretation from the MRZ parser plus ordinary
# visual OCR over all three physical split regions. It never silently
# overwrites the primary result.
final_guess = final_result.get("final_guess") or {}
if final_guess:
    st.subheader("Final guess — independently cross-validated")
    st.caption(
        "This is a separate safety-net result. It is shown only as a validated "
        "guess and does not silently replace the MRZ-primary result above."
    )

    guess_rows = []
    guess_data = final_guess.get("data", {}) or {}
    guess_validation = final_guess.get("validation", {}) or {}
    guess_sources = final_guess.get("sources", {}) or {}
    guess_conflicts = final_guess.get("conflicts", {}) or {}

    for field, label in LABELS.items():
        value = guess_data.get(field)
        if value in (None, ""):
            value = "—"
        state = guess_validation.get(field, "not validated")
        source = ", ".join(guess_sources.get(field, [])) or "—"
        conflict = guess_conflicts.get(field)
        if conflict:
            source += " | conflict: " + " / ".join(conflict)
            if state == "validated":
                state = "validated — conflict also observed"
        guess_rows.append({
            "Field": label,
            "Final guess": value,
            "Validation": state,
            "Evidence": source,
        })

    st.table(guess_rows)
    st.caption(
        f"Validated fields: {final_guess.get('validated_field_count', 0)} / {len(LABELS)}. "
        "Visual-only corrections require independent corroboration; ambiguous values are not invented."
    )

render_warnings(
    final_result,
    "7. Final validation and warnings",
)


# ============================================================
# 8. Technical details
# ============================================================

if show_tech or not final_result.get("success"):
    st.header("8. Technical details")

    # ---------------- Stage 1 ----------------
    with st.expander("Stage 1 — existing MRZ pipeline"):
        st.write(
            "Selected orientation: "
            f"**{stage1_debug.get('selected_orientation') if stage1_debug.get('selected_orientation') is not None else 'None'}°**"
        )
        st.write(
            "Best MRZ attempt: "
            f"**{stage1_debug.get('mrz_variant_used') or 'None'}**"
        )
        st.write("**Parsed MRZ lines:**")
        st.code(
            "\n".join(
                line
                for line in stage1_debug.get("mrz_lines", [])
                if line
            )
            or "(none)"
        )
        st.write("**MRZ validation:**")
        st.json(stage1_debug.get("mrz_validation") or {})
        st.write("**MRZ OCR attempts:**")
        for name, text in stage1_debug.get("mrz_ocr_texts", {}).items():
            with st.expander(name):
                st.code(text or "(empty)")
        if stage1_debug.get("orientation_scores"):
            st.write("**Orientation evaluation:**")
            st.json(stage1_debug.get("orientation_scores"))

    # ---------------- Stage 1 visual OCR ----------------
    with st.expander("Stage 1 — visual OCR"):
        st.write(
            "Selected pass: "
            f"**{stage1_debug.get('visual_selected_pass') or 'None'}**"
        )
        st.code(stage1_debug.get("visual_ocr_text") or "(empty)")
        st.json(stage1_debug.get("visual_fields") or {})

    # ---------------- Stage 2 border / split ----------------
    with st.expander("Stage 2 — border detection and normalization"):
        st.json(
            {
                "input_size": stage2_debug.get("input_size"),
                "selected_orientation_from_stage1": stage2_debug.get(
                    "selected_orientation_from_stage1"
                ),
                "border_detection": stage2_debug.get("border_detection"),
                "crop": stage2_debug.get("crop"),
                "normalized_size": stage2_debug.get("normalized_size"),
                "split": stage2_debug.get("split"),
            }
        )

    # ---------------- Stage 2 MRZ ----------------
    with st.expander("Stage 2 — MRZ primary analysis"):
        st.json(stage2_debug.get("stage2_mrz") or {})

    # ---------------- Stage 2 visual OCR ----------------
    with st.expander("Stage 2 — visual OCR"):
        st.code(stage2_debug.get("stage2_visual_text") or "(empty)")
        st.json(stage2_debug.get("stage2_visual_fields") or {})
        st.write("**Per-part visual OCR:**")
        st.json(stage2_debug.get("stage2_part_visual_ocr") or {})

    # ---------------- Cross-check ----------------
    with st.expander("Stage 2 — MRZ vs visual cross-check"):
        st.json(final_result.get("cross_check") or {})

    # ---------------- Structured output ----------------
    with st.expander("Final structured output (future API shape)"):
        st.code(
            json.dumps(
                {
                    "success": final_result.get("success"),
                    "data": final_result.get("data"),
                    "field_status": final_result.get("field_status"),
                    "status_reasons": final_result.get("status_reasons"),
                    "warnings": final_result.get("warnings"),
                    "stage2": final_result.get("stage2"),
                    "stage2_source_priority": final_result.get("stage2_source_priority"),
                },
                indent=2,
            ),
            language="json",
        )
