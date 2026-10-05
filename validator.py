"""
Passport field validation.

Visual OCR is the primary source when it can confidently read a field.

For names specifically:
    - MRZ structure determines which value is surname vs given names.
    - Visual OCR is used only to confirm the spelling.
    - Visual OCR cannot swap surname and given names.

MRZ is used as a fallback when visual OCR cannot read a field.
"""

import re
import datetime
import unicodedata
from difflib import SequenceMatcher


FIELDS = [
    "passport_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "sex",
    "date_of_issue",
    "date_of_expiry",
]


MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ],
        start=1,
    )
}


# ============================================================
# BASIC NORMALIZATION
# ============================================================

def _norm_alnum(value: str) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        (value or "").upper(),
    )


def _clean_name_value(value: str) -> str:
    """
    Normalize OCR name text while preserving real letters.
    """
    value = (
        unicodedata.normalize(
            "NFKD",
            value or "",
        )
        .encode(
            "ascii",
            "ignore",
        )
        .decode()
        .upper()
    )

    value = re.sub(
        r"[^A-Z\- ]",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def _name_similarity(a: str, b: str) -> float:
    """
    Compare two names after removing spaces and punctuation.
    """
    a = re.sub(
        r"[^A-Z]",
        "",
        _clean_name_value(a),
    )

    b = re.sub(
        r"[^A-Z]",
        "",
        _clean_name_value(b),
    )

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def _visual_name_matches_mrz(
    visual_value: str | None,
    mrz_value: str | None,
) -> bool:
    """
    Check whether a visually OCR'd name matches the
    name value assigned by the MRZ.

    IMPORTANT:
        MRZ determines field identity.
        Visual OCR only confirms spelling.

    This prevents:
        surname <-> given_names
    from being accidentally swapped by visual OCR.
    """

    if not visual_value or not mrz_value:
        return False

    visual = _clean_name_value(
        visual_value
    )

    mrz = _clean_name_value(
        mrz_value
    )

    if not visual or not mrz:
        return False

    # Exact normalized match.
    if visual == mrz:
        return True

    # Very close OCR spelling.
    if _name_similarity(
        visual,
        mrz,
    ) >= 0.86:
        return True

    # Handle spacing differences:
    #
    # DE LA PAZ
    # DELAPAZ
    #
    visual_compact = re.sub(
        r"[^A-Z]",
        "",
        visual,
    )

    mrz_compact = re.sub(
        r"[^A-Z]",
        "",
        mrz,
    )

    if (
        visual_compact
        and mrz_compact
        and visual_compact == mrz_compact
    ):
        return True

    return False


# ============================================================
# DATE FUNCTIONS
# ============================================================

def mrz_date_to_iso(
    yymmdd: str,
    kind: str,
) -> str | None:
    """
    Convert MRZ YYMMDD to YYYY-MM-DD.
    """

    if not re.fullmatch(
        r"\d{6}",
        yymmdd or "",
    ):
        return None

    yy = int(yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])

    if kind == "birth":

        current_yy = (
            datetime.date.today().year % 100
        )

        year = (
            2000 + yy
            if yy <= current_yy
            else 1900 + yy
        )

    else:
        year = 2000 + yy

    try:
        return datetime.date(
            year,
            mm,
            dd,
        ).isoformat()

    except ValueError:
        return None


def find_visual_dates(
    visual_text: str,
) -> list[datetime.date]:
    """
    Find common printed passport date formats.

    Examples:

        24 Oct 2022
        24 OCT 2022
        24-Oct-2022
        24/10/2022
        24.10.2022
        24-10-2022
    """

    visual_text = (
        visual_text or ""
    )

    found = []

    today = datetime.date.today()

    # --------------------------------------------------------
    # Named month dates
    # --------------------------------------------------------

    named = re.compile(
        r"\b"
        r"(\d{1,2})"
        r"\s*"
        r"([A-Za-z]{3,})"
        r"\s*"
        r"(\d{2,4})"
        r"\b"
    )

    for day, month, year in named.findall(
        visual_text
    ):

        month_number = MONTHS.get(
            month[:3].upper()
        )

        if not month_number:
            continue

        year_number = int(year)

        if len(year) == 2:

            year_number = (
                2000 + year_number
                if 2000 + year_number <= today.year
                else 1900 + year_number
            )

        try:

            found.append(
                datetime.date(
                    year_number,
                    month_number,
                    int(day),
                )
            )

        except ValueError:
            pass

    # --------------------------------------------------------
    # Numeric dates
    # --------------------------------------------------------

    numeric = re.compile(
        r"\b"
        r"(\d{1,2})"
        r"\s*[/.\-]\s*"
        r"(\d{1,2})"
        r"\s*[/.\-]\s*"
        r"(\d{2,4})"
        r"\b"
    )

    for day, month, year in numeric.findall(
        visual_text
    ):

        try:

            day_number = int(day)
            month_number = int(month)
            year_number = int(year)

            if len(year) == 2:

                year_number = (
                    2000 + year_number
                    if 2000 + year_number <= today.year
                    else 1900 + year_number
                )

            found.append(
                datetime.date(
                    year_number,
                    month_number,
                    day_number,
                )
            )

        except ValueError:
            pass

    return found


def _targeted_date_candidates(
    visual_fields: dict | None,
    field: str,
) -> list[datetime.date]:
    """Collect every valid date candidate produced by one targeted visual field."""
    fields = visual_fields if isinstance(visual_fields, dict) else {}
    item = fields.get(field, {})
    if not isinstance(item, dict):
        return []

    candidates: set[datetime.date] = set()

    value = item.get("value")
    if value:
        try:
            candidates.add(datetime.date.fromisoformat(str(value)))
        except (TypeError, ValueError):
            pass

    raw = item.get("raw") or ""
    candidates.update(find_visual_dates(str(raw)))

    return sorted(candidates)


def _labeled_date_candidates(
    visual_fields: dict | None,
    label_patterns: tuple[str, ...],
) -> list[datetime.date]:
    """Extract dates occurring near a semantic printed-field label.

    Broad OCR can find legitimate dates plus spurious dates from the photo or
    background. For Date of Issue in particular, chronological ordering alone
    is not enough. This helper uses the OCR text surrounding the actual label.
    """
    fields = visual_fields if isinstance(visual_fields, dict) else {}
    meta = fields.get("_meta", {})
    if not isinstance(meta, dict):
        return []

    texts = []
    scan_texts = meta.get("date_scan_texts", {}) or {}
    if isinstance(scan_texts, dict):
        texts.extend(str(v or "") for v in scan_texts.values())
    texts.append(str(meta.get("full_text") or ""))

    results: set[datetime.date] = set()
    label_regex = re.compile(
        r"(?:" + "|".join(label_patterns) + r")",
        re.IGNORECASE,
    )

    for text in texts:
        if not text:
            continue
        for match in label_regex.finditer(text):
            window = text[match.start(): match.start() + 180]
            results.update(find_visual_dates(window))

    return sorted(results)


def _all_visual_date_candidates(
    visual_text: str,
    visual_fields: dict | None = None,
) -> list[datetime.date]:
    """
    Collect every valid date found by visual OCR.

    Sources:
      1. General visual OCR text.
      2. Broad date-only OCR passes stored in visual_fields["_meta"].
      3. Targeted visual date fields and their raw OCR text.
    """
    candidates = set(find_visual_dates(visual_text))
    fields = visual_fields if isinstance(visual_fields, dict) else {}

    meta = fields.get("_meta", {})
    if isinstance(meta, dict):
        for value in meta.get("date_candidates", []) or []:
            try:
                candidates.add(datetime.date.fromisoformat(str(value)))
            except (TypeError, ValueError):
                pass

        # Re-parse every broad OCR pass. This matters because one OCR pass can
        # contain more than one date on the page.
        for raw in (meta.get("date_scan_texts", {}) or {}).values():
            if raw:
                candidates.update(find_visual_dates(str(raw)))

    for field in (
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
    ):
        candidates.update(
            _targeted_date_candidates(
                fields,
                field,
            )
        )

    return sorted(candidates)


def infer_date_roles(
    visual_dates: list[datetime.date],
    mrz_dob: str | None,
    mrz_exp: str | None,
    visual_fields: dict | None = None,
) -> dict:
    """
    Assign DOB / Issue / Expiry from visual date candidates.

    The normal chronological rule remains:
        earliest -> DOB
        middle   -> Issue
        latest   -> Expiry

    For Date of Issue, however, we first use a targeted OCR candidate from the
    printed Date-of-Issue field when that candidate is a single date between
    the MRZ DOB and MRZ expiry. This makes the result robust when broad OCR
    finds extra noise dates or misses one of the other printed dates.
    """
    dates = sorted(set(visual_dates))

    # Broad date OCR often has one clean pass containing exactly the three
    # printed passport dates plus another noisy pass containing a false date.
    # Prefer a three-date OCR pass when it contains the MRZ DOB and expiry.
    preferred_three_dates = None
    fields = visual_fields if isinstance(visual_fields, dict) else {}
    meta = fields.get("_meta", {}) if isinstance(fields, dict) else {}
    scan_texts = meta.get("date_scan_texts", {}) if isinstance(meta, dict) else {}
    if isinstance(scan_texts, dict):
        for raw in scan_texts.values():
            candidate_set = sorted(set(find_visual_dates(str(raw or ""))))
            if len(candidate_set) == 3:
                preferred_three_dates = candidate_set
                break

    if preferred_three_dates:
        dates = preferred_three_dates

    base = {
        "dob": None,
        "issue": None,
        "expiry": None,
        "method": "insufficient-dates",
        "candidates": dates,
    }

    # Prefer a date explicitly OCR'd near the printed Date of Issue label.
    # This defeats common broad-OCR noise such as a stray 2019/02/05 token.
    labeled_issue = _labeled_date_candidates(
        visual_fields,
        (
            r"DATE\s*(?:OF\s*)?ISSUE",
            r"DATE\s*OF\s*ISSU[E3]",
            r"DATE\s*OF\s*EXPEDITION",
        ),
    )

    try:
        dob = (
            datetime.date.fromisoformat(mrz_dob)
            if mrz_dob
            else None
        )
    except (TypeError, ValueError):
        dob = None

    try:
        exp = (
            datetime.date.fromisoformat(mrz_exp)
            if mrz_exp
            else None
        )
    except (TypeError, ValueError):
        exp = None

    # --------------------------------------------------------
    # FIRST: use a targeted Date-of-Issue candidate.
    # --------------------------------------------------------
    targeted_issue = sorted(
        set(
            _targeted_date_candidates(
                visual_fields,
                "date_of_issue",
            )
        )
        | set(labeled_issue)
    )

    if dob and exp and dob < exp:
        targeted_between = [
            d
            for d in targeted_issue
            if dob < d < exp
        ]

        if len(targeted_between) == 1:
            issue = targeted_between[0]

            # Keep DOB/expiry as visual values only when the actual visual
            # candidate list contains the MRZ anchor. Otherwise the caller
            # will naturally fall back to the MRZ for that field.
            visual_dob = dob if dob in dates else None
            visual_exp = exp if exp in dates else None

            return {
                "dob": visual_dob,
                "issue": issue,
                "expiry": visual_exp,
                "method": "targeted-issue-plus-mrz-anchors",
                "candidates": dates,
            }

    # --------------------------------------------------------
    # EXACTLY THREE DISTINCT VISUAL DATES.
    # --------------------------------------------------------
    if len(dates) == 3:
        chronological = {
            "dob": dates[0],
            "issue": dates[1],
            "expiry": dates[2],
            "method": "three-dates-chronological",
            "candidates": dates,
        }

        # Strong semantic assignment: MRZ identifies DOB + expiry;
        # the remaining visual date is therefore issue date.
        if (
            dob in dates
            and exp in dates
            and dob != exp
        ):
            remaining = [
                d
                for d in dates
                if d not in {dob, exp}
            ]

            if len(remaining) == 1:
                return {
                    "dob": dob,
                    "issue": remaining[0],
                    "expiry": exp,
                    "method": "three-dates-plus-mrz-anchors",
                    "candidates": dates,
                }

        # If the middle date is actually an MRZ anchor, reassign the roles
        # using the MRZ semantic fields.
        if chronological["issue"] in {dob, exp}:
            if (
                dob in dates
                and exp in dates
                and dob != exp
            ):
                remaining = [
                    d
                    for d in dates
                    if d not in {dob, exp}
                ]

                if len(remaining) == 1:
                    return {
                        "dob": dob,
                        "issue": remaining[0],
                        "expiry": exp,
                        "method": "mrz-corrected-issue-role",
                        "candidates": dates,
                    }

        return chronological

    # --------------------------------------------------------
    # MORE THAN THREE CANDIDATES.
    # --------------------------------------------------------
    # Extra OCR dates are common. First use the MRZ DOB/expiry as semantic
    # anchors. When OCR produces a duplicate of an anchor with a corrupted
    # year but the same day/month, collapse that OCR artifact onto the MRZ date.
    if dob and exp and dob < exp:
        normalized_dates = set(dates)

        for candidate in dates:
            if candidate == dob or candidate == exp:
                continue

            # Same day/month as DOB or expiry strongly indicates a year OCR
            # error rather than a genuinely separate passport date. Only do
            # this cleanup in the >3-candidate case where OCR noise is already
            # established.
            if candidate.day == dob.day and candidate.month == dob.month:
                normalized_dates.discard(candidate)
                normalized_dates.add(dob)
                continue

            if candidate.day == exp.day and candidate.month == exp.month:
                normalized_dates.discard(candidate)
                normalized_dates.add(exp)

        dates = sorted(normalized_dates)

    if (
        dob
        and exp
        and dob < exp
        and dob in dates
        and exp in dates
    ):
        between = [
            d
            for d in dates
            if dob < d < exp
        ]

        if len(between) == 1:
            return {
                "dob": dob,
                "issue": between[0],
                "expiry": exp,
                "method": "mrz-anchored-single-middle-date",
                "candidates": dates,
            }

    return {
        "dob": None,
        "issue": None,
        "expiry": None,
        "method": "ambiguous-date-set",
        "candidates": dates,
    }


def guess_issue_date(
    visual_text: str,
    dob_iso,
    expiry_iso,
) -> str | None:
    """Backward-compatible helper using chronological visual dates."""
    roles = infer_date_roles(
        find_visual_dates(visual_text),
        dob_iso,
        expiry_iso,
        visual_fields=None,
    )
    issue = roles.get("issue")
    return (
        issue.isoformat()
        if isinstance(issue, datetime.date)
        else None
    )


def _valid_iso_date(
    value: str | None,
) -> bool:

    if not value:
        return False

    try:

        datetime.date.fromisoformat(
            value
        )

        return True

    except ValueError:
        return False


# ============================================================
# REMOVE MRZ FROM GENERAL OCR
# ============================================================

def _remove_mrz_lines(
    raw_text: str,
) -> str:

    kept = []

    for line in (
        raw_text or ""
    ).splitlines():

        cleaned = re.sub(
            r"\s+",
            "",
            line,
        )

        if not cleaned:
            continue

        valid_chars = len(
            re.findall(
                r"[A-Z0-9<]",
                cleaned.upper(),
            )
        )

        ratio = (
            valid_chars
            / max(
                len(cleaned),
                1,
            )
        )

        if (
            len(cleaned) >= 25
            and ratio > 0.85
        ):
            continue

        kept.append(line)

    return "\n".join(kept)


# ============================================================
# NAME DETECTION
# ============================================================

def _line_is_name_candidate(
    line: str,
) -> bool:

    value = _clean_name_value(
        line
    )

    words = value.split()

    if not (
        1 <= len(words) <= 5
    ):
        return False

    blocked = {
        "PASSPORT",
        "PASAPORTE",
        "NATIONALITY",
        "NATIONALLY",
        "DATE",
        "BIRTH",
        "ISSUE",
        "EXPIRATION",
        "EXPIRY",
        "AUTHORITY",
        "UNITED",
        "STATES",
        "AMERICA",
        "DEPARTMENT",
        "TYPE",
        "SEX",
        "SURNAME",
        "GIVEN",
        "GIVENNAME",
        "GIVENNAMES",
    }

    for word in words:

        if not re.fullmatch(
            r"[A-Z]+(?:-[A-Z]+)?",
            word,
        ):
            return False

        if word in blocked:
            return False

    return True


def _field_label_match(
    line: str,
    field: str,
) -> float:

    value = _clean_name_value(
        line
    )

    compact = re.sub(
        r"[^A-Z]",
        "",
        value,
    )

    if field == "surname":

        labels = [
            "SURNAME",
            "SUR NAME",
            "APELLIDOS",
            "NOM",
            "FAMILYNAME",
        ]

        direct = {
            "SURNAME",
            "SUREAME",
            "FAMILYNAME",
        }

    else:

        labels = [
            "GIVEN NAMES",
            "GIVEN NAME",
            "GIVEN",
            "PRENOM",
            "PRENOMS",
            "NOMBRES",
            "FIRSTNAME",
        ]

        direct = {
            "GIVENNAMES",
            "GIVENNAME",
            "GIVEN",
            "PRENOM",
            "PRENOMS",
            "NOMBRES",
            "FIRSTNAME",
        }

    if any(
        item in compact
        for item in direct
    ):
        return 1.0

    scores = []

    for label in labels:

        normalized_label = re.sub(
            r"[^A-Z]",
            "",
            label,
        )

        scores.append(
            SequenceMatcher(
                None,
                compact,
                normalized_label,
            ).ratio()
        )

    return max(
        scores,
        default=0.0,
    )


def _visual_name_value(
    raw_ocr_text: str,
    field: str,
    mrz_value: str,
) -> tuple[str | None, float]:

    best_value = None
    best_score = 0.0

    for source in re.split(
        r"---[^\n]*---",
        raw_ocr_text or "",
    ):

        lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip()
        ]

        for i, line in enumerate(
            lines
        ):

            if (
                _field_label_match(
                    line,
                    field,
                )
                < 0.55
            ):
                continue

            for j in range(
                i + 1,
                min(
                    i + 4,
                    len(lines),
                ),
            ):

                candidate = (
                    _clean_name_value(
                        lines[j]
                    )
                )

                if not _line_is_name_candidate(
                    candidate
                ):
                    continue

                score = _name_similarity(
                    candidate,
                    mrz_value,
                )

                mrz_compact = re.sub(
                    r"[^A-Z]",
                    "",
                    _clean_name_value(
                        mrz_value
                    ),
                )

                candidate_compact = re.sub(
                    r"[^A-Z]",
                    "",
                    candidate,
                )

                if (
                    candidate_compact
                    and mrz_compact.startswith(
                        candidate_compact
                    )
                ):
                    score = max(
                        score,
                        0.97,
                    )

                if score > best_score:

                    best_value = candidate
                    best_score = score

    return (
        best_value,
        best_score,
    )


# ============================================================
# NAME CROSS CHECK
# ============================================================

def _name_token_confirmed(
    token: str,
    words: list[str],
) -> tuple[bool, float]:

    best = 0.0

    token_clean = (
        _clean_name_value(
            token
        )
        .replace(
            " ",
            "",
        )
    )

    for word in words:

        word_clean = (
            _clean_name_value(
                word
            )
            .replace(
                " ",
                "",
            )
        )

        if not word_clean:
            continue

        score = (
            100
            * SequenceMatcher(
                None,
                token_clean,
                word_clean,
            ).ratio()
        )

        best = max(
            best,
            score,
        )

        if (
            token_clean
            == word_clean
        ):
            return (
                True,
                100.0,
            )

        # Handle noisy MRZ suffixes.
        if (
            len(token_clean)
            > len(word_clean)
            and token_clean.startswith(
                word_clean
            )
        ):

            suffix = token_clean[
                len(word_clean):
            ]

            if (
                len(suffix) <= 10
                and set(suffix)
                <= set("KCLSEX")
            ):
                return (
                    True,
                    max(
                        score,
                        97.0,
                    ),
                )

        if (
            len(token_clean) > 3
            and abs(
                len(word_clean)
                - len(token_clean)
            ) <= 2
            and score >= 85
        ):
            return (
                True,
                score,
            )

    return (
        False,
        best,
    )


def cross_check_with_visual_text(
    mrz_data: dict,
    raw_ocr_text: str,
    visual_fields: dict | None = None,
) -> dict:

    visual = _remove_mrz_lines(
        raw_ocr_text
    ).upper()

    words = re.findall(
        r"[A-Z]{2,}",
        visual,
    )

    results = {}

    visual_fields = (
        visual_fields or {}
    )

    for field in (
        "surname",
        "given_names",
    ):

        value = (
            mrz_data.get(
                field,
                "",
            )
            or ""
        )

        targeted = (
            visual_fields.get(
                field,
                {},
            )
            if isinstance(
                visual_fields,
                dict,
            )
            else {}
        )

        targeted_value = targeted.get(
            "value"
        )

        if targeted_value:

            score = _name_similarity(
                value,
                targeted_value,
            )

            results[field] = {
                "mrz_value": value,
                "visual_value": targeted_value,
                "found_in_visual_text": (
                    score >= 0.72
                ),
                "similarity_score": round(
                    score * 100,
                    1,
                ),
                "method": targeted.get(
                    "method",
                    "targeted",
                ),
            }

            continue

        visual_value, visual_score = (
            _visual_name_value(
                visual,
                field,
                value,
            )
        )

        tokens = [
            token
            for token in value.split()
            if len(token) >= 2
        ]

        checks = [
            _name_token_confirmed(
                token,
                words,
            )
            for token in tokens
        ]

        token_ok = (
            bool(checks)
            and all(
                ok
                for ok, _ in checks
            )
        )

        found = (
            bool(
                visual_value
                and visual_score >= 0.72
            )
            or token_ok
        )

        results[field] = {
            "mrz_value": value,
            "visual_value": visual_value,
            "found_in_visual_text": found,
            "similarity_score": round(
                (
                    visual_score * 100.0
                    if visual_value
                    else min(
                        (
                            score
                            for _, score
                            in checks
                        ),
                        default=0.0,
                    )
                ),
                1,
            ),
        }

    return results


# ============================================================
# NAME PLAUSIBILITY
# ============================================================

def _name_plausibility(
    value: str,
) -> tuple[bool, str]:

    value = re.sub(
        r"\s+",
        " ",
        (value or "").strip().upper(),
    )

    if not value:
        return (
            False,
            "Name field is empty.",
        )

    tokens = value.split()

    if not tokens:
        return (
            False,
            "Name field is empty.",
        )

    for token in tokens:

        if not re.fullmatch(
            r"[A-Z]+(?:-[A-Z]+)?",
            token,
        ):
            return (
                False,
                f"Name contains non-letter OCR characters: {token!r}.",
            )

        if len(token) >= 5:

            counts = {
                c: token.count(c)
                for c in set(token)
            }

            if (
                max(counts.values())
                / len(token)
                >= 0.75
            ):
                return (
                    False,
                    f"Name token {token!r} contains an implausible repeated-character pattern.",
                )

    letters = "".join(
        tokens
    )

    if (
        len(letters) >= 6
        and len(set(letters)) <= 2
    ):
        return (
            False,
            "Name contains too little character diversity and is likely OCR noise.",
        )

    return (
        True,
        "Name has a plausible alphabetic structure.",
    )


# ============================================================
# GENERAL CROSS-CHECK
# ============================================================

def _close(
    a: str,
    b: str,
    max_edits: int = 2,
) -> bool:

    a = _norm_alnum(a)
    b = _norm_alnum(b)

    if not a or not b:
        return False

    if len(a) != len(b):
        return False

    return (
        sum(
            x != y
            for x, y in zip(a, b)
        )
        <= max_edits
    )


def cross_check_number_and_dates(
    mrz_data: dict,
    dob_iso,
    exp_iso,
    visual_text: str,
    visual_fields: dict | None = None,
) -> dict:

    visual = _remove_mrz_lines(
        visual_text
    ).upper()

    visual_fields = (
        visual_fields or {}
    )

    out = {}

    # --------------------------------------------------------
    # Passport number
    # --------------------------------------------------------

    number = (
        mrz_data.get(
            "passport_number"
        )
        or ""
    ).replace(
        "<",
        "",
    )

    targeted_number = _visual_value(
        visual_fields,
        "passport_number",
    )

    if targeted_number:

        if (
            _norm_alnum(
                targeted_number
            )
            == _norm_alnum(
                number
            )
        ):

            out["passport_number"] = {
                "status": "match",
                "visual_value": targeted_number,
            }

        else:

            out["passport_number"] = {
                "status": "conflict",
                "visual_value": targeted_number,
            }

    else:

        tokens = [
            token
            for token in re.findall(
                r"\b[A-Z0-9]{7,12}\b",
                visual,
            )
            if any(
                c.isdigit()
                for c in token
            )
        ]

        if not number:

            out["passport_number"] = {
                "status": "not found",
                "visual_value": None,
            }

        elif number in tokens:

            out["passport_number"] = {
                "status": "match",
                "visual_value": number,
            }

        else:

            near = [
                token
                for token in tokens
                if _close(
                    token,
                    number,
                    2,
                )
            ]

            if near:

                out["passport_number"] = {
                    "status": "conflict",
                    "visual_value": near[0],
                }

            else:

                out["passport_number"] = {
                    "status": "not found",
                    "visual_value": None,
                }

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    vdates = _all_visual_date_candidates(
        visual_text,
        visual_fields,
    )

    for field, iso in (
        (
            "date_of_birth",
            dob_iso,
        ),
        (
            "date_of_expiry",
            exp_iso,
        ),
    ):

        targeted = (
            visual_fields.get(
                field,
                {},
            )
            if isinstance(
                visual_fields,
                dict,
            )
            else {}
        )

        targeted_value = targeted.get(
            "value"
        )

        if targeted_value:

            if (
                iso
                and targeted_value == iso
            ):

                out[field] = {
                    "status": "match",
                    "visual_value": targeted_value,
                    "method": targeted.get(
                        "method",
                        "targeted",
                    ),
                }

            else:

                out[field] = {
                    "status": "conflict",
                    "visual_value": targeted_value,
                    "method": targeted.get(
                        "method",
                        "targeted",
                    ),
                }

            continue

        if not iso:

            out[field] = {
                "status": "not found",
                "visual_value": None,
            }

            continue

        try:

            mrz_date = (
                datetime.date.fromisoformat(
                    iso
                )
            )

        except ValueError:

            out[field] = {
                "status": "not found",
                "visual_value": None,
            }

            continue

        if mrz_date in vdates:

            out[field] = {
                "status": "match",
                "visual_value": iso,
            }

            continue

        near = [
            x
            for x in vdates
            if (
                (
                    x.day,
                    x.month,
                )
                == (
                    mrz_date.day,
                    mrz_date.month,
                )
            )
            or (
                (
                    x.year,
                    x.month,
                )
                == (
                    mrz_date.year,
                    mrz_date.month,
                )
            )
        ]

        if near:

            out[field] = {
                "status": "conflict",
                "visual_value": near[0].isoformat(),
            }

        else:

            out[field] = {
                "status": "not found",
                "visual_value": None,
            }

    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    targeted_sex = (
        visual_fields.get(
            "sex",
            {},
        ).get(
            "value"
        )
        if isinstance(
            visual_fields,
            dict,
        )
        else None
    )

    if targeted_sex in {
        "M",
        "F",
    }:

        mrz_sex = (
            mrz_data.get(
                "sex",
                "",
            )
            or ""
        ).upper()

        out["sex"] = {
            "status": (
                "match"
                if targeted_sex == mrz_sex
                else "conflict"
            ),
            "visual_value": targeted_sex,
        }

    else:

        out["sex"] = {
            "status": "not found",
            "visual_value": None,
        }

    return out


# ============================================================
# VISUAL FIELD HELPERS
# ============================================================

def _visual_item(
    visual_fields: dict | None,
    field: str,
) -> dict:

    if not isinstance(
        visual_fields,
        dict,
    ):
        return {}

    item = visual_fields.get(
        field,
        {},
    )

    return (
        item
        if isinstance(
            item,
            dict,
        )
        else {}
    )


def _visual_value(
    visual_fields: dict | None,
    field: str,
):

    item = _visual_item(
        visual_fields,
        field,
    )

    value = item.get(
        "value"
    )

    if value in (
        None,
        "",
    ):
        return None

    return str(
        value
    ).strip()


def _visual_confidence(
    visual_fields: dict | None,
    field: str,
) -> float:

    item = _visual_item(
        visual_fields,
        field,
    )

    try:

        return float(
            item.get(
                "confidence",
                0,
            )
            or 0
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _visual_valid(
    field: str,
    value: str | None,
) -> bool:

    if not value:
        return False

    value = str(
        value
    ).strip().upper()

    # --------------------------------------------------------
    # Passport number
    # --------------------------------------------------------

    if field == "passport_number":

        compact = re.sub(
            r"[^A-Z0-9]",
            "",
            value,
        )

        return (
            6 <= len(compact) <= 12
            and any(
                c.isdigit()
                for c in compact
            )
        )

    # --------------------------------------------------------
    # Names
    # --------------------------------------------------------

    if field in {
        "surname",
        "given_names",
    }:

        return bool(
            re.fullmatch(
                r"[A-Z][A-Z '\-]{1,59}",
                value,
            )
        )

    # --------------------------------------------------------
    # Nationality
    # --------------------------------------------------------

    if field == "nationality":

        return bool(
            re.fullmatch(
                r"[A-Z]{3}",
                value,
            )
        )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    if field in {
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
    }:

        return _valid_iso_date(
            value
        )

    # --------------------------------------------------------
    # Sex
    # --------------------------------------------------------

    if field == "sex":

        return value in {
            "M",
            "F",
        }

    return False


# ============================================================
# COUNTRY NORMALIZATION
# ============================================================

def _country_text_to_code(
    value: str | None,
) -> str | None:

    if not value:
        return None

    raw = re.sub(
        r"\s+",
        " ",
        value.upper(),
    ).strip()

    if re.fullmatch(
        r"[A-Z]{3}",
        raw,
    ):
        return raw

    aliases = {

        "UNITED STATES":
            "USA",

        "UNITED STATES OF AMERICA":
            "USA",

        "UNITED KINGDOM":
            "GBR",

        "GREAT BRITAIN":
            "GBR",

        "UK":
            "GBR",

        "INDIA":
            "IND",

        "CHINA":
            "CHN",

        "SINGAPORE":
            "SGP",

        "MALAYSIA":
            "MYS",

        "INDONESIA":
            "IDN",

        "THAILAND":
            "THA",

        "PHILIPPINES":
            "PHL",

        "FINLAND":
            "FIN",

        "OMAN":
            "OMN",

        "TAIWAN":
            "TWN",

        "HONG KONG":
            "HKG",

        "GERMANY":
            "DEU",

        "FRANCE":
            "FRA",

        "ITALY":
            "ITA",

        "SPAIN":
            "ESP",

        "CANADA":
            "CAN",

        "AUSTRALIA":
            "AUS",

        "NEW ZEALAND":
            "NZL",

        "IRELAND":
            "IRL",

        "JAPAN":
            "JPN",

        "SOUTH KOREA":
            "KOR",

        "REPUBLIC OF KOREA":
            "KOR",

        "UNITED ARAB EMIRATES":
            "ARE",

        "QATAR":
            "QAT",

        "SAUDI ARABIA":
            "SAU",

        "VIETNAM":
            "VNM",

        "BANGLADESH":
            "BGD",

        "NEPAL":
            "NPL",

        "PAKISTAN":
            "PAK",

        "SRI LANKA":
            "LKA",

        "BRUNEI":
            "BRN",

        "CAMBODIA":
            "KHM",

        "LAOS":
            "LAO",

        "MYANMAR":
            "MMR",

        "MALDIVES":
            "MDV",

        "MAURITIUS":
            "MUS",

        "SOUTH AFRICA":
            "ZAF",

        "EGYPT":
            "EGY",

        "TURKEY":
            "TUR",

        "SWITZERLAND":
            "CHE",

        "AUSTRIA":
            "AUT",

        "BELGIUM":
            "BEL",

        "NETHERLANDS":
            "NLD",

        "NORWAY":
            "NOR",

        "SWEDEN":
            "SWE",

        "DENMARK":
            "DNK",

        "POLAND":
            "POL",

        "PORTUGAL":
            "PRT",

        "GREECE":
            "GRC",

        "CZECH REPUBLIC":
            "CZE",

        "CZECHIA":
            "CZE",

        "HUNGARY":
            "HUN",

        "ROMANIA":
            "ROU",

        "UKRAINE":
            "UKR",

        "RUSSIA":
            "RUS",

        "MEXICO":
            "MEX",

        "BRAZIL":
            "BRA",

        "ARGENTINA":
            "ARG",

        "CHILE":
            "CHL",

        "COLOMBIA":
            "COL",

        "PERU":
            "PER",
    }

    if raw in aliases:
        return aliases[raw]

    try:

        import pycountry

        try:

            return pycountry.countries.lookup(
                raw
            ).alpha_3

        except LookupError:

            fuzzy = (
                pycountry.countries.search_fuzzy(
                    raw
                )
            )

            if fuzzy:
                return fuzzy[0].alpha_3

    except Exception:
        pass

    return None


# ============================================================
# MRZ HELPERS
# ============================================================

def _mrz_field(
    mrz_result: dict | None,
    field: str,
) -> str | None:

    if not isinstance(
        mrz_result,
        dict,
    ):
        return None

    if not mrz_result.get(
        "success"
    ):
        return None

    value = (
        mrz_result.get(
            "data",
            {},
        ).get(
            field
        )
    )

    if value in (
        None,
        "",
    ):
        return None

    return str(
        value
    ).strip()


def _mrz_field_valid(
    mrz_result: dict | None,
    validation_key: str,
) -> bool:

    return bool(
        isinstance(
            mrz_result,
            dict,
        )
        and mrz_result.get(
            "success"
        )
        and mrz_result.get(
            "validation",
            {},
        ).get(
            validation_key
        )
    )


# ============================================================
# FINAL RESULT
# ============================================================

def failure_result(
    message: str,
) -> dict:

    return {
        "success": False,

        "data": {
            field: None
            for field in FIELDS
        },

        "field_status": {
            field: "not detected"
            for field in FIELDS
        },

        "status_reasons": {
            field: "Nothing could be read."
            for field in FIELDS
        },

        "warnings": [
            message,
            "Employee must enter all fields manually.",
        ],

        "cross_check": {},
    }


def build_result(
    mrz_result: dict | None,
    visual_text: str,
    visual_fields: dict | None = None,
) -> dict:

    visual_fields = (
        visual_fields
        if isinstance(
            visual_fields,
            dict,
        )
        else {}
    )

    visual_text = (
        visual_text
        or ""
    )

    # --------------------------------------------------------
    # MRZ data
    # --------------------------------------------------------

    d = (
        mrz_result.get(
            "data",
            {},
        )
        if isinstance(
            mrz_result,
            dict,
        )
        else {}
    )

    v = (
        mrz_result.get(
            "validation",
            {},
        )
        if isinstance(
            mrz_result,
            dict,
        )
        else {}
    )

    repaired = (
        set(
            mrz_result.get(
                "repaired",
                [],
            )
        )
        if isinstance(
            mrz_result,
            dict,
        )
        else set()
    )

    mrz_is_usable = bool(
        isinstance(
            mrz_result,
            dict,
        )
        and mrz_result.get(
            "success"
        )
    )

    mrz_dob = mrz_date_to_iso(
        d.get(
            "date_of_birth",
            "",
        ),
        "birth",
    )

    mrz_exp = mrz_date_to_iso(
        d.get(
            "date_of_expiry",
            "",
        ),
        "expiry",
    )

    mrz_values = {
        "passport_number":
            _mrz_field(
                mrz_result,
                "passport_number",
            ),

        "surname":
            _mrz_field(
                mrz_result,
                "surname",
            ),

        "given_names":
            _mrz_field(
                mrz_result,
                "given_names",
            ),

        "nationality":
            _mrz_field(
                mrz_result,
                "nationality",
            ),

        "date_of_birth":
            mrz_dob,

        "sex":
            _mrz_field(
                mrz_result,
                "sex",
            ),

        "date_of_expiry":
            mrz_exp,
    }

    warnings = (
        list(
            mrz_result.get(
                "warnings",
                [],
            )
        )
        if isinstance(
            mrz_result,
            dict,
        )
        else []
    )

    data = {}
    status = {}
    why = {}

    # --------------------------------------------------------
    # Set field
    # --------------------------------------------------------

    def set_field(
        field: str,
        value,
        state: str,
        reason: str,
    ):

        # MRZ filler characters are useful internally for parsing, but they
        # must never appear in the final employee-facing output.
        if isinstance(value, str):
            value = value.replace("<", "").replace(">", "").strip()

        if value in (
            None,
            "",
        ):

            data[field] = None
            status[field] = "not detected"
            why[field] = "No value found."

        else:

            data[field] = value
            status[field] = state
            why[field] = reason

    # --------------------------------------------------------
    # Visual first -> MRZ fallback
    # --------------------------------------------------------

    def visual_then_mrz(
        field: str,
        visual_value: str | None,
        mrz_value: str | None,
        mrz_valid: bool,
        label: str,
        confidence: float,
    ):

        # ====================================================
        # 1. VISUAL OCR FIRST
        # ====================================================

        if (
            visual_value
            and _visual_valid(
                field,
                visual_value,
            )
        ):

            if mrz_value:

                # ------------------------------------------------
                # Names
                # ------------------------------------------------

                if field in {
                    "surname",
                    "given_names",
                }:

                    agrees = (
                        _visual_name_matches_mrz(
                            visual_value,
                            mrz_value,
                        )
                    )

                # ------------------------------------------------
                # Passport number
                # ------------------------------------------------

                elif field == "passport_number":

                    agrees = (
                        _norm_alnum(
                            visual_value
                        )
                        == _norm_alnum(
                            mrz_value
                        )
                    )

                # ------------------------------------------------
                # Other fields
                # ------------------------------------------------

                else:

                    agrees = (
                        visual_value.upper()
                        == mrz_value.upper()
                    )

                if agrees:

                    return (
                        visual_value,
                        "reliable",
                        (
                            "Visual OCR and MRZ "
                            f"independently agree on "
                            f"the {label}."
                        ),
                    )

                # IMPORTANT:
                # Keep visual value when it conflicts.
                return (
                    visual_value,
                    "needs verification",
                    (
                        f"Visual OCR reads "
                        f"{visual_value}, while "
                        f"the MRZ reads "
                        f"{mrz_value}. "
                        "The visual value is kept "
                        "as the primary value; "
                        "verify the passport image."
                    ),
                )

            # Visual OCR only.
            return (
                visual_value,
                (
                    "reliable"
                    if confidence >= 80
                    else "needs verification"
                ),
                (
                    f"The {label} was extracted "
                    "from the printed passport "
                    "text by visual OCR."
                ),
            )

        # ====================================================
        # 2. MRZ FALLBACK
        # ====================================================

        if (
            mrz_value
            and mrz_valid
        ):

            if field in repaired:

                return (
                    mrz_value,
                    "needs verification",
                    (
                        f"Visual OCR did not read "
                        f"the {label}. The MRZ "
                        "fallback required an OCR "
                        "repair, so confirm the "
                        "value against the "
                        "passport image."
                    ),
                )

            return (
                mrz_value,
                "mrz validated",
                (
                    f"Visual OCR did not read "
                    f"the {label}; the MRZ was "
                    "used as fallback and its "
                    "check digit passed."
                ),
            )

        # ====================================================
        # 3. VISUAL VALUE WITHOUT MRZ CONFIRMATION
        # ====================================================

        if visual_value:

            return (
                visual_value,
                "needs verification",
                (
                    f"Visual OCR produced a "
                    f"possible {label}, but "
                    "it could not be independently "
                    "validated."
                ),
            )

        # ====================================================
        # 4. NOTHING
        # ====================================================

        return (
            None,
            "not detected",
            f"Could not detect the {label}.",
        )

    # ========================================================
    # VISUAL NATIONALITY
    # ========================================================

    visual_nationality_raw = _visual_value(
        visual_fields,
        "nationality",
    )

    visual_nationality = (
        _country_text_to_code(
            visual_nationality_raw
        )
    )

    # ========================================================
    # VISUAL PASSPORT NUMBER
    # ========================================================

    visual_number = _visual_value(
        visual_fields,
        "passport_number",
    )

    if not visual_number:

        candidates = [
            token
            for token in re.findall(
                r"\b[A-Z0-9]{6,12}\b",
                visual_text.upper(),
            )
            if (
                any(
                    c.isdigit()
                    for c in token
                )
                and any(
                    c.isalpha()
                    for c in token
                )
            )
        ]

        if candidates:
            visual_number = candidates[0]

    # ========================================================
    # PASSPORT NUMBER
    # ========================================================

    value, state, reason = visual_then_mrz(
        "passport_number",
        visual_number,
        mrz_values[
            "passport_number"
        ],
        _mrz_field_valid(
            mrz_result,
            "document_number_valid",
        ),
        "passport number",
        _visual_confidence(
            visual_fields,
            "passport_number",
        )
        or (
            70.0
            if visual_number
            else 0.0
        ),
    )

    set_field(
        "passport_number",
        value,
        state,
        reason,
    )

    # ========================================================
    # NAMES
    # ========================================================
    #
    # MRZ STRUCTURE determines identity:
    #
    #     P<CCC PRIMARY<<SECONDARY<<<<<<<<
    #
    # PRIMARY   → surname
    # SECONDARY → given names
    #
    # Visual OCR can only confirm spelling.
    # It cannot redefine which field is which.
    # ========================================================

    for field, label in (
        (
            "surname",
            "surname",
        ),
        (
            "given_names",
            "given names",
        ),
    ):

        mrz_value = (
            mrz_values.get(
                field
            )
            or ""
        ).strip()

        visual_value = _visual_value(
            visual_fields,
            field,
        )

        # ----------------------------------------------------
        # MRZ gives the structural field assignment.
        # ----------------------------------------------------

        if mrz_value:

            # Visual OCR did not return a candidate.
            if not visual_value:

                set_field(
                    field,
                    mrz_value,
                    "mrz validated",
                    (
                        f"Visual OCR did not "
                        f"read the {label}. "
                        "The MRZ structure "
                        f"identifies this value "
                        f"as the {label}; "
                        "the MRZ value was used "
                        "as fallback."
                    ),
                )

                continue

            # ------------------------------------------------
            # Visual OCR returned something.
            # Verify it belongs to THIS MRZ field.
            # ------------------------------------------------

            if _visual_name_matches_mrz(
                visual_value,
                mrz_value,
            ):

                # Keep the clean visual spelling.
                set_field(
                    field,
                    visual_value,
                    "reliable",
                    (
                        f"Visual OCR reads "
                        f"{visual_value} and it "
                        f"matches the {label} "
                        "identified by the "
                        "MRZ structure."
                    ),
                )

            else:

                # The visual OCR candidate may have
                # come from the wrong field.
                #
                # DO NOT allow it to swap names.
                set_field(
                    field,
                    mrz_value,
                    "needs verification",
                    (
                        f"Visual OCR produced "
                        f"{visual_value!r}, but "
                        f"it does not match the "
                        f"MRZ {label} "
                        f"{mrz_value!r}. "
                        "The MRZ structure is "
                        "used to keep the name "
                        "assigned to the correct "
                        "field. Verify the "
                        "printed name."
                    ),
                )

        else:

            # ------------------------------------------------
            # No MRZ name exists.
            # Only in this case can visual OCR
            # determine the value by itself.
            # ------------------------------------------------

            if (
                visual_value
                and _visual_valid(
                    field,
                    visual_value,
                )
            ):

                confidence = (
                    _visual_confidence(
                        visual_fields,
                        field,
                    )
                )

                set_field(
                    field,
                    visual_value,
                    (
                        "reliable"
                        if confidence >= 80
                        else "needs verification"
                    ),
                    (
                        f"The {label} was "
                        "extracted from "
                        "the printed passport "
                        "text by visual OCR."
                    ),
                )

            else:

                set_field(
                    field,
                    None,
                    "not detected",
                    (
                        f"Could not detect "
                        f"the {label} from "
                        "either source."
                    ),
                )

    # ========================================================
    # NATIONALITY
    # ========================================================

    value, state, reason = visual_then_mrz(
        "nationality",
        visual_nationality,
        mrz_values[
            "nationality"
        ],
        bool(
            v.get(
                "nationality_format_valid"
            )
        ),
        "nationality",
        _visual_confidence(
            visual_fields,
            "nationality",
        ),
    )

    set_field(
        "nationality",
        value,
        state,
        reason,
    )

    # ========================================================
    # DATES
    # ========================================================
    #
    # Visual OCR is the primary source for all three printed dates.
    #
    # Primary visual rule:
    #     earliest -> DOB
    #     middle   -> Issue
    #     latest   -> Expiry
    #
    # MRZ is then used as an independent semantic anchor for DOB + Expiry.
    # Date of Issue is not expected in the MRZ, so its confidence is NOT
    # lowered merely because the MRZ has no issue-date field.
    # ========================================================

    date_candidates = _all_visual_date_candidates(
        visual_text,
        visual_fields,
    )

    date_roles = infer_date_roles(
        date_candidates,
        mrz_dob,
        mrz_exp,
        visual_fields=visual_fields,
    )

    visual_dob_date = date_roles.get("dob")
    visual_issue_date = date_roles.get("issue")
    visual_expiry_date = date_roles.get("expiry")

    # --------------------------------------------------------
    # DATE OF BIRTH
    # --------------------------------------------------------

    if isinstance(visual_dob_date, datetime.date):
        visual_dob_iso = visual_dob_date.isoformat()

        if (
            mrz_dob
            and visual_dob_iso == mrz_dob
            and _mrz_field_valid(mrz_result, "birth_date_valid")
        ):
            set_field(
                "date_of_birth",
                visual_dob_iso,
                "reliable",
                (
                    "Visual OCR identified the earliest passport date as the "
                    "date of birth, and the MRZ independently confirms it."
                ),
            )
        elif mrz_dob:
            set_field(
                "date_of_birth",
                visual_dob_iso,
                "needs verification",
                (
                    f"Visual OCR assigned {visual_dob_iso} as the date of birth, "
                    f"but the MRZ identifies {mrz_dob}. Verify the passport image."
                ),
            )
        else:
            set_field(
                "date_of_birth",
                visual_dob_iso,
                "reliable",
                "Date of birth was identified from the chronological visual OCR date set.",
            )
    elif mrz_dob:
        state = (
            "needs verification"
            if "date_of_birth" in repaired
            else "mrz validated"
        )
        set_field(
            "date_of_birth",
            mrz_dob,
            state,
            (
                "Visual OCR could not classify the printed dates; the MRZ "
                "date-of-birth value was used as fallback."
            ),
        )
    else:
        set_field(
            "date_of_birth",
            None,
            "not detected",
            "Could not identify the date of birth.",
        )

    # --------------------------------------------------------
    # DATE OF EXPIRY
    # --------------------------------------------------------

    if isinstance(visual_expiry_date, datetime.date):
        visual_expiry_iso = visual_expiry_date.isoformat()

        if (
            mrz_exp
            and visual_expiry_iso == mrz_exp
            and _mrz_field_valid(mrz_result, "expiry_date_valid")
        ):
            set_field(
                "date_of_expiry",
                visual_expiry_iso,
                "reliable",
                (
                    "Visual OCR identified the latest passport date as the "
                    "date of expiry, and the MRZ independently confirms it."
                ),
            )
        elif mrz_exp:
            set_field(
                "date_of_expiry",
                visual_expiry_iso,
                "needs verification",
                (
                    f"Visual OCR assigned {visual_expiry_iso} as the date of "
                    f"expiry, but the MRZ identifies {mrz_exp}. Verify the passport image."
                ),
            )
        else:
            set_field(
                "date_of_expiry",
                visual_expiry_iso,
                "reliable",
                "Date of expiry was identified from the chronological visual OCR date set.",
            )
    elif mrz_exp:
        state = (
            "needs verification"
            if "date_of_expiry" in repaired
            else "mrz validated"
        )
        set_field(
            "date_of_expiry",
            mrz_exp,
            state,
            (
                "Visual OCR could not classify the printed dates; the MRZ "
                "date-of-expiry value was used as fallback."
            ),
        )
    else:
        set_field(
            "date_of_expiry",
            None,
            "not detected",
            "Could not identify the date of expiry.",
        )

    # --------------------------------------------------------
    # SEX
    # --------------------------------------------------------
    # Sex is represented in the MRZ, so use that as the reliable fallback
    # whenever visual OCR misses the small printed M/F value.
    # --------------------------------------------------------

    visual_sex = _visual_value(
        visual_fields,
        "sex",
    )

    if visual_sex:
        visual_sex = visual_sex.upper()

    mrz_sex = (
        mrz_values.get(
            "sex",
        )
        or ""
    ).upper()

    if visual_sex in {"M", "F"}:
        if mrz_sex in {"M", "F"} and visual_sex != mrz_sex:
            set_field(
                "sex",
                visual_sex,
                "needs verification",
                (
                    f"Visual OCR reads {visual_sex}, while the MRZ reads "
                    f"{mrz_sex}. Verify the printed sex value."
                ),
            )
        else:
            set_field(
                "sex",
                visual_sex,
                "reliable",
                "Visual OCR identified the printed sex value.",
            )
    elif mrz_sex in {"M", "F"}:
        sex_state = (
            "needs verification"
            if "sex" in repaired
            else "mrz validated"
        )
        set_field(
            "sex",
            mrz_sex,
            sex_state,
            (
                "Visual OCR did not read the printed sex value; the MRZ "
                "was used as fallback."
            ),
        )
    else:
        set_field(
            "sex",
            None,
            "not detected",
            "Could not detect the sex from visual OCR or the MRZ.",
        )

    # --------------------------------------------------------
    # DATE OF ISSUE
    # --------------------------------------------------------
    #
    # Issue date is visual-only.
    # A successfully inferred middle date is considered reliable; it is NOT
    # downgraded because the standard MRZ does not contain an issue-date field.
    # --------------------------------------------------------

    if isinstance(visual_issue_date, datetime.date):
        issue_iso = visual_issue_date.isoformat()
        set_field(
            "date_of_issue",
            issue_iso,
            "reliable",
            (
                "Visual OCR identified the middle chronological passport date "
                "as the date of issue. DOB and expiry are independently checked "
                "against the MRZ."
            ),
        )
    else:
        set_field(
            "date_of_issue",
            None,
            "not detected",
            (
                "Visual OCR did not produce an unambiguous set of passport "
                "dates from which the date of issue could be identified."
            ),
        )
        warnings.append(
            "Date of issue was not confidently detected from the visual date set; "
            "verify or enter it manually."
        )

    # Save role inference for debugging.
    date_role_debug = {
        "candidates": [
            d.isoformat()
            for d in date_roles.get("candidates", [])
            if isinstance(d, datetime.date)
        ],
        "dob": (
            visual_dob_date.isoformat()
            if isinstance(visual_dob_date, datetime.date)
            else None
        ),
        "issue": (
            visual_issue_date.isoformat()
            if isinstance(visual_issue_date, datetime.date)
            else None
        ),
        "expiry": (
            visual_expiry_date.isoformat()
            if isinstance(visual_expiry_date, datetime.date)
            else None
        ),
        "method": date_roles.get("method"),
    }

    # ========================================================
    # CROSS-CHECK EVIDENCE
    # ========================================================

    cross = cross_check_number_and_dates(
        d,
        mrz_dob,
        mrz_exp,
        visual_text,
        visual_fields,
    )

    cross["date_roles"] = date_role_debug

    names_cross = (
        cross_check_with_visual_text(
            d,
            visual_text,
            visual_fields,
        )
    )

    # Add targeted visual values to debug information.
    for field in (
        "passport_number",
        "date_of_birth",
        "date_of_expiry",
        "date_of_issue",
        "sex",
    ):

        existing = cross.get(
            field,
            {},
        )

        if not isinstance(
            existing,
            dict,
        ):
            existing = {}

        visual_value = _visual_value(
            visual_fields,
            field,
        )

        if visual_value:

            existing[
                "visual_value"
            ] = visual_value

        cross[field] = existing

    # ========================================================
    # WARNINGS
    # ========================================================

    for field in (
        "passport_number",
        "date_of_birth",
        "date_of_expiry",
        "sex",
    ):

        check = cross.get(
            field,
            {},
        )

        if (
            check.get(
                "status"
            )
            == "conflict"
        ):

            warnings.append(
                f"{field.replace('_', ' ').title()}: "
                "visual OCR and MRZ disagree "
                f"(visual: {check.get('visual_value')}, "
                f"MRZ: {check.get('mrz_value')}). "
                "The visual value is kept as primary; "
                "verify against the passport image."
            )

    # ========================================================
    # EXPIRY
    # ========================================================

    if mrz_exp:

        try:

            if (
                datetime.date.fromisoformat(
                    mrz_exp
                )
                < datetime.date.today()
            ):

                warnings.append(
                    "The MRZ indicates that "
                    "this passport is expired."
                )

        except ValueError:
            pass

    # ========================================================
    # FINAL SUCCESS
    # ========================================================

    usable = sum(
        data.get(
            field
        )
        not in (
            None,
            "",
        )
        for field in FIELDS
    )

    success = (
        usable >= 3
        or mrz_is_usable
    )

    if not success:

        return failure_result(
            (
                "The passport image did "
                "not yield enough readable "
                "fields. Please upload a "
                "clearer data-page image."
            )
        )

    return {
        "success": True,

        "data": {
            field: data.get(
                field
            )
            for field in FIELDS
        },

        "field_status": {
            field: status.get(
                field,
                "not detected",
            )
            for field in FIELDS
        },

        "status_reasons": {
            field: why.get(
                field,
                "No value found.",
            )
            for field in FIELDS
        },

        "warnings": warnings,

        "cross_check": {
            "names": names_cross,
            **cross,
        },
    }