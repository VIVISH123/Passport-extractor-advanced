"""TD3 passport MRZ parser and check-digit validator."""

import re

try:
    import pycountry
except ImportError:
    pycountry = None

_WEIGHTS = (7, 3, 1)

_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B"}
_TO_DIGIT = {v: k for k, v in _TO_LETTER.items()}

_CONF_ALNUM = {
    "0": "OD86", "O": "0D", "D": "0O", "1": "IL7", "I": "1L", "L": "1I",
    "2": "Z7", "Z": "2", "3": "8", "4": "9A", "5": "S68", "S": "5",
    "6": "G5B0", "G": "6", "7": "12", "8": "B0369", "B": "8", "9": "84",
    "U": "V", "V": "U",
}
_CONF_DIGIT = {k: "".join(c for c in v if c.isdigit()) for k, v in _CONF_ALNUM.items() if k.isdigit()}


def check_digit(text: str) -> str:
    total = 0
    for i, ch in enumerate(text):
        if ch.isdigit():
            value = int(ch)
        elif "A" <= ch <= "Z":
            value = ord(ch) - 55
        else:
            value = 0
        total += value * _WEIGHTS[i % 3]
    return str(total % 10)


def _ok(field: str, digit: str) -> bool:
    return digit.isdigit() and check_digit(field) == digit


def _valid_yymmdd(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])", value))


_LOOKALIKE_FILLER = str.maketrans({"«": "<", "‹": "<", "〈": "<", "＜": "<", "≺": "<"})


def _clean_line(line: str) -> str:
    line = line.translate(_LOOKALIKE_FILLER).upper()
    line = re.sub(r"\s+", "", line)
    return re.sub(r"[^A-Z0-9<]", "", line)


def find_mrz_lines(raw_text: str) -> list[str]:
    """Extract the two MRZ rows and repair harmless OCR line fragmentation.

    Sparse OCR (especially Tesseract PSM 11) can split a long MRZ row into two
    visual text boxes, for example:
        P<RUSPOPOVA<<POLINA<<<<<<<<
        <<<<<<<
    Those pieces are still one physical MRZ row and must be joined before the
    TD3 parser is called.
    """
    raw_lines = []
    for raw_line in raw_text.splitlines():
        cleaned = _clean_line(raw_line)
        if cleaned:
            raw_lines.append(cleaned)

    merged = []
    i = 0
    while i < len(raw_lines):
        current = raw_lines[i]

        # First MRZ row: P< / V< + names. Join short filler-only fragments
        # emitted by sparse OCR until the row is close to the 44-char target.
        if (
            len(current) >= 8
            and current[0] in {"P", "V"}
            and current[1:2] == "<"
            and len(current) < 44
        ):
            j = i + 1
            while j < len(raw_lines) and len(current) < 44:
                nxt = raw_lines[j]
                if re.fullmatch(r"<+", nxt):
                    current += nxt
                    j += 1
                else:
                    break
            merged.append(current)
            i = j
            continue

        # Second MRZ row can also be fragmented by sparse OCR. Join only a
        # numeric-looking fragment with another fragment; never join a second
        # full MRZ row accidentally.
        if len(current) >= 20 and sum(c.isdigit() for c in current) >= 5 and len(current) < 44:
            j = i + 1
            while j < len(raw_lines) and len(current) < 44:
                nxt = raw_lines[j]
                if not nxt or (nxt[0] in {"P", "V"} and nxt[1:2] == "<"):
                    break
                # If the row already has the TD3 payload through the final
                # check-digit area (42+ chars), a stray sparse-OCR fragment
                # after it is almost certainly noise.  Do not append it.
                if len(current) >= 42:
                    break
                if re.fullmatch(r"<+", nxt) or len(current) < 40:
                    current += nxt
                    j += 1
                else:
                    break
            merged.append(current)
            i = j
            continue

        merged.append(current)
        i += 1

    found = []
    for cleaned in merged:
        if len(cleaned) < 30:
            continue
        if "<" not in cleaned and sum(c.isdigit() for c in cleaned) < 6:
            continue
        found.append(cleaned)
    return found

def pad_to_length(line: str, length: int = 44) -> str:
    return line + "<" * (length - len(line)) if len(line) < length else line[:length]


def _fix_types(line: str, alpha_spans: list, numeric_spans: list, label: str, notes: list) -> str:
    chars = list(line)
    for spans, table in ((alpha_spans, _TO_LETTER), (numeric_spans, _TO_DIGIT)):
        for start, end in spans:
            for i in range(start, min(end, len(chars))):
                if chars[i] in table:
                    notes.append(
                        f"{label} position {i + 1}: '{chars[i]}' read as '{table[chars[i]]}'. Verify."
                    )
                    chars[i] = table[chars[i]]
    return "".join(chars)


def _score_line2(line2: str) -> int:
    if len(line2) < 44:
        line2 = pad_to_length(line2)
    return sum([
        _ok(line2[0:9], line2[9]),
        _ok(line2[13:19], line2[19]),
        _ok(line2[21:27], line2[27]),
        _ok(line2[28:42], line2[42]),
        _ok(line2[0:10] + line2[13:20] + line2[21:43], line2[43]),
    ])


def _line2_candidate_score(line: str) -> tuple:
    """Score a TD3 line 2 using its fixed field positions.

    Length alone is not enough: one missing/extra OCR character shifts every
    field after it. Prefer candidates that preserve the TD3 structure and
    validate the passport-number, DOB, expiry, personal-data and composite
    check digits.
    """
    if len(line) != 44:
        return (-999, 0, 0, 0)
    structural = 0
    structural += int(line[10:13].isalpha()) * 2
    structural += int(line[13:19].isdigit()) * 2
    structural += int(line[20] in "MF<")
    structural += int(line[21:27].isdigit()) * 2
    passport_ok = int(_ok(line[0:9], line[9]))
    dob_ok = int(_ok(line[13:19], line[19]))
    expiry_ok = int(_ok(line[21:27], line[27]))
    personal_ok = int(_ok(line[28:42], line[42]) or (set(line[28:42]) == {"<"} and line[42] in ("<", "0")))
    composite_ok = int(_ok(line[0:10] + line[13:20] + line[21:43], line[43]))
    # Composite validation is the strongest alignment signal when a character
    # has been dropped/inserted because it spans the fixed fields. Give it more
    # weight than the optional personal-number check.
    checks = passport_ok + dob_ok + expiry_ok + personal_ok + composite_ok
    weighted = passport_ok + dob_ok + expiry_ok + personal_ok + (composite_ok * 3)
    return (weighted * 100 + structural * 10, checks, structural, composite_ok)


def _loose_td3_candidates(raw: str):
    """Generate a bounded set of TD3 repairs for OCR strings that lost characters.

    OCR frequently drops the first one or two passport-number characters.  We do
    not brute-force all 44 positions.  Instead, use the fixed TD3 anchors
    (nationality, DOB, sex, expiry) to propose only plausible alignments, then
    let the ICAO check digits decide which alignment is best.
    """
    raw = _clean_line(raw)
    out = []

    # Candidate nationality anchors.  Keep both real-looking ISO codes and
    # synthetic/test codes such as XXX/UTO because structure matters more than
    # a local country database.
    nat_hits = []
    for m in re.finditer(r"[A-Z]{3}", raw):
        nat_hits.append(m.start())

    # Search for the characteristic TD3 tail:
    #   nationality + YYMMDD + check + sex + YYMMDD + check
    # OCR may confuse I/1, O/0, S/5, etc., so allow those common forms.
    digitish = set("0123456789OQIDZSGB")
    sexish = set("MF<")
    for nat_pos in nat_hits:
        tail = raw[nat_pos + 3:]
        if len(tail) < 14:
            continue
        # The first 6 characters after nationality are DOB, followed by a
        # check digit, sex, expiry, check digit. We only use the anchor if the
        # character classes are plausible.
        dob_part = tail[:6]
        dob_cd = tail[6:7]
        sex = tail[7:8]
        exp_part = tail[8:14]
        exp_cd = tail[14:15]
        if (
            len(dob_part) == 6 and all(c in digitish for c in dob_part)
            and dob_cd in digitish
            and sex in sexish
            and len(exp_part) == 6 and all(c in digitish for c in exp_part)
            and exp_cd in digitish
        ):
            # The TD3 prefix must occupy positions 1-13, so insert the
            # missing/extra OCR characters immediately before nationality.
            target_prefix_len = 13
            current_prefix_len = nat_pos + 3
            delta = target_prefix_len - current_prefix_len
            if 0 <= delta <= 4:
                # Most missing-prefix cases are filler/alphanumeric passport
                # characters. Build candidates from a small alphabet and then
                # score them with all check digits.  If Part 1 later supplies a
                # passport-number hint, stage2 can make this even more precise.
                alphabet = "<ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                if delta == 0:
                    out.append(raw)
                elif delta == 1:
                    for ch in alphabet:
                        out.append(raw[:nat_pos] + ch + raw[nat_pos:])
                elif delta == 2:
                    # Two missing characters are common.  Limit the first nine
                    # positions to a passport-number-like alphabet; the check
                    # digit will eliminate almost all candidates.
                    first = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                    for a in first:
                        for b in first:
                            out.append(raw[:nat_pos] + a + b + raw[nat_pos:])
                elif delta == 3:
                    # Three-character losses are rare; use a narrower alphabet
                    # and only the first passport-number positions.
                    first = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
                    for a in first:
                        for b in first:
                            for c in "0123456789<ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                                out.append(raw[:nat_pos] + a + b + c + raw[nat_pos:])

    return out


def _length_repair_line2(raw: str, notes: list) -> str:
    """Repair OCR shifts using TD3 positions, anchors and check digits.

    This is intentionally tolerant: a line does not have to be exactly 44
    characters when it enters the parser.  We first try the normal one-character
    repair path, then a bounded anchor-based repair for strings that lost two or
    more characters.  The best candidate is selected by field structure and
    ICAO check digits rather than by length alone.
    """
    raw = _clean_line(raw)
    if len(raw) == 44:
        return raw

    base = pad_to_length(raw)
    best = base
    best_score = _line2_candidate_score(base)
    best_edit = None

    candidates = []
    if len(raw) < 44 and len(raw) >= 40:
        missing = 44 - len(raw)
        for i in range(len(raw) + 1):
            candidates.append((raw[:i] + "<" * missing + raw[i:], f"insert < at {i + 1}"))
    elif len(raw) > 44 and len(raw) <= 48:
        extra = len(raw) - 44
        for i in range(len(raw) - extra + 1):
            candidates.append((raw[:i] + raw[i + extra:], f"delete {extra} char(s) at {i + 1}"))

    for candidate in _loose_td3_candidates(raw):
        candidates.append((candidate, "anchor-based TD3 insertion"))

    # Keep the existing targeted boundary repair for 43/44/45-character rows.
    if len(raw) in (43, 44, 45):
        if len(raw) == 43:
            for i in range(0, 12):
                for ch in "<0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    candidates.append((raw[:i] + ch + raw[i:], f"insert '{ch}' at {i + 1}"))
        elif len(raw) == 45:
            for i in range(0, 12):
                candidates.append((raw[:i] + raw[i + 1:], f"delete at {i + 1}"))

    # Deduplicate before scoring. A repeated OCR candidate should never win just
    # because it was generated by more than one repair route.
    seen = set()
    for candidate, edit in candidates:
        if len(candidate) != 44 or candidate in seen:
            continue
        seen.add(candidate)
        score = _line2_candidate_score(candidate)
        if score > best_score:
            best, best_score, best_edit = candidate, score, edit

    if best != base and best_edit:
        notes.append(
            f"Line 2 was positionally repaired ({best_edit}) because the TD3 field layout/check digits improved. Verify."
        )
    return best


def _repair_field(line2: str, start: int, end: int, cd_pos: int, allowed: dict, is_date: bool):
    field, cd = line2[start:end], line2[cd_pos]
    if _ok(field, cd) or not cd.isdigit():
        return None

    hits = []
    for i, ch in enumerate(field):
        for alt in allowed.get(ch, ""):
            candidate = field[:i] + alt + field[i + 1:]
            if _ok(candidate, cd) and (not is_date or _valid_yymmdd(candidate)):
                hits.append((i, ch, alt))

    if len(hits) != 1:
        return None

    i, ch, alt = hits[0]
    return line2[:start] + field[:i] + alt + field[i + 1:] + line2[end:], f"position {start + i + 1}: '{ch}' -> '{alt}'"


def _repair_line2(line2: str, notes: list) -> tuple:
    repaired = []
    fields = {
        "passport_number": (0, 9, 9, _CONF_ALNUM, False),
        "date_of_birth": (13, 19, 19, _CONF_DIGIT, True),
        "date_of_expiry": (21, 27, 27, _CONF_DIGIT, True),
    }

    for name, (start, end, cd_pos, allowed, is_date) in fields.items():
        result = _repair_field(line2, start, end, cd_pos, allowed, is_date)
        if result:
            line2, description = result
            repaired.append(name)
            notes.append(f"{name.replace('_', ' ').title()}: OCR correction accepted because the check digit passed ({description}). Confirm against the photo.")

    return line2, repaired



def _is_iso3_country(code: str) -> bool:
    code = (code or "").upper()
    if not re.fullmatch(r"[A-Z]{3}", code):
        return False
    if pycountry is None:
        return False
    try:
        return pycountry.countries.get(alpha_3=code) is not None
    except Exception:
        return False


def _repair_missing_issuing_country(line1: str, nationality: str, notes: list) -> str:
    """Keep the issuing-country slot fixed; do not invent a country code.

    The three-character issuing-state field is structurally fixed by TD3, but
    synthetic/test documents and OCR can contain codes that are not in a local
    ISO database. Inserting the nationality into this slot can shift the entire
    surname/name field, so this repair is intentionally disabled.
    """
    return line1

def _trim_name_filler_noise(line1: str, notes: list) -> str:
    """Remove OCR garbage that appears after a long MRZ filler run in the name area."""
    if len(line1) != 44:
        return line1
    start = line1.find("<<", 5)
    if start < 0:
        return line1

    given = line1[start + 2:]
    match = re.search(r"<{4,}", given)
    if not match:
        return line1
    tail = given[match.end():]
    if not tail or set(tail) <= {"<"}:
        return line1

    cut = start + 2 + match.start()
    cleaned = (line1[:cut] + "<" * (44 - cut))[:44]
    notes.append("OCR noise after the MRZ name filler was discarded. Verify names.")
    return cleaned

def _split_names(field: str) -> tuple:
    primary, separator, secondary = field.partition("<<")
    surname = primary.replace("<", " ").strip()
    given_names = secondary.replace("<", " ").strip()
    return surname, given_names


def _parse_pair(raw1: str, raw2: str, swapped: bool) -> dict:
    warnings = []
    if swapped:
        warnings.append("The two MRZ lines were read in reverse order and were swapped back.")

    line1 = pad_to_length(raw1)
    if len(raw1) != 44:
        warnings.append("Line 1 was not exactly 44 characters, so it was padded or trimmed. Verify names.")

    line2 = _length_repair_line2(_clean_line(raw2), warnings)

    # IMPORTANT: do not turn letters into '<' in line 1. Names have no check digits.
    line1 = _fix_types(line1, [(0, 44)], [], "Line 1", warnings)
    line2 = _fix_types(line2, [(10, 13), (20, 21)], [(9, 10), (13, 20), (21, 28), (42, 44)], "Line 2", warnings)
    line2, repaired = _repair_line2(line2, warnings)

    # The issuing-state slot in TD3 is fixed at positions 3-5. OCR can insert
    # or substitute a character immediately before the surname, producing cases
    # such as P<KINDAHMED... when the line-2 nationality is IND. If line 2 has
    # a structurally valid nationality, use it as an alignment anchor rather
    # than allowing the OCR error to shift the surname by one character.
    nationality_hint = line2[10:13]
    if re.fullmatch(r"[A-Z]{3}", nationality_hint):
        if line1[2:5] != nationality_hint and line1.startswith(("P<", "V<")):
            candidate_line1 = line1[:2] + nationality_hint + line1[5:]
            old_name = line1[5:].split("<<", 1)[0].replace("<", " ").strip()
            new_name = candidate_line1[5:].split("<<", 1)[0].replace("<", " ").strip()
            if new_name and (not old_name or len(new_name) >= len(old_name)):
                line1 = candidate_line1
                warnings.append(
                    f"Line 1 issuing-state alignment was repaired from '{line1[2:5]}' using the structurally valid line-2 nationality '{nationality_hint}'. Verify the name."
                )

    line1 = _repair_missing_issuing_country(line1, nationality_hint, warnings)
    line1 = _trim_name_filler_noise(line1, warnings)

    surname, given_names = _split_names(line1[5:44])

    doc_no = line2[0:9]
    nationality = line2[10:13]
    dob = line2[13:19]
    sex = line2[20]
    exp = line2[21:27]
    personal = line2[28:42]

    validation = {
        "document_number_valid": _ok(doc_no, line2[9]),
        "birth_date_valid": _ok(dob, line2[19]),
        "birth_date_format_valid": _valid_yymmdd(dob),
        "expiry_date_valid": _ok(exp, line2[27]),
        "expiry_date_format_valid": _valid_yymmdd(exp),
        "personal_number_valid": _ok(personal, line2[42]) or (set(personal) == {"<"} and line2[42] in ("<", "0")),
        "composite_valid": _ok(line2[0:10] + line2[13:20] + line2[21:43], line2[43]),
        # TD3-style MRZs can begin with P (passport) or V (visa). For OCR
        # extraction, an unknown-but-structurally-valid three-letter issuing
        # code such as UTO must not invalidate the whole MRZ.
        "country_format_valid": bool(re.fullmatch(r"[A-Z]{3}", line1[2:5])) and line1[0] in {"P", "V"},
        "nationality_format_valid": bool(re.fullmatch(r"[A-Z]{1,3}<{0,2}", nationality)),
        "sex_format_valid": sex in ("M", "F", "<"),
    }
    validation["overall_valid"] = all(validation.values())

    return {
        "success": True,
        "line1": line1,
        "line2": line2,
        "warnings": warnings,
        "repaired": repaired,
        "data": {
            "document_type": line1[0:2].replace("<", ""),
            "issuing_country": line1[2:5].replace("<", ""),
            "surname": surname,
            "given_names": given_names,
            "passport_number": doc_no,
            "nationality": nationality.replace("<", ""),
            "date_of_birth": dob,
            "sex": sex,
            "date_of_expiry": exp,
        },
        "validation": validation,
    }


def _quality(parsed: dict) -> int:
    return sum(bool(v) for k, v in parsed["validation"].items() if k != "overall_valid") * 10 + int(parsed["validation"]["overall_valid"])


def line1_plausibility(line1: str, nationality: str) -> int:
    score = 0
    score += 2 if line1[0] in {"P", "V"} else 0
    score += 2 if re.fullmatch(r"[A-Z]{1,3}<{0,2}", line1[2:5]) else 0
    score += 6 if nationality and line1[2:5].replace("<", "") == nationality else 0
    score += 3 if "<<" in line1[5:] else 0
    score += round(4 * sum(c.isalpha() or c == "<" for c in line1) / 44)
    score -= 2 * sum(c.isdigit() for c in line1)
    return score


def line1_is_good(parsed: dict) -> bool:
    line1 = parsed["line1"]
    nat = parsed["data"]["nationality"]
    return line1[0] in {"P", "V"} and "<<" in line1[5:] and bool(nat)


def with_line1(parsed: dict, other: dict) -> dict:
    new = _parse_pair(other["line1"], parsed["line2"], False)
    new["repaired"] = parsed.get("repaired", [])
    new["warnings"] = [w for w in parsed["warnings"] if not w.startswith("Line 1")] + [w for w in new["warnings"] if w not in parsed["warnings"]]
    return new


def parse_mrz(raw_text: str) -> dict:
    lines = find_mrz_lines(raw_text)
    if len(lines) < 2:
        return {"success": False, "error": f"Could not find 2 MRZ lines in the OCR text (found {len(lines)})."}

    digit_count = lambda s: sum(c.isdigit() for c in s)
    parses = []
    for i in range(len(lines) - 1):
        a, b = lines[i], lines[i + 1]
        swapped = digit_count(a) > digit_count(b)
        raw1, raw2 = (b, a) if swapped else (a, b)
        parsed = _parse_pair(raw1, raw2, swapped)
        parses.append(parsed)

    best = max(parses, key=_quality)
    nat = best["data"]["nationality"]
    alt = max(parses, key=lambda p: line1_plausibility(p["line1"], nat))
    if alt is not best and line1_plausibility(alt["line1"], nat) > line1_plausibility(best["line1"], nat):
        best = with_line1(best, alt)
    return best
