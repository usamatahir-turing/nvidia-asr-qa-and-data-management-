#!/usr/bin/env python3
"""Thorough NSV fix_words regression checks (run: python test_fix_words_nsv.py)."""

from __future__ import annotations

import re
import sys
from collections import defaultdict

from fix_seglst_tokens import (
    ALLOWED_NSVS,
    TOKEN_SPELLING_FIXES,
    fix_words,
    normalize_token_content,
)

ZW = "\u200b"


def bracket_variants(body: str) -> list[tuple[str, str]]:
    """Return (input, description) variants for a token body."""
    variants: list[tuple[str, str]] = []

    def add(inp: str, desc: str) -> None:
        variants.append((inp, desc))

    add(f"[{body}]", "bracketed")
    add(f"[{body.upper()}]", "bracketed_upper")
    add(f"[ {body} ]", "bracketed_spaced")
    add(f"[{ZW}{body}]", "bracketed_zw")
    add(f"{body}]", "missing_open")
    add(f"{body} ]", "missing_open_space_before_close")
    add(f"[[{body}]]", "double_brackets")
    add(f"[{body}]]", "extra_close")
    add(f"[[{body}]", "extra_open")

    if "-" in body:
        parts = body.split("-")
        if len(parts) == 2:
            a, b = parts
            add(f"[{a} {b}]", "bracketed_space")
            add(f"[{a}- {b}]", "bracketed_hyphen_space")
            add(f"[{a} - {b}]", "bracketed_spaced_hyphen")
            add(f"{a}- {b}]", "missing_open_hyphen_space")
            add(f"{a} {b}]", "missing_open_space")
            add(f"speech {body}]", "missing_open_with_speech_prefix")
    else:
        add(f"speech {body}]", "missing_open_with_speech_prefix")

    add(f"[ {body} and more speech", "unclosed_then_speech")
    add(f"[{body} and more speech", "unclosed_no_leading_space")

    return variants


def expected_for(desc: str, canonical: str) -> str:
    if desc == "missing_open_with_speech_prefix":
        return f"speech [{canonical}]"
    if desc in {"unclosed_then_speech", "unclosed_no_leading_space"}:
        return f"[{canonical}] and more speech"
    return f"[{canonical}]"


def has_duplicate_prefix(out: str, canonical: str) -> bool:
    """Detect ``other- [other-noise]`` style duplicate residue."""
    patterns = [
        r"other-\s*\[other-noise\]",
        r"lip-\s*\[lip-smack\]",
        r"clear-\s*\[clear-throat\]",
        r"tongue-\s*\[tongue-click\]",
        r"teeth-\s*\[teeth-suck\]",
    ]
    for pat in patterns:
        if re.search(pat, out):
            return True
    return False


def main() -> int:
    failures: list[str] = []
    stats = defaultdict(int)

    print("Testing canonical ALLOWED_NSVS...")
    for canonical in sorted(ALLOWED_NSVS):
        for inp, desc in bracket_variants(canonical):
            out = fix_words(inp)
            exp = expected_for(desc, canonical)
            stats["canonical_total"] += 1
            if out != exp:
                failures.append(
                    f"[{canonical}] {desc}: in={inp!r} out={out!r} exp={exp!r}"
                )
                stats["canonical_fail"] += 1
            elif has_duplicate_prefix(out, canonical):
                failures.append(f"[{canonical}] {desc}: duplicate prefix in {out!r}")
                stats["canonical_fail"] += 1
            else:
                stats["canonical_pass"] += 1

    print("Testing TOKEN_SPELLING_FIXES misspellings...")
    for misspelling, canonical in sorted(TOKEN_SPELLING_FIXES.items()):
        for inp, desc in bracket_variants(misspelling):
            if desc in {
                "missing_open_with_speech_prefix",
                "unclosed_then_speech",
                "unclosed_no_leading_space",
            }:
                continue
            out = fix_words(inp)
            exp = expected_for(desc, canonical)
            stats["misspelling_total"] += 1
            if out != exp:
                failures.append(
                    f"[{misspelling}->{canonical}] {desc}: in={inp!r} out={out!r} exp={exp!r}"
                )
                stats["misspelling_fail"] += 1
            elif has_duplicate_prefix(out, canonical):
                failures.append(
                    f"[{misspelling}->{canonical}] {desc}: duplicate prefix in {out!r}"
                )
                stats["misspelling_fail"] += 1
            else:
                stats["misspelling_pass"] += 1

    print("Testing known regression cases...")
    regressions = [
        ("other-noise]", "[other-noise]"),
        ("other- noise]", "[other-noise]"),
        ("lip-smack]", "[lip-smack]"),
        ("lip- smack]", "[lip-smack]"),
        ("clear-throat]", "[clear-throat]"),
        ("clear- throat]", "[clear-throat]"),
        ("teeth-suck]", "[teeth-suck]"),
        ("tongue-click]", "[tongue-click]"),
        ("[click]", "[other-noise]"),
        ("[noise]", "[other-noise]"),
        ("[smack]", "[lip-smack]"),
        ("inhale][", "[inhale]"),
        ("clear--throat]", "[clear-throat]"),
        (f"[\u200binhale]", "[inhale]"),
    ]
    for inp, expected in regressions:
        out = fix_words(inp)
        stats["regression_total"] += 1
        if out != expected:
            failures.append(f"[regression] in={inp!r} exp={expected!r} got={out!r}")
            stats["regression_fail"] += 1
        else:
            stats["regression_pass"] += 1

    for token in sorted(ALLOWED_NSVS):
        if normalize_token_content(token) != token:
            failures.append(f"normalize_token_content({token!r}) != {token!r}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Canonical:   {stats['canonical_pass']}/{stats['canonical_total']} passed")
    print(
        f"Misspelling: {stats['misspelling_pass']}/{stats['misspelling_total']} passed"
    )
    print(
        f"Regression:  {stats['regression_pass']}/{stats['regression_total']} passed"
    )
    print(f"Total failures: {len(failures)}")

    if failures:
        print()
        print("FAILURES:")
        for line in failures:
            print(f"  - {line}")
        return 1

    print("All tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
