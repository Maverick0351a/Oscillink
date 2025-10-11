from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

# Minimal curated common typos list (extend as needed)
COMMON_TYPOS: Dict[str, str] = {
    "teh": "the",
    "recieve": "receive",
    "seperate": "separate",
    "definately": "definitely",
    "occured": "occurred",
    "accomodate": "accommodate",
    "acheive": "achieve",
    "adress": "address",
    "becuase": "because",
    "comming": "coming",
    "enviroment": "environment",
    "goverment": "government",
    "independant": "independent",
    "occuring": "occurring",
    "occurence": "occurrence",
    "publically": "publicly",
    "seperately": "separately",
    "wich": "which",
}

# Patterns to preserve technical/code-like tokens
SKIP_PATTERNS = [
    re.compile(r"^[A-Z]{2,}$"),  # Acronyms: API, LLM, SPD
    re.compile(r"^[A-Za-z]+_[A-Za-z0-9_]+$"),  # snake_case
    re.compile(r"^[A-Za-z]+[A-Z][A-Za-z0-9]+$"),  # camelCase/PascalCase
    re.compile(r".*[0-9].*"),  # contains digit
    re.compile(r".*[()\[\]{}<>`].*"),  # code-like
]

DEFAULT_PRESERVE = {
    "Oscillink",
    "lamG",
    "lamC",
    "lamQ",
    "SPD",
    "kNN",
    "k-NN",
    "API",
    "LLM",
    "FFT",
}

_PUNCT = ",.!?;:\"'’”()[]{}"


def _match_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src.istitle():
        return repl.title()
    return repl


def _is_code_fence(line: str) -> bool:
    return line.strip().startswith("```")


def _should_skip_token(core: str, preserve: Iterable[str]) -> bool:
    if not core:
        return True
    if core in preserve:
        return True
    if "://" in core or "@" in core:
        return True
    if not core.isascii():
        return True
    return any(p.match(core) for p in SKIP_PATTERNS)


def _process_token(tok: str, preserve: Iterable[str]) -> str:
    if not tok:
        return tok
    left = ""
    right = ""
    core = tok
    while core and core[0] in _PUNCT:
        left += core[0]
        core = core[1:]
    while core and core[-1] in _PUNCT:
        right = core[-1] + right
        core = core[:-1]

    if _should_skip_token(core, preserve):
        return tok

    lower = core.lower()
    repl = COMMON_TYPOS.get(lower)
    if repl is None:
        return tok
    fixed = _match_case(core, repl)
    return f"{left}{fixed}{right}"


def smart_correct(text: str, custom_preserve: Optional[List[str]] = None) -> str:
    """Correct common typos while preserving technical terms and code.

    - Only applies corrections for words present in COMMON_TYPOS (case-insensitive)
    - Skips tokens matching technical/code-like patterns
    - Preserves case (upper/title/lower) when applying replacements
    - Respects Markdown code fences ```...```
    """
    preserve = set(DEFAULT_PRESERVE)
    if custom_preserve:
        preserve.update(custom_preserve)

    out_lines: List[str] = []
    in_code = False
    for line in text.splitlines():
        if _is_code_fence(line):
            in_code = not in_code
            out_lines.append(line)
            continue
        if in_code:
            out_lines.append(line)
            continue

        new_tokens = [_process_token(tok, preserve) for tok in line.split(" ")]
        out_lines.append(" ".join(new_tokens))

    return "\n".join(out_lines)


__all__ = ["smart_correct", "COMMON_TYPOS"]
