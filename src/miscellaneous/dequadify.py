#!/usr/bin/env python3
"""
quad_to_dd.py — replace quad-precision (Q-suffix) float literals in C source
files with double-double {hi, lo} compound-literal initializers, and replace
all FLIT() calls with FCONST().

Usage:
    python quad_to_dd.py file.c [file.h ...]

Transformations (applied in order):
    dd_from_f128(<Q-literal>)   →  ((dd_t){hi, lo})
    dd_from_f128(M_PIq)         →  ((dd_t){hi_pi, lo_pi})
    FCONST(<Q-literal>)         →  FCONST(hi, lo)
    FCONST(M_PIq)               →  FCONST(hi_pi, lo_pi)
    FLIT(<Q-literal>)           →  FCONST(hi, lo)
    FLIT(M_PIq)                 →  FCONST(hi_pi, lo_pi)
    FLIT(<non-Q expr>)          →  FCONST(<non-Q expr>)      [fallback]

After running this script on all source files you can remove:
    - dd_from_f128() / f128_from_dd() function definitions in nanofft.h
    - #include <quadmath.h>
    - dd_from_f128 / f128_from_dd prototypes / uses everywhere

Notes:
    • mpmath is used at 128-bit (≈ quad) precision so the hi/lo split is
      faithful to the original quad constant, not merely to a double
      approximation of its decimal text.
    • The hi component is the nearest double; lo = round(x − hi) in double.
    • Output double literals use %.17g which guarantees IEEE 754 round-trip.
      A hex-float fallback fires in the (practically impossible) event that
      %.17g doesn't round-trip.
    • M_PIq (GCC quadmath π) is recognised as a special token and replaced
      with the correctly split π constant.
    • The script is idempotent: running it twice produces the same result.
"""

import re
import sys
import mpmath

# ---------------------------------------------------------------------------
# Precision setup — 128-bit significand matches GCC __float128 / quadmath
# ---------------------------------------------------------------------------
mpmath.mp.prec = 128

_MP_PI = mpmath.pi          # the only named quad constant seen so far
_NAMED = {'M_PIq': _MP_PI}  # extend here if more quadmath macros appear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _c_double(f: float) -> str:
    """Format a Python float as a C double literal that round-trips exactly."""
    if f != f:          # NaN — shouldn't occur in polynomial coefficients
        return '__builtin_nan("")'
    s = f'{f:.17g}'
    if float(s) == f:
        return s
    # Fallback: C99 hex float — always exact
    return f.__format__('a')


def _dd(mp_val: mpmath.mpf) -> tuple[float, float]:
    """Split an mpmath value into an exactly-representable (hi, lo) pair."""
    hi = float(mp_val)
    lo = float(mp_val - mpmath.mpf(hi))
    return hi, lo


def _fmt_dd(mp_val: mpmath.mpf) -> str:
    """Return 'hi, lo' suitable for FCONST(hi, lo) or (dd_t){hi, lo}."""
    hi, lo = _dd(mp_val)
    return f'{_c_double(hi)}, {_c_double(lo)}'


def _parse_q(raw: str) -> mpmath.mpf:
    """
    Convert a matched Q-literal token or named constant to mpmath.

    raw may be:
      - a named macro like 'M_PIq'
      - a signed decimal Q-suffix literal like '-0.000123Q'
    """
    raw = raw.strip()
    if raw in _NAMED:
        return _NAMED[raw]
    # Strip trailing Q (or q, though GCC only uses uppercase)
    num_str = raw.rstrip('Qq').replace(' ', '')
    return mpmath.mpf(num_str)


# ---------------------------------------------------------------------------
# Regex pieces
# ---------------------------------------------------------------------------

# A signed decimal Q-suffix literal.  The sign may be separated from the
# digits by optional whitespace (odd but harmless to accept).
# Captures the full token including trailing Q.
_Q_NUM_PAT = r'[+-]?\s*(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[Qq]'

# Named quad constant tokens we handle
_NAMED_PAT = '|'.join(re.escape(k) for k in _NAMED)

# Either a named constant or a Q-suffix literal
_Q_ARG_PAT = rf'(?:{_NAMED_PAT}|{_Q_NUM_PAT})'

# dd_from_f128(<Q-arg>) — the whole call is replaced
_DD_FROM_F128_RE = re.compile(
    rf'dd_from_f128\(\s*({_Q_ARG_PAT})\s*\)'
)

# FCONST(<Q-arg>) or FLIT(<Q-arg>)
_FCONST_Q_RE = re.compile(
    rf'(?:FCONST|FLIT)\(\s*({_Q_ARG_PAT})\s*\)'
)

# FLIT(<non-Q content>) — anything left after the Q passes above
# Matches single-level parens; nested parens are not expected here.
_FLIT_PLAIN_RE = re.compile(r'FLIT\(([^)]*)\)')


# ---------------------------------------------------------------------------
# Replacement callbacks
# ---------------------------------------------------------------------------

def _repl_dd_from_f128(m: re.Match) -> str:
    mp_val = _parse_q(m.group(1))
    return f'((dd_t){{{_fmt_dd(mp_val)}}})'


def _repl_fconst_or_flit_q(m: re.Match) -> str:
    mp_val = _parse_q(m.group(1))
    return f'FCONST({_fmt_dd(mp_val)})'


def _repl_flit_plain(m: re.Match) -> str:
    inner = m.group(1).strip()
    return f'FCONST({inner})'


# ---------------------------------------------------------------------------
# Main transformation
# ---------------------------------------------------------------------------

def transform(src: str) -> str:
    """Apply all replacements to a source string and return the result."""
    # Step 1: direct dd_from_f128(<Q-arg>) calls
    src = _DD_FROM_F128_RE.sub(_repl_dd_from_f128, src)

    # Step 2: FCONST(<Q-arg>) and FLIT(<Q-arg>)
    src = _FCONST_Q_RE.sub(_repl_fconst_or_flit_q, src)

    # Step 3: remaining FLIT(<anything>) → FCONST(<anything>)
    src = _FLIT_PLAIN_RE.sub(_repl_flit_plain, src)

    return src


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    for path in sys.argv[1:]:
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                original = fh.read()
        except OSError as exc:
            print(f'[error] cannot read {path}: {exc}', file=sys.stderr)
            continue

        updated = transform(original)

        if updated == original:
            print(f'[no change] {path}')
            continue

        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(updated)
        print(f'[updated]   {path}')


if __name__ == '__main__':
    main()
