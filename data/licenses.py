"""The data license allow-list (docs/PLAN.md "Data").

Allowed for training and benchmark data: Apache-2.0, MIT, BSD, CC0, CC BY
(any version), ODC-By 1.0, public domain under FSEK article 31, and text
authored by the lab. Everything else is out: ShareAlike, NonCommercial,
NoDerivatives, research-only, unlicensed, unverifiable upstream.

The check is deny-by-default. A license id passes only if it is spelled
exactly as one of the ids below. Nothing is inferred from free text.
"""

from __future__ import annotations

import re

# SPDX identifiers, plus two LicenseRef ids of our own.
ALLOWED_LICENSE_IDS: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "CC0-1.0",
        "CC-BY-1.0",
        "CC-BY-2.0",
        "CC-BY-2.5",
        "CC-BY-3.0",
        "CC-BY-4.0",
        # Attribution-only, like CC BY, for databases rather than works. It
        # "explicitly include[s] commercial use", does "not exclude any field
        # of endeavour" and puts no share-alike condition on works produced
        # from the database (opendatacommons.org/licenses/by/1-0, read
        # 2026-09-20), so it fails none of the exclusions above. Its
        # obligation is attribution and keeping the notices intact, which the
        # manifest carries and the release honours. Not to be confused
        # with ODbL, which is share-alike and stays denied below.
        "ODC-By-1.0",
        # Turkish legislation and court decisions: no copyright under FSEK article 31.
        "LicenseRef-TR-FSEK-31",
        # Written by the lab. Released under CC BY 4.0 with HakemBench.
        "LicenseRef-ufakai-authored",
    }
)

AUTHORED_LICENSE_ID = "LicenseRef-ufakai-authored"

# Words that must not appear in the license string as shown on the source
# when the entry claims an allowed id. This catches a wrong id, for example
# "CC-BY-4.0" typed for a set whose card says "cc-by-nc-4.0". It is a
# consistency check on top of the allow-list, not a replacement for it.
_DENY_TOKENS: frozenset[str] = frozenset(
    {
        "nc",
        "noncommercial",
        "nd",
        "noderivatives",
        "noderivs",
        "sa",
        "sharealike",
        "research",
        "academic",
        "gpl",
        "agpl",
        "lgpl",
        "odbl",
        "proprietary",
        "unknown",
        "unlicensed",
        "other",
    }
)
_DENY_PHRASES: tuple[tuple[str, str], ...] = (
    ("non", "commercial"),
    ("share", "alike"),
    ("no", "derivatives"),
    ("no", "derivs"),
    ("all", "rights"),
)


def deny_markers(license_as_shown: str) -> list[str]:
    """Return the denied terms found in a license string, in order of appearance."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", license_as_shown.lower()) if t]
    found = [t for t in tokens if t in _DENY_TOKENS]
    found += [
        f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False) if (a, b) in _DENY_PHRASES
    ]
    return found
